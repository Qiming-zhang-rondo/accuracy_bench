from unittest.mock import patch

import pytest
import torch

hf = pytest.importorskip('transformers.models.qwen4_exp.modeling_qwen4_exp')
from tests.test_qwen4_exp import checkpoint
from accuracy_checker import qwen4_tp as tp


@pytest.fixture(autouse=True)
def reset_runtime():
    tp.install_query_parallel('pp', query_block=64, min_tokens=1024)
    yield
    tp.install_query_parallel('pp', query_block=64, min_tokens=1024)


def data(model, length, dtype, padding=False):
    model.to(dtype)
    idx = model.model.layers[1].self_attn.indexer.eval()
    h = torch.randn(2, length, model.config.hidden_size, dtype=dtype)
    positions = torch.arange(length)[None].expand(2, -1)
    pe = model.model.rotary_emb(h, positions)
    mask = torch.ones(2, 1, length, length, dtype=torch.bool).tril()
    if padding:
        mask[0, :, :, :2] = False  # left padding
        mask[1, :, :, -2:] = False  # right padding
        if length > 3:
            mask[1, :, :, 3] = False  # non-contiguous key-valid list
    return idx, h, pe, mask


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('length,block,padding', [(7, 2, False), (13, 4, True), (3, 8, True)])
@pytest.mark.parametrize('boolean', [False, True])
def test_qsa_mask_exact(checkpoint, dtype, length, block, padding, boolean):
    _, model = checkpoint
    idx, h, pe, mask = data(model, length, dtype, padding)
    if not boolean:
        mask = torch.where(mask, torch.tensor(0, dtype=dtype), torch.finfo(dtype).min)
    with torch.no_grad():
        expected = idx(h, pe, mask, None)
        tp.install_query_parallel('tp', [['cpu', 'cpu:0', 'cpu:1']], query_block=block, min_tokens=1)
        actual = idx(h, pe, mask, None)
    assert torch.equal(actual, expected)


def test_ties_and_chunked_prefill_cache(checkpoint):
    _, model = checkpoint
    idx, h, pe, mask = data(model, 13, torch.float32)
    idx.index_qk_proj.weight.data.zero_()
    # Simulate a preceding cached prefix; each runtime updates once.
    class Cache:
        def __init__(self):
            self.calls = 0
        def update_indexer(self, keys, layer):
            self.calls += 1
            return torch.cat((torch.zeros(2, 4, keys.shape[-1]), keys), dim=1)
    ref_cache, tp_cache = Cache(), Cache()
    with torch.no_grad():
        expected = idx(h[:, 4:], pe, mask[:, :, 4:], ref_cache)
        tp.install_query_parallel('tp', [['cpu', 'cpu:0']], query_block=3, min_tokens=1)
        actual = idx(h[:, 4:], pe, mask[:, :, 4:], tp_cache)
    assert torch.equal(actual, expected)
    assert ref_cache.calls == tp_cache.calls == 1


def test_custom_mask_single_device_and_decode_fall_back(checkpoint):
    _, model = checkpoint
    idx, h, pe, mask = data(model, 8, torch.float32)
    mask[0, 0, 4, 0] = False  # not a shared key-padding rule
    with torch.no_grad():
        expected = idx(h, pe, mask, None)
        tp.install_query_parallel('tp', [['cpu', 'cpu:0']], min_tokens=1)
        with patch.object(tp, '_query_parallel_indices', side_effect=AssertionError('must fall back')):
            assert torch.equal(idx(h, pe, mask, None), expected)
            idx(h[:, :1], tuple(p[:, :1] for p in pe), mask[:, :, :1, :1], None)
        tp.install_query_parallel('tp', [['cpu']], min_tokens=1)
        with patch.object(tp, '_query_parallel_indices', side_effect=AssertionError('single device')):
            assert torch.equal(idx(h, pe, mask, None), expected)


def test_transport_and_sync_are_per_device_not_per_query(checkpoint):
    _, model = checkpoint
    idx, h, pe, mask = data(model, 19, torch.bfloat16)
    moves, syncs = [], []
    real_copy = tp._copy_to
    def spy(tensor, device):
        moves.append((tuple(tensor.shape), tensor.dtype, str(device)))
        return real_copy(tensor, device)
    tp.install_query_parallel('tp', [['cpu', 'cpu:0']], query_block=4, min_tokens=1)
    with torch.no_grad(), patch.object(tp, '_copy_to', side_effect=spy), patch.object(tp, '_synchronise', side_effect=lambda d: syncs.append(str(d))):
        idx(h, pe, mask, None)
    q_moves = [shape for shape, dtype, _ in moves if len(shape) == 4]
    assert len(q_moves) == 5
    assert all(shape[1] <= 4 for shape in q_moves)
    assert all(dtype != torch.float32 for _, dtype, _ in moves)
    # Two batches of compressed keys, once on each worker, plus int32 ID metadata.
    k_moves = [(shape, device) for shape, dtype, device in moves if len(shape) == 2 and dtype == torch.bfloat16]
    assert len(k_moves) == 4
    assert len(syncs) == 5  # owner readiness + 2 workers + gather + final mask


def test_round_robin_assignments_bounded():
    for length in (65536, 10000):
        groups = tp._query_block_assignments(length, 1024, 8)
        blocks = sorted(block for group in groups for block in group)
        assert blocks[0][0] == 0 and blocks[-1][1] == length
        assert all(0 < end-start <= 1024 for start, end in blocks)
        assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
        assert max(map(len, groups)) - min(map(len, groups)) <= 1


def test_model_logits_and_decode_with_tp_prefill(checkpoint):
    _, model = checkpoint
    ids = torch.tensor([[3, 4, 7, 2, 8, 9, 5]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        ref = model(ids, use_cache=True)
        expected_decode = model(torch.tensor([[6]]), past_key_values=ref.past_key_values).logits
        tp.install_query_parallel('tp', [['cpu', 'cpu:0']], query_block=3, min_tokens=1)
        actual = model(ids, use_cache=False).logits
        cached = model(ids, use_cache=True)
        actual_decode = model(torch.tensor([[6]]), past_key_values=cached.past_key_values).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=0, atol=0)


def test_workers_are_launched_concurrently(checkpoint):
    import threading
    _, model = checkpoint
    idx, h, pe, mask = data(model, 8, torch.float32)
    barrier = threading.Barrier(2)
    original = tp._query_worker
    seen = set()
    def worker(*args):
        seen.add(threading.get_ident())
        barrier.wait(timeout=10)  # Sequential invocation cannot pass.
        return original(*args)
    tp.install_query_parallel('tp', [['cpu', 'cpu:0']], query_block=2, min_tokens=1)
    with torch.no_grad(), patch.object(tp, '_query_worker', side_effect=worker):
        idx(h, pe, mask, None)
    assert len(seen) == 2


def test_real_two_accelerators(checkpoint):
    # Run this on A3, not merely CPU aliases: exercises P2P copies, independent
    # streams, producer/consumer readiness and final owner gather.
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    backend = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda'
    api = getattr(torch, backend)
    if not api.is_available() or api.device_count() < 2:
        pytest.skip('requires two real NPUs or CUDA devices')
    _, model = checkpoint
    idx, h, pe, mask = data(model, 33, torch.bfloat16, padding=True)
    devices = [f'{backend}:0', f'{backend}:1']
    idx.to(devices[0])
    h, mask = h.to(devices[0]), mask.to(devices[0])
    pe = tuple(t.to(devices[0]) for t in pe)
    with torch.no_grad():
        expected = idx(h, pe, mask, None)
        tp.install_query_parallel('tp', [devices], query_block=8, min_tokens=1)
        actual = idx(h, pe, mask, None)
    assert torch.equal(actual, expected)
