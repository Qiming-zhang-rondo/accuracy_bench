"""UT for dual-level (coarse + fine) subgraph decomposition.

Verifies that mla_fine=True returns BOTH 'self_attn' (coarse) AND
'self_attn.o_proj' etc. (fine) in the subgraph list.
"""
import pytest
from unittest.mock import MagicMock


def _make_glm_moe_dsa_layer():
    """Create a mock GLM MoE DSA layer with MLA attention + MoE MLP."""
    layer = MagicMock()

    # self_attn with MLA projections
    attn = MagicMock()
    for attr in ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa',
                 'kv_b_proj', 'o_proj']:
        setattr(attn, attr, MagicMock())
    # indexer
    indexer = MagicMock()
    for attr in ['wq_b', 'wk', 'k_norm', 'weights_proj']:
        setattr(indexer, attr, MagicMock())
    attn.indexer = indexer
    layer.self_attn = attn

    # mlp (MoE)
    mlp = MagicMock()
    mlp.gate = MagicMock()
    mlp.shared_experts = MagicMock()
    mlp.experts = MagicMock()
    layer.mlp = mlp

    return layer


def test_mla_fine_includes_coarse_self_attn():
    """mla_fine=True should include 'self_attn' (coarse) alongside fine subgraphs."""
    from accuracy_checker.subgraph_locate import get_subgraph_names

    layer = _make_glm_moe_dsa_layer()
    names = get_subgraph_names('glm_moe_dsa', layer, mla_fine=True)

    # Coarse 'self_attn' must be present
    assert 'self_attn' in names, f"Coarse 'self_attn' missing from {names}"

    # Fine subgraphs must also be present
    assert 'self_attn.o_proj' in names, f"Fine 'self_attn.o_proj' missing from {names}"
    assert 'self_attn.q_a_proj' in names
    assert 'self_attn.q_b_proj' in names
    assert 'self_attn.kv_a_proj_with_mqa' in names
    assert 'self_attn.kv_b_proj' in names

    # MLP subgraphs
    assert 'mlp.gate' in names
    assert 'mlp.shared_experts' in names
    assert 'mlp.experts' in names

    # Coarse 'self_attn' should come BEFORE fine subgraphs
    idx_coarse = names.index('self_attn')
    idx_fine = names.index('self_attn.o_proj')
    assert idx_coarse < idx_fine, f"Coarse should come before fine: {names}"


def test_mla_fine_false_only_coarse():
    """mla_fine=False should only return coarse subgraphs."""
    from accuracy_checker.subgraph_locate import get_subgraph_names

    layer = _make_glm_moe_dsa_layer()
    names = get_subgraph_names('glm_moe_dsa', layer, mla_fine=False)

    assert names == ['self_attn', 'mlp.gate', 'mlp.shared_experts', 'mlp.experts']


def test_coarse_self_attn_not_in_inner_subgraphs():
    """'self_attn' (coarse) must NOT be in _MLA_ATTN_INNER_SUBGRAPHS,
    so it gets patch recovery in diagnose_layer."""
    from accuracy_checker.subgraph_locate import _MLA_ATTN_INNER_SUBGRAPHS

    assert 'self_attn' not in _MLA_ATTN_INNER_SUBGRAPHS
    assert 'self_attn.o_proj' not in _MLA_ATTN_INNER_SUBGRAPHS
    assert 'self_attn.q_a_proj' in _MLA_ATTN_INNER_SUBGRAPHS
