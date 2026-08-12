"""DSpark checkpoint contract validation and L1 comparison.

DSpark is a speculative *draft* model.  Unlike a normal CausalLM it consumes
hidden states exported by its verifier/target model.  Keeping that distinction
explicit prevents a draft checkpoint from silently entering the normal
embedding -> decoder -> lm_head path with meaningless synthetic inputs.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .block_compare_types import BlockCompareReport, BlockCompareResult
from .metrics import compute_all_metrics


logger = logging.getLogger(__name__)

_DEEPSPEC_ARCHITECTURES = {"Qwen3DSparkModel", "Gemma4DSparkModel"}
_SPECULATORS_ARCHITECTURES = {
    "DSparkDraftModel",
    "DSparkSpeculator",
}
_VLLM_DSPARK_ARCHITECTURES = {"K3DSparkModel"}
_CONTRACT_FIELDS = (
    "flavor",
    "architecture",
    "block_size",
    "num_anchors",
    "draft_layers",
    "target_layer_ids",
    "hidden_size",
    "vocab_size",
    "mask_token_id",
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "sample_from_anchor",
    "verifier_model",
)


def _first(mapping: Mapping[str, Any], *names: str, default=None):
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def _as_tuple(value) -> Tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    return tuple(int(item) for item in value)


def _as_int(value) -> Optional[int]:
    return None if value is None else int(value)


@dataclass(frozen=True)
class DSparkContract:
    """Normalized parameter contract shared by DeepSpec and Speculators."""

    flavor: str
    architecture: str
    block_size: int
    num_anchors: Optional[int]
    draft_layers: int
    target_layer_ids: Tuple[int, ...]
    hidden_size: int
    vocab_size: int
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    enable_confidence_head: bool
    confidence_head_with_markov: bool
    sample_from_anchor: bool
    verifier_model: Optional[str] = None
    backbone_parameters: Tuple[Tuple[str, Any], ...] = ()

    @property
    def target_hidden_width(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    def summary(self) -> str:
        return (
            f"flavor={self.flavor}, arch={self.architecture}, "
            f"block={self.block_size}, anchors={self.num_anchors}, "
            f"draft_layers={self.draft_layers}, "
            f"target_layers={list(self.target_layer_ids)}, hidden={self.hidden_size}, "
            f"vocab={self.vocab_size}, markov={self.markov_head_type}:{self.markov_rank}, "
            f"confidence={self.enable_confidence_head}, verifier={self.verifier_model}"
        )


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_config_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _backbone_parameters(
    config: Mapping[str, Any],
    transformer: Mapping[str, Any],
) -> Tuple[Tuple[str, Any], ...]:
    """Capture structural fields that must not drift between ref and quant."""
    names = (
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "hidden_act",
        "attention_bias",
        "attention_dropout",
        "layer_types",
        "sliding_window",
        "max_window_layers",
        "max_position_embeddings",
        "rope_theta",
        "rope_parameters",
        "rope_scaling",
        "target_hidden_size",
        "target_num_hidden_layers",
        "num_target_layers",
        "apply_verifier_norm",
        "use_draft_vocab",
    )
    parameters = []
    for name in names:
        value = _first(config, name, default=_first(transformer, name))
        if value is not None:
            parameters.append((name, _freeze_config_value(value)))
    return tuple(parameters)


def is_dspark_config(config: Mapping[str, Any]) -> bool:
    """Return whether a raw ``config.json`` describes a standalone DSpark draft."""
    architectures = set(config.get("architectures") or [])
    spec_cfg = config.get("speculators_config") or {}
    algorithm = spec_cfg.get("algorithm") if isinstance(spec_cfg, Mapping) else None
    return bool(
        architectures
        & (
            _DEEPSPEC_ARCHITECTURES
            | _SPECULATORS_ARCHITECTURES
            | _VLLM_DSPARK_ARCHITECTURES
        )
        or str(config.get("speculators_model_type", "")).lower() == "dspark"
        or str(algorithm or "").lower() == "dspark"
        or str(config.get("model_type", "")).lower()
        in {"dspark", "qwen3_dspark", "k3_dspark"}
    )


def _classify_dspark_config(config: Mapping[str, Any], architecture: str) -> str:
    """Identify the incompatible checkpoint ecosystems sharing the DSpark name."""
    model_type = str(config.get("model_type", "")).lower()
    if architecture in _VLLM_DSPARK_ARCHITECTURES or model_type == "k3_dspark":
        return "vllm_k3"
    if config.get("_torchspec_version") or model_type == "qwen3_dspark":
        return "torchspec"
    if architecture in _DEEPSPEC_ARCHITECTURES:
        return "deepspec"
    if (
        str(config.get("speculators_model_type", "")).lower() == "dspark"
        or isinstance(config.get("speculators_config"), Mapping)
        or architecture == "DSparkSpeculator"
    ):
        return "speculators"
    if architecture in _SPECULATORS_ARCHITECTURES:
        return "specforge_remote"
    return "unknown"


def is_dspark_checkpoint(model_path: Optional[str]) -> bool:
    if not model_path or not os.path.isdir(model_path):
        return False
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return is_dspark_config(json.load(handle))
    except (OSError, ValueError, TypeError):
        return False


def _resolve_verifier_model(config: Mapping[str, Any]) -> Optional[str]:
    spec_cfg = config.get("speculators_config") or {}
    if not isinstance(spec_cfg, Mapping):
        return None
    verifier = spec_cfg.get("verifier") or {}
    if isinstance(verifier, Mapping):
        return _first(verifier, "name_or_path", "model", "model_name_or_path")
    return verifier if isinstance(verifier, str) else None


def normalize_dspark_contract(config: Mapping[str, Any]) -> DSparkContract:
    """Normalize a DSpark ``config.json`` and validate required parameters."""
    if not is_dspark_config(config):
        raise ValueError("config.json is not a standalone DSpark checkpoint")

    architectures = list(config.get("architectures") or [])
    architecture = architectures[0] if architectures else "DSparkDraftModel"
    flavor = _classify_dspark_config(config, architecture)
    if flavor not in {"deepspec", "speculators"}:
        details = {
            "vllm_k3": (
                "K3DSparkModel uses vLLM's native MLA DSpark runtime and its "
                "block_size comes from speculative_config.num_speculative_tokens"
            ),
            "torchspec": (
                "TorchSpec training checkpoints require the TorchSpec wrapper/API"
            ),
            "specforge_remote": (
                "remote-code SpecForge DSparkDraftModel uses deployment-time "
                "noise embeddings and a different forward contract"
            ),
            "unknown": "the checkpoint does not expose a supported DSpark API marker",
        }.get(flavor, "the checkpoint uses an unsupported DSpark API")
        raise NotImplementedError(
            f"DSpark checkpoint flavor {flavor!r} is recognized but is not yet "
            f"runnable in acc_bench L1: {details}. Use --mode boundary with the "
            "verifier/target model plus --framework_bad_output for deployment "
            "boundary testing."
        )
    transformer = config.get("transformer_layer_config") or {}
    if not isinstance(transformer, Mapping):
        transformer = {}
    spec_cfg = config.get("speculators_config") or {}
    if not isinstance(spec_cfg, Mapping):
        spec_cfg = {}

    target_layer_ids = _as_tuple(_first(
        config,
        "target_layer_ids",
        "dspark_target_layer_ids",
        "aux_hidden_state_layer_ids",
        default=_first(spec_cfg, "target_layer_ids", "aux_hidden_state_layer_ids"),
    ))
    block_size = _as_int(_first(
        config, "block_size", "n_predict", "num_lookahead_tokens",
        default=_first(spec_cfg, "block_size", "num_speculative_tokens"),
    ))
    num_anchors = _as_int(_first(config, "num_anchors", "max_anchors"))
    draft_layers = _as_int(_first(
        config, "num_layers", "num_hidden_layers",
        default=_first(transformer, "num_hidden_layers"),
    ))
    hidden_size = _as_int(_first(
        config, "hidden_size", default=_first(transformer, "hidden_size")
    ))
    vocab_size = _as_int(_first(
        config, "draft_vocab_size", "vocab_size",
        default=_first(transformer, "vocab_size"),
    ))
    mask_token_id = _as_int(_first(
        config, "mask_token_id", default=_first(transformer, "mask_token_id")
    ))

    missing = []
    for field_name, value in (
        ("block_size/n_predict", block_size),
        ("num_hidden_layers/num_layers", draft_layers),
        ("target_layer_ids/aux_hidden_state_layer_ids", target_layer_ids),
        ("hidden_size", hidden_size),
        ("vocab_size/draft_vocab_size", vocab_size),
        ("mask_token_id", mask_token_id),
    ):
        if value is None or value == ():
            missing.append(field_name)
    if missing:
        raise ValueError("DSpark config missing required fields: " + ", ".join(missing))
    if flavor == "deepspec" and num_anchors is None:
        raise ValueError("DeepSpec DSpark config missing required field: num_anchors")
    verifier_model = _resolve_verifier_model(config)
    if flavor == "speculators" and not verifier_model:
        raise ValueError(
            "Speculators DSpark config missing required field: "
            "speculators_config.verifier.name_or_path"
        )

    markov_rank = int(config.get("markov_rank", 0))
    confidence = bool(config.get("enable_confidence_head", False))
    confidence_with_markov = bool(config.get("confidence_head_with_markov", False))
    if confidence_with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov=True requires markov_rank > 0"
        )
    if int(block_size) <= 0 or int(draft_layers) <= 0 or int(hidden_size) <= 0:
        raise ValueError("DSpark block_size, draft_layers and hidden_size must be positive")
    if int(vocab_size) <= 0:
        raise ValueError("DSpark vocab_size must be positive")
    if num_anchors is not None and num_anchors <= 0:
        raise ValueError("DSpark num_anchors must be positive")
    if any(layer_id < -1 for layer_id in target_layer_ids):
        raise ValueError("DSpark target_layer_ids cannot contain IDs below -1")
    if any(
        right <= left
        for left, right in zip(target_layer_ids, target_layer_ids[1:])
    ):
        raise ValueError("DSpark target_layer_ids must be strictly increasing")
    # DeepSpec uses num_target_layers for the verifier depth. Other ecosystems
    # also use that name for the number of exported auxiliary states, so do not
    # apply an invalid upper-bound check to standard Speculators configs.
    num_target_layers = (
        _as_int(_first(config, "num_target_layers"))
        if flavor == "deepspec"
        else None
    )
    non_embedding_ids = [layer_id for layer_id in target_layer_ids if layer_id >= 0]
    if (
        num_target_layers is not None
        and non_embedding_ids
        and max(non_embedding_ids) >= num_target_layers
    ):
        raise ValueError(
            "DSpark target_layer_ids exceeds num_target_layers: "
            f"max={max(non_embedding_ids)}, num_target_layers={num_target_layers}"
        )

    return DSparkContract(
        flavor=flavor,
        architecture=architecture,
        block_size=int(block_size),
        num_anchors=num_anchors,
        draft_layers=int(draft_layers),
        target_layer_ids=target_layer_ids,
        hidden_size=int(hidden_size),
        vocab_size=int(vocab_size),
        mask_token_id=int(mask_token_id),
        markov_rank=markov_rank,
        markov_head_type=str(config.get("markov_head_type", "none")),
        enable_confidence_head=confidence,
        confidence_head_with_markov=confidence_with_markov,
        sample_from_anchor=bool(config.get("sample_from_anchor", True)),
        verifier_model=verifier_model,
        backbone_parameters=_backbone_parameters(config, transformer),
    )


def load_dspark_contract(model_path: str) -> DSparkContract:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        return normalize_dspark_contract(json.load(handle))


def validate_dspark_pair(
    ref_contract: DSparkContract,
    quant_contract: DSparkContract,
) -> None:
    """Fail before model loading when ref and quant draft contracts differ."""
    mismatches = []
    for name in _CONTRACT_FIELDS:
        ref_value = getattr(ref_contract, name)
        quant_value = getattr(quant_contract, name)
        if ref_value != quant_value:
            mismatches.append(f"{name}: ref={ref_value!r}, quant={quant_value!r}")
    ref_backbone = dict(ref_contract.backbone_parameters)
    quant_backbone = dict(quant_contract.backbone_parameters)
    for name in sorted(set(ref_backbone) | set(quant_backbone)):
        ref_value = ref_backbone.get(name)
        quant_value = quant_backbone.get(name)
        if ref_value != quant_value:
            mismatches.append(
                f"backbone.{name}: ref={ref_value!r}, quant={quant_value!r}"
            )
    if mismatches:
        raise ValueError("DSpark ref/quant parameter mismatch:\n  " + "\n  ".join(mismatches))


@dataclass
class DSparkSample:
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    loss_mask: torch.Tensor
    verifier_last_hidden_states: Optional[torch.Tensor] = None
    document_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None

    def to(self, device: str, dtype: torch.dtype) -> "DSparkSample":
        return DSparkSample(
            input_ids=self.input_ids.to(device=device, dtype=torch.long),
            hidden_states=self.hidden_states.to(device=device, dtype=dtype),
            loss_mask=self.loss_mask.to(device=device),
            verifier_last_hidden_states=(
                None if self.verifier_last_hidden_states is None
                else self.verifier_last_hidden_states.to(device=device, dtype=dtype)
            ),
            document_ids=(
                None if self.document_ids is None
                else self.document_ids.to(device=device, dtype=torch.long)
            ),
            position_ids=(
                None if self.position_ids is None
                else self.position_ids.to(device=device, dtype=torch.long)
            ),
        )


def _load_torch_payload(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_field(payload: Mapping[str, Any], *names: str):
    value = _first(payload, *names)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def load_dspark_sample(path: str, contract: DSparkContract) -> DSparkSample:
    """Load one verifier hidden-state sample from a ``.pt`` cache file.

    Supported field aliases match both official DeepSpec data and the
    Speculators data pipeline.  A 4-D hidden tensor ``[B,S,N,H]`` is flattened
    to the model input convention ``[B,S,N*H]``.
    """
    payload = _load_torch_payload(path)
    if isinstance(payload, (list, tuple)):
        if not payload:
            raise ValueError("DSpark sample file is empty")
        payload = payload[0]
    if isinstance(payload, Mapping) and isinstance(payload.get("sample"), Mapping):
        payload = payload["sample"]
    if not isinstance(payload, Mapping):
        raise TypeError("DSpark sample must be a dict (or a list containing one dict)")

    input_ids = _tensor_field(payload, "input_ids", "tokens")
    hidden_states = _tensor_field(
        payload, "hidden_states", "target_hidden_states", "aux_hidden_states"
    )
    loss_mask = _tensor_field(payload, "loss_mask", "attention_loss_mask")
    verifier_last = _tensor_field(
        payload, "verifier_last_hidden_states", "target_last_hidden_states",
        "last_hidden_states",
    )
    document_ids = _tensor_field(payload, "document_ids", "sequence_ids")
    position_ids = _tensor_field(payload, "position_ids")
    missing = [
        name for name, value in (
            ("input_ids", input_ids),
            ("hidden_states/target_hidden_states", hidden_states),
            ("loss_mask", loss_mask),
        ) if value is None
    ]
    if missing:
        raise ValueError("DSpark sample missing required fields: " + ", ".join(missing))

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if loss_mask.ndim == 1:
        loss_mask = loss_mask.unsqueeze(0)
    if input_ids.ndim != 2 or loss_mask.ndim != 2:
        raise ValueError("DSpark input_ids and loss_mask must be [B,S]")
    if (
        hidden_states.ndim == 3
        and input_ids.shape[0] == 1
        and tuple(hidden_states.shape[:2]) != tuple(input_ids.shape[:2])
        and hidden_states.shape[0] == input_ids.shape[1]
    ):
        # Common unbatched DeepSpec export: [S,N,H].
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.flatten(start_dim=-2)
    if hidden_states.ndim != 3:
        raise ValueError(
            "DSpark hidden_states must be [B,S,N*H] or [B,S,N,H], "
            f"got {tuple(hidden_states.shape)}"
        )
    if verifier_last is not None and verifier_last.ndim == 2:
        verifier_last = verifier_last.unsqueeze(0)
    if verifier_last is not None and verifier_last.ndim != 3:
        raise ValueError("DSpark verifier_last_hidden_states must be [B,S,H]")

    batch_seq = tuple(input_ids.shape[:2])
    if tuple(hidden_states.shape[:2]) != batch_seq:
        raise ValueError(
            f"DSpark input/hidden shape mismatch: input={tuple(input_ids.shape)}, "
            f"hidden={tuple(hidden_states.shape)}"
        )
    if tuple(loss_mask.shape[:2]) != batch_seq:
        raise ValueError(
            f"DSpark input/loss_mask shape mismatch: input={tuple(input_ids.shape)}, "
            f"loss_mask={tuple(loss_mask.shape)}"
        )
    if contract.flavor == "speculators" and input_ids.shape[0] != 1:
        raise ValueError(
            "Speculators DSpark training forward currently requires batch size 1"
        )
    if hidden_states.shape[-1] != contract.target_hidden_width:
        raise ValueError(
            "DSpark hidden-state width mismatch: "
            f"expected {contract.target_hidden_width} "
            f"({len(contract.target_layer_ids)} target layers x {contract.hidden_size}), "
            f"got {hidden_states.shape[-1]}"
        )
    if verifier_last is not None:
        if tuple(verifier_last.shape[:2]) != batch_seq:
            raise ValueError("DSpark verifier_last_hidden_states batch/sequence mismatch")
        if verifier_last.shape[-1] != contract.hidden_size:
            raise ValueError(
                "DSpark verifier_last_hidden_states width mismatch: "
                f"expected {contract.hidden_size}, got {verifier_last.shape[-1]}"
            )
    if document_ids is None:
        document_ids = torch.zeros_like(input_ids, dtype=torch.long)
    elif document_ids.ndim == 1:
        document_ids = document_ids.unsqueeze(0)
    if document_ids.ndim != 2:
        raise ValueError("DSpark document_ids must be [B,S]")
    if tuple(document_ids.shape[:2]) != batch_seq:
        raise ValueError("DSpark document_ids batch/sequence mismatch")
    if position_ids is not None:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim != 2:
            raise ValueError("DSpark position_ids must be [B,S]")
        if tuple(position_ids.shape[:2]) != batch_seq:
            raise ValueError("DSpark position_ids batch/sequence mismatch")

    return DSparkSample(
        input_ids=input_ids.long(),
        hidden_states=hidden_states,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last,
        document_ids=document_ids.long(),
        position_ids=position_ids,
    )


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _seed_device(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if str(device).startswith("npu") and hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:  # pragma: no cover - depends on torch_npu version
            pass
    elif str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_forward_kwargs(
    model,
    sample: DSparkSample,
    max_anchors: int,
) -> Dict[str, Any]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "target_hidden_states" in parameters:
        if hasattr(model, "num_anchors"):
            model.num_anchors = int(max_anchors)
        kwargs = {
            "input_ids": sample.input_ids,
            "target_hidden_states": sample.hidden_states,
            "loss_mask": sample.loss_mask,
        }
        if sample.verifier_last_hidden_states is not None:
            kwargs["target_last_hidden_states"] = sample.verifier_last_hidden_states
        return kwargs
    if "hidden_states" in parameters and "verifier_last_hidden_states" in parameters:
        if sample.verifier_last_hidden_states is None:
            raise ValueError(
                "Speculators DSpark sample requires verifier_last_hidden_states; "
                "export it together with auxiliary hidden_states"
            )
        kwargs = {
            "hidden_states": sample.hidden_states,
            "input_ids": sample.input_ids,
            "loss_mask": sample.loss_mask,
            "verifier_last_hidden_states": sample.verifier_last_hidden_states,
            "document_ids": sample.document_ids,
            "max_anchors": int(max_anchors),
        }
        if sample.position_ids is not None:
            kwargs["position_ids"] = sample.position_ids
        return kwargs
    raise TypeError(
        f"Unsupported DSpark forward contract on {type(model).__name__}; expected "
        "DeepSpec (target_hidden_states) or Speculators (hidden_states + "
        "verifier_last_hidden_states) API"
    )


def _capture_model_outputs(
    model,
    sample: DSparkSample,
    seed: int,
    device: str,
    max_anchors: int,
):
    captures: Dict[str, torch.Tensor] = {}
    hooks = []

    def register(name, module):
        if module is None:
            return

        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                captures[name] = tensor.detach().float().cpu()

        hooks.append(module.register_forward_hook(hook))

    layers = getattr(model, "layers", None)
    if layers is None:
        try:
            from .utils import get_decoder_layers
            layers = get_decoder_layers(model)
        except ValueError:
            layers = []
    for index, layer in enumerate(layers):
        register(f"layer.{index}.dspark", layer)
    for name in (
        "embed_tokens", "fc", "hidden_norm", "verifier_norm", "norm",
        "lm_head", "markov_head", "confidence_head",
    ):
        register(f"dspark.{name}", getattr(model, name, None))
    # Both DeepSpec and Speculators call Markov/confidence helper methods
    # directly, so a hook on the parent module may not fire.  Capture their
    # actual embedding/projection submodules as well.
    for head_name in ("markov_head", "confidence_head"):
        head = getattr(model, head_name, None)
        if head is None:
            continue
        for sub_name, sub_module in head.named_modules():
            if sub_name:
                register(f"dspark.{head_name}.{sub_name}", sub_module)

    _seed_device(seed, device)
    try:
        with torch.no_grad():
            output = model(**_build_forward_kwargs(model, sample, max_anchors))
        for name in (
            "draft_logits", "confidence_pred", "confidence_logits",
            "aligned_target_logits",
        ):
            tensor = _first_tensor(getattr(output, name, None))
            if tensor is not None:
                captures[f"dspark.{name}"] = tensor.detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return captures


class DSparkComparator:
    """Compare standalone DSpark ref/quant checkpoints on verifier cache data."""

    def __init__(
        self,
        ref_model_path: str,
        quant_model_path: str,
        sample_path: str,
        ref_device: str = "npu:0",
        quant_device: str = "npu:1",
        dtype: torch.dtype = torch.bfloat16,
        quant_method: str = "dequantize",
        seed: int = 0,
        max_anchors: int = 8,
        verbose: bool = True,
    ):
        self.ref_model_path = ref_model_path
        self.quant_model_path = quant_model_path
        self.ref_device = ref_device
        self.quant_device = quant_device
        self.dtype = dtype
        if quant_method not in {"dequantize", "fake_quant"}:
            raise ValueError(
                "DSpark quant_method must be 'dequantize' or 'fake_quant'"
            )
        self.quant_method = quant_method
        self.seed = int(seed)
        if int(max_anchors) <= 0:
            raise ValueError("DSpark max_anchors must be positive")
        self.max_anchors = int(max_anchors)
        self.verbose = verbose
        self.ref_contract = load_dspark_contract(ref_model_path)
        self.quant_contract = load_dspark_contract(quant_model_path)
        validate_dspark_pair(self.ref_contract, self.quant_contract)
        self.sample = load_dspark_sample(sample_path, self.ref_contract)
        valid = (
            (self.sample.loss_mask[:, :-1] > 0.5)
            & (self.sample.loss_mask[:, 1:] > 0.5)
        )
        min_valid_anchors = int(valid.sum(dim=1).min().item())
        if min_valid_anchors <= 0:
            raise ValueError("DSpark sample contains no valid anchor/target pair")
        limits = [self.max_anchors, min_valid_anchors]
        if self.ref_contract.num_anchors is not None:
            limits.append(self.ref_contract.num_anchors)
        self.max_anchors = min(limits)

    def compare(self) -> BlockCompareReport:
        from .model_loader import load_model_for_comparison
        from .utils import clear_device_cache

        if self.verbose:
            logger.info("[DSpark L1] parameter contract aligned")
            logger.info("  %s", self.ref_contract.summary())
            logger.info("  verifier cache: input=%s hidden=%s", tuple(self.sample.input_ids.shape), tuple(self.sample.hidden_states.shape))
            logger.info("  runtime max_anchors: %d", self.max_anchors)

        ref_model = None
        quant_model = None
        ref_sample = None
        quant_sample = None
        try:
            ref_model, _ = load_model_for_comparison(
                self.ref_model_path,
                device=self.ref_device,
                dtype=self.dtype,
                use_fake_quant=False,
            )
            quant_model, _ = load_model_for_comparison(
                self.quant_model_path,
                device=self.quant_device,
                dtype=self.dtype,
                use_fake_quant=(self.quant_method == "fake_quant"),
            )
            ref_model.eval()
            quant_model.eval()

            ref_sample = self.sample.to(self.ref_device, self.dtype)
            quant_sample = self.sample.to(self.quant_device, self.dtype)
            ref_outputs = _capture_model_outputs(
                ref_model, ref_sample, self.seed, self.ref_device, self.max_anchors
            )
            quant_outputs = _capture_model_outputs(
                quant_model, quant_sample, self.seed, self.quant_device,
                self.max_anchors,
            )
        finally:
            ref_model = None
            quant_model = None
            ref_sample = None
            quant_sample = None
            clear_device_cache([self.ref_device, self.quant_device])

        missing_ref = sorted(set(quant_outputs) - set(ref_outputs))
        missing_quant = sorted(set(ref_outputs) - set(quant_outputs))
        if missing_ref or missing_quant:
            raise RuntimeError(
                "DSpark ref/quant runtime output mismatch: "
                f"missing_ref={missing_ref}, missing_quant={missing_quant}"
            )

        report = BlockCompareReport(
            comparison_scope="weight_only",
            quant_method=self.quant_method,
            activation_quant_enabled=False,
        )
        for name, ref_value in ref_outputs.items():
            quant_value = quant_outputs[name]
            if ref_value.shape != quant_value.shape:
                raise RuntimeError(
                    f"DSpark output shape mismatch at {name}: "
                    f"ref={tuple(ref_value.shape)}, quant={tuple(quant_value.shape)}"
                )
            report.results.append(BlockCompareResult(
                layer_name=name,
                metrics=compute_all_metrics(ref_value, quant_value),
            ))
        if not report.results:
            raise RuntimeError("DSpark forward produced no comparable tensors")
        return report


__all__ = [
    "DSparkContract", "DSparkSample", "DSparkComparator",
    "is_dspark_config", "is_dspark_checkpoint", "normalize_dspark_contract",
    "load_dspark_contract", "validate_dspark_pair", "load_dspark_sample",
]
