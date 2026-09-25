"""Deterministic loading for non-quantized causal language model baselines."""

from typing import Literal, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from huyawo_quant.contracts import ModelIdentity, TokenizerIdentity

BaselineDtype = Literal["bfloat16", "float16"]


def _resolve_torch_dtype(dtype: BaselineDtype) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16

    if dtype == "float16":
        return torch.float16

    raise ValueError(f"Unsupported baseline dtype: {dtype!r}")


def _resolve_cuda_device(cuda_device_index: int) -> torch.device:
    if isinstance(cuda_device_index, bool) or not isinstance(cuda_device_index, int):
        raise TypeError("cuda_device_index must be an integer")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_count = torch.cuda.device_count()

    if cuda_device_index < 0 or cuda_device_index >= device_count:
        raise ValueError(
            "Invalid CUDA device index: "
            f"{cuda_device_index}; available device count: {device_count}"
        )

    return torch.device("cuda", cuda_device_index)


def load_baseline(
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
    *,
    dtype: BaselineDtype,
    cuda_device_index: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load one immutable baseline model and tokenizer on an explicit CUDA device."""
    if not isinstance(model_identity, ModelIdentity):
        raise TypeError("model_identity must be a ModelIdentity")

    if not isinstance(tokenizer_identity, TokenizerIdentity):
        raise TypeError("tokenizer_identity must be a TokenizerIdentity")

    torch_dtype = _resolve_torch_dtype(dtype)
    device = _resolve_cuda_device(cuda_device_index)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_identity.tokenizer_id,
        revision=tokenizer_identity.resolved_revision,
        token=False,
        trust_remote_code=False,
    )

    if tokenizer is None:
        raise RuntimeError("Tokenizer loading returned None")

    model = AutoModelForCausalLM.from_pretrained(
        model_identity.model_id,
        revision=model_identity.resolved_revision,
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch_dtype,
    )

    module = cast(torch.nn.Module, model)
    module.to(device)
    module.eval()

    return model, tokenizer
