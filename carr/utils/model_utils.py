import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_mixtral_4bit(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    torch_dtype: str = "float16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
):
    """
    Load Mixtral 8x7B in 4-bit quantization using bitsandbytes.

    Returns:
        (model, tokenizer)
    """
    dtype = getattr(torch, torch_dtype, torch.float16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model: {model_name} (4-bit, dtype={torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    logger.info(
        f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params"
    )
    return model, tokenizer


def print_trainable_summary(model):
    """Print a summary of trainable vs frozen parameters."""
    trainable = 0
    frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable += p.numel()
            print(f"  [TRAIN] {name:60s} {tuple(p.shape)}")
        else:
            frozen += p.numel()

    print(f"\nTrainable : {trainable:>12,}")
    print(f"Frozen    : {frozen:>12,}")
    print(f"Total     : {trainable + frozen:>12,}")
    print(f"Trainable%: {100 * trainable / (trainable + frozen):.4f}%")
