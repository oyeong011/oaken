#!/usr/bin/env python3
"""Sweep HF KV-cache modes and keep OOM rows as first-class results.

This is a measurement harness for long-context/big-batch capacity boundaries.
It is not an Oaken implementation. It measures the practical trade-off between
dynamic, quantized, offloaded, and no-cache generation paths.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CSV_COLUMNS = [
    "gpu",
    "model",
    "cache_mode",
    "batch_size",
    "seq_len",
    "dtype",
    "status",
    "oom",
    "peak_memory_mb",
    "allocated_memory_mb",
    "reserved_memory_mb",
    "kv_theory_mb",
    "kv_actual_mb",
    "kv_actual_over_theory",
    "non_kv_overhead_mb",
    "tokens_per_sec",
    "latency_ms",
    "generated_tokens",
    "error_message",
    "max_new_tokens",
    "base_allocated_mb",
    "peak_delta_memory_mb",
    "free_before_mb",
    "free_after_mb",
    "total_vram_mb",
    "max_position_embeddings",
    "kv_formula_type",
    "position_valid",
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
]


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int | None
    kv_formula_type: str


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}


def parse_values(values: list[str], cast: type = int) -> list[Any]:
    parsed: list[Any] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(cast(item))
    return parsed


def get_attr(config: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def infer_model_shape(config: Any) -> ModelShape:
    num_layers = int(get_attr(config, ["num_hidden_layers", "n_layer", "num_layers"], 0))
    num_attention_heads = int(get_attr(config, ["num_attention_heads", "n_head", "num_heads"], 0))
    num_key_value_heads = int(
        get_attr(config, ["num_key_value_heads", "num_kv_heads", "n_kv_heads"], num_attention_heads)
    )

    head_dim = get_attr(config, ["head_dim", "attention_head_dim"], None)
    if head_dim is None:
        hidden_size = get_attr(config, ["hidden_size", "n_embd", "d_model"], None)
        if hidden_size is None or not num_attention_heads:
            raise ValueError("Cannot infer head_dim from model config")
        head_dim = int(hidden_size) // num_attention_heads

    if not (num_layers and num_attention_heads and num_key_value_heads and head_dim):
        raise ValueError(f"Incomplete model shape in config: {config}")
    max_position_embeddings = get_attr(
        config,
        ["max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"],
        None,
    )
    kv_formula_type = "gqa_mqa" if num_key_value_heads != num_attention_heads else "mha"
    return ModelShape(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=int(head_dim),
        max_position_embeddings=None if max_position_embeddings is None else int(max_position_embeddings),
        kv_formula_type=kv_formula_type,
    )


def mb(value: float | int) -> float:
    return float(value) / (1024**2)


def theoretical_kv_bytes(shape: ModelShape, batch_size: int, seq_len: int, dtype_name: str) -> int:
    return int(
        2
        * shape.num_layers
        * batch_size
        * seq_len
        * shape.num_key_value_heads
        * shape.head_dim
        * DTYPE_BYTES[dtype_name]
    )


def position_valid(shape: ModelShape, seq_len: int) -> bool:
    if shape.max_position_embeddings is None:
        return True
    return seq_len <= shape.max_position_embeddings


def tensor_bytes(obj: Any) -> int:
    """Recursively sum tensor bytes in legacy tuples and newer HF Cache objects."""
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(value) for value in obj.values())

    total = 0
    for attr in ("key_cache", "value_cache"):
        if hasattr(obj, attr):
            total += tensor_bytes(getattr(obj, attr))
    for attr in ("_quantized_key_cache", "_quantized_value_cache"):
        if hasattr(obj, attr):
            total += tensor_bytes(getattr(obj, attr))
    if total:
        return total

    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return sum(tensor_bytes(item) for item in obj)
        except TypeError:
            return 0
    return 0


def make_inputs(tokenizer: Any, batch_size: int, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = tokenizer.pad_token_id
    if token_id is None:
        token_id = 0
    input_ids = torch.full((batch_size, seq_len), int(token_id), dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_mem_info() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def build_generate_kwargs(cache_mode: str, quant_backend: str, quant_nbits: int) -> dict[str, Any]:
    if cache_mode == "dynamic":
        return {"use_cache": True, "cache_implementation": "dynamic"}
    if cache_mode == "quantized":
        return {
            "use_cache": True,
            "cache_implementation": "quantized",
            "cache_config": {"backend": quant_backend, "nbits": quant_nbits},
        }
    if cache_mode == "offloaded":
        return {"use_cache": True, "cache_implementation": "offloaded"}
    if cache_mode == "no_cache":
        return {"use_cache": False}
    raise ValueError(f"Unsupported cache mode: {cache_mode}")


def prefill_kv_bytes(model: Any, inputs: dict[str, torch.Tensor]) -> int:
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    return int(tensor_bytes(getattr(outputs, "past_key_values", None)))


def empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def bench_one(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    gpu_name: str,
    dtype_name: str,
    cache_mode: str,
    batch_size: int,
    seq_len: int,
    max_new_tokens: int,
    shape: ModelShape,
    device: torch.device,
    warmup: bool,
    quant_backend: str,
    quant_nbits: int,
    include_traceback: bool,
) -> dict[str, Any]:
    base_allocated = int(torch.cuda.memory_allocated(device) if device.type == "cuda" else 0)
    kv_theory = theoretical_kv_bytes(shape, batch_size, seq_len, dtype_name)
    row: dict[str, Any] = {
        "gpu": gpu_name,
        "model": model_name,
        "cache_mode": cache_mode,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype": dtype_name,
        "status": "ok",
        "oom": False,
        "peak_memory_mb": math.nan,
        "allocated_memory_mb": math.nan,
        "reserved_memory_mb": math.nan,
        "kv_theory_mb": mb(kv_theory),
        "kv_actual_mb": math.nan,
        "kv_actual_over_theory": math.nan,
        "non_kv_overhead_mb": math.nan,
        "tokens_per_sec": math.nan,
        "latency_ms": math.nan,
        "generated_tokens": 0,
        "error_message": "",
        "max_new_tokens": max_new_tokens,
        "base_allocated_mb": mb(base_allocated),
        "peak_delta_memory_mb": math.nan,
        "free_before_mb": math.nan,
        "free_after_mb": math.nan,
        "total_vram_mb": math.nan,
        "position_valid": position_valid(shape, seq_len),
        **asdict(shape),
    }

    inputs: dict[str, torch.Tensor] | None = None
    try:
        inputs = make_inputs(tokenizer, batch_size, seq_len, device)
        actual_kv = prefill_kv_bytes(model, inputs)
        row["kv_actual_mb"] = mb(actual_kv)
        row["kv_actual_over_theory"] = actual_kv / kv_theory if kv_theory else math.nan

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)

        if warmup:
            warm_inputs = make_inputs(tokenizer, 1, min(seq_len, 32), device)
            with torch.inference_mode():
                model.generate(
                    **warm_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    **build_generate_kwargs(cache_mode, quant_backend, quant_nbits),
                )
            del warm_inputs
            synchronize(device)

        free_before, total_vram = cuda_mem_info()
        row["free_before_mb"] = mb(free_before)
        row["total_vram_mb"] = mb(total_vram)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                **build_generate_kwargs(cache_mode, quant_backend, quant_nbits),
            )
        synchronize(device)
        elapsed = time.perf_counter() - start

        generated_tokens = int(max(0, generated.shape[-1] - seq_len) * batch_size)
        row["generated_tokens"] = generated_tokens
        row["latency_ms"] = elapsed * 1000.0
        row["tokens_per_sec"] = generated_tokens / elapsed if elapsed > 0 else math.nan
        del generated
    except BaseException as exc:  # noqa: BLE001 - benchmark failures are data.
        row["oom"] = is_oom(exc)
        row["status"] = "oom" if row["oom"] else "error"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        if include_traceback:
            row["error_message"] += "\n" + traceback.format_exc()
    finally:
        if device.type == "cuda":
            row["peak_memory_mb"] = mb(torch.cuda.max_memory_allocated(device))
            row["allocated_memory_mb"] = mb(torch.cuda.memory_allocated(device))
            row["reserved_memory_mb"] = mb(torch.cuda.max_memory_reserved(device))
            row["peak_delta_memory_mb"] = row["peak_memory_mb"] - row["base_allocated_mb"]
            free_after, total_vram = cuda_mem_info()
            row["free_after_mb"] = mb(free_after)
            row["total_vram_mb"] = mb(total_vram)
        if not math.isnan(float(row["peak_memory_mb"])) and not math.isnan(float(row["kv_actual_mb"])):
            row["non_kv_overhead_mb"] = row["peak_memory_mb"] - row["kv_actual_mb"]
        del inputs
        empty_cache(device)
    return row


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def resolve_model_path(model: str, local_model_dir: str | None) -> str:
    if local_model_dir:
        return local_model_dir
    if "/" not in model:
        return model
    local_candidate = Path("/home/ssu/models") / model.rsplit("/", 1)[-1]
    if local_candidate.exists():
        return str(local_candidate)
    opt_candidate = Path("/home/ssu/models") / model.rsplit("/", 1)[-1].replace("opt-", "opt-")
    if opt_candidate.exists():
        return str(opt_candidate)
    return model


def load_tokenizer(model_path: str, trust_remote_code: bool, local_files_only: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-id", dest="model", required=True)
    parser.add_argument("--gpu-name", default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="fp16")
    parser.add_argument("--batch-sizes", nargs="+", required=True, help="Space or comma separated batch sizes")
    parser.add_argument("--seq-lens", nargs="+", required=True, help="Space or comma separated sequence lengths")
    parser.add_argument("--cache-modes", nargs="+", default=["dynamic", "quantized", "offloaded", "no_cache"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", "--out", dest="output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-model-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--quant-backend", default="quanto", choices=["quanto", "HQQ"])
    parser.add_argument("--quant-nbits", type=int, default=4)
    parser.add_argument("--debug-traceback", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    metric_dtype = args.dtype if device.type == "cuda" else "fp32"
    output = Path(args.output)
    model_path = resolve_model_path(args.model, args.local_model_dir)

    tokenizer = load_tokenizer(model_path, args.trust_remote_code, args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(device)
    if len(tokenizer) > getattr(model.get_input_embeddings(), "num_embeddings", len(tokenizer)):
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    shape = infer_model_shape(model.config)
    gpu_name = args.gpu_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    batch_sizes = parse_values(args.batch_sizes, int)
    seq_lens = parse_values(args.seq_lens, int)
    cache_modes = parse_values(args.cache_modes, str)

    print(f"Loaded {args.model} from {model_path} on {device}; shape={shape}", flush=True)
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for cache_mode in cache_modes:
                print(f"case batch={batch_size} seq={seq_len} cache={cache_mode}", flush=True)
                row = bench_one(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model,
                    gpu_name=gpu_name,
                    dtype_name=metric_dtype,
                    cache_mode=cache_mode,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    max_new_tokens=args.max_new_tokens,
                    shape=shape,
                    device=device,
                    warmup=args.warmup,
                    quant_backend=args.quant_backend,
                    quant_nbits=args.quant_nbits,
                    include_traceback=args.debug_traceback,
                )
                print(
                    "  -> "
                    f"{row['status']} "
                    f"peak_mb={row['peak_memory_mb']:.1f} "
                    f"tps={row['tokens_per_sec']} "
                    f"err={str(row['error_message'])[:120]}",
                    flush=True,
                )
                write_row(output, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
