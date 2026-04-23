"""Sparse V dequant skip A/B benchmark.

Measures decode throughput with and without TurboQuant at short and long
context. The sparse V skip is compiled into the Metal kernel — to test
without it, revert the kernel change and clear ~/.cache/vllm-metal/.

Usage:
  VLLM_METAL_TURBOQUANT=1 python3 benchmark_sparse_v.py
"""

import os
import time

import torch

os.environ.setdefault("VLLM_METAL_TURBOQUANT", "1")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
MAX_MODEL_LEN = 8192


def bench(llm, prompt, label, params):
    """Single benchmark run: 3 prompts, measure tok/s."""
    # Warmup
    llm.generate(["Hello"], params)

    prompts = [prompt] * 3
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / elapsed
    sample = outputs[0].outputs[0].text[:100].replace("\n", " ")
    print(f"  {label}: {total_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s")
    print(f"    Sample: {sample}")
    return tps


def main():
    tq_enabled = os.environ.get("VLLM_METAL_TURBOQUANT", "0") == "1"
    mode = "TurboQuant (sparse V)" if tq_enabled else "Baseline (no TQ)"
    print(f"\n{'='*60}")
    print(f"  {mode}")
    print(f"{'='*60}\n")

    llm = LLM(model=MODEL, dtype="float16", max_model_len=MAX_MODEL_LEN)
    params = SamplingParams(max_tokens=200, temperature=0)

    # Short context
    bench(llm, "Explain quantum computing:", "Short context", params)

    # Medium context (~2K tokens)
    med_prompt = ("The history of science spans many centuries. " * 100
                  + "To summarize the key developments:")
    bench(llm, med_prompt, "Medium context (~2K)", params)

    # Long context (~4K tokens)
    long_prompt = ("Throughout human history, technology has advanced. " * 200
                   + "The most important takeaway is:")
    bench(llm, long_prompt, "Long context (~4K)", params)

    del llm


if __name__ == "__main__":
    main()
