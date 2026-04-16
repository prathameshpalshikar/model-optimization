"""
PHASE 1: BASELINE MEASUREMENT (HUGGING FACE)
Download/load base model, measure throughput/memory, and save multilingual samples.
"""

import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_current_memory_mb, get_multilingual_eval_set, get_peak_memory_mb, start_memory_tracking

MODEL_ID = "Qwen/Qwen3.5-2B"
BASELINE_DIR = Path(__file__).parent.parent / "baselines"


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> tuple[str, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded, elapsed, int(new_tokens)


def safe_preview(text: str, width: int = 35) -> str:
    return text[:width].encode("ascii", errors="replace").decode("ascii")


def main():
    print("=" * 80)
    print("PHASE 1: BASELINE MEASUREMENT (HF)")
    print("=" * 80)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    stamp = timestamp_slug()
    output_metrics = BASELINE_DIR / f"baseline_measurements_{stamp}.json"
    output_samples = BASELINE_DIR / f"baseline_outputs_samples_{stamp}.txt"

    print("\n[1/4] Loading model from Hugging Face...")
    print(f"Model: {MODEL_ID}")
    start_memory_tracking()
    rss_before = get_current_memory_mb()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    rss_after_load = get_current_memory_mb()
    peak_mem_mb = get_peak_memory_mb()
    print(f"  RSS before load: {rss_before:.1f} MB")
    print(f"  RSS after load:  {rss_after_load:.1f} MB")

    print("\n[2/4] Measuring throughput...")
    throughput_prompts = [
        "What is the capital of France?",
        "Express machine learning in one sentence.",
    ]
    times = []
    tokens = []
    for i, prompt in enumerate(throughput_prompts, 1):
        print(f"  [{i}/{len(throughput_prompts)}] {prompt[:45]}...", end=" ")
        try:
            _, elapsed, new_tokens = generate_text(model, tokenizer, prompt, max_new_tokens=50)
            times.append(elapsed)
            tokens.append(new_tokens)
            print(f"OK ({elapsed:.2f}s, {new_tokens} tokens)")
        except Exception as exc:
            print(f"FAILED ({exc})")

    total_time = sum(times)
    total_tokens = sum(tokens)
    throughput = (total_tokens / total_time) if total_time > 0 else 0.0
    avg_latency = (total_time / len(times)) if times else 0.0

    print("\n[3/4] Generating multilingual output samples...")
    prompts_by_lang = get_multilingual_eval_set()
    with output_samples.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("BASELINE MULTILINGUAL OUTPUT SAMPLES\n")
        handle.write(f"Model: {MODEL_ID}\n")
        handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write("=" * 80 + "\n\n")
        for lang, prompts in prompts_by_lang.items():
            handle.write(f"{'=' * 80}\nLANGUAGE: {lang.upper()}\nSamples: {len(prompts)}\n{'=' * 80}\n\n")
            for idx, prompt in enumerate(prompts, 1):
                print(f"  {lang.upper():10s} [{idx}/{len(prompts)}] {safe_preview(prompt)}...", end=" ")
                try:
                    output_text, _, _ = generate_text(model, tokenizer, prompt, max_new_tokens=60)
                    handle.write(f"[Prompt {idx}]\n{prompt}\n\n[Output]\n{output_text}\n")
                    print("OK")
                except Exception as exc:
                    handle.write(f"[Prompt {idx}]\n{prompt}\n\n[Output]\n[ERROR: {exc}]\n")
                    print("FAILED")
                handle.write("-" * 80 + "\n\n")

    print("\n[4/4] Saving metrics...")
    mem = psutil.virtual_memory()
    metrics = {
        "phase": "baseline",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "throughput_metrics": {
            "tokens_per_sec": throughput,
            "avg_latency_sec": avg_latency,
            "num_test_calls": len(throughput_prompts),
            "generated_tokens_total": total_tokens,
        },
        "memory_metrics": {
            "rss_before_mb": rss_before,
            "rss_after_model_load_mb": rss_after_load,
            "peak_tracemalloc_mb": peak_mem_mb,
        },
        "system_metrics": {
            "cpu_cores": psutil.cpu_count(),
            "total_ram_gb": mem.total / (1024 ** 3),
            "available_ram_gb": mem.available / (1024 ** 3),
        },
        "multilingual_samples": {lang: len(prompts) for lang, prompts in prompts_by_lang.items()},
        "artifacts": {
            "samples_file": str(output_samples),
        },
    }
    with output_metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nPHASE 1 COMPLETE")
    print(f"Throughput: {throughput:.2f} tokens/sec")
    print(f"Metrics: {output_metrics}")
    print(f"Samples: {output_samples}")


if __name__ == "__main__":
    main()
