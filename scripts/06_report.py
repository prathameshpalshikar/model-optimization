"""
PHASE 6: FINAL REPORT
Aggregate HF optimization phases and GGUF/llama.cpp deployment results.
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
FINAL_MD = REPORTS_DIR / "final_report.md"
FINAL_JSON = REPORTS_DIR / "final_metrics.json"


def latest_file(path: Path, pattern: str):
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json(path: Path):
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_json(latest_file(ROOT / "baselines", "baseline_measurements_*.json"))
    prune = load_json(latest_file(ROOT / "reports" / "pruning", "pruning_report_*.json"))
    quant = load_json(latest_file(ROOT / "reports" / "quantization", "quantization_report_*.json"))
    qat = load_json(latest_file(ROOT / "reports" / "qat_lite", "qat_lite_report_*.json"))
    runtime = load_json(latest_file(ROOT / "reports" / "runtime", "runtime_report_*.json"))

    gguf_size_mb = quant.get("quantized_gguf_size_mb") if quant else None
    runtime_mem_mb = runtime.get("peak_runtime_memory_mb") if runtime else None
    runtime_tps = runtime.get("average_tokens_per_sec") if runtime else None

    # Relaxed deployment size target per user request.
    # Use MiB threshold corresponding to ~1.2 GiB.
    size_ok = bool(gguf_size_mb is not None and gguf_size_mb <= 1230)
    mem_ok = bool(runtime_mem_mb is not None and runtime_mem_mb <= 2048)
    stable_ok = bool(runtime and runtime.get("stable_inference", False))
    throughput_ok = bool(runtime_tps is not None and runtime_tps >= 5.0)

    failure_analysis = []
    if not size_ok:
        failure_analysis.append("Final GGUF artifact exceeds the 1.2 GB size target.")
    if not mem_ok:
        failure_analysis.append("llama.cpp runtime memory exceeds the 2 GB target.")
    if not stable_ok:
        failure_analysis.append("llama.cpp multi-turn stability validation failed.")
    if not throughput_ok:
        failure_analysis.append("llama.cpp throughput is below 5 tokens/sec.")

    final = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hf_pipeline": {
            "base_model": baseline.get("model") if baseline else None,
            "pruning_selected": prune.get("selected") if prune else None,
            "qat_report": qat,
        },
        "deployable_artifact": {
            "format": "GGUF",
            "quantization": quant.get("quantization_method") if quant else None,
            "path": quant.get("quantized_gguf_path") if quant else None,
            "size_mb": gguf_size_mb,
        },
        "runtime_validation": {
            "runtime": runtime.get("runtime") if runtime else None,
            "average_tokens_per_sec": runtime_tps,
            "peak_runtime_memory_mb": runtime_mem_mb,
            "stable_inference": runtime.get("stable_inference") if runtime else None,
        },
        "failure_analysis": failure_analysis,
        "deployable": all([size_ok, mem_ok, stable_ok, throughput_ok]),
    }
    with FINAL_JSON.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)

    lines = [
        "# Final Edge Deployability Report",
        "",
        f"Generated: {final['timestamp']}",
        "",
        "## HF optimization pipeline",
        f"- Base model: {final['hf_pipeline']['base_model']}",
        f"- Pruning selection: {final['hf_pipeline']['pruning_selected']}",
        "",
        "## Final deployable artifact",
        f"- Format: {final['deployable_artifact']['format']}",
        f"- Quantization: {final['deployable_artifact']['quantization']}",
        f"- Path: {final['deployable_artifact']['path']}",
        f"- Size (MB): {final['deployable_artifact']['size_mb']}",
        "",
        "## llama.cpp runtime validation",
        f"- Average tokens/sec: {final['runtime_validation']['average_tokens_per_sec']}",
        f"- Peak runtime memory (MB): {final['runtime_validation']['peak_runtime_memory_mb']}",
        f"- Stable inference: {final['runtime_validation']['stable_inference']}",
        "",
        "## Failure analysis",
    ]
    if failure_analysis:
        lines.extend([f"- {item}" for item in failure_analysis])
    else:
        lines.append("- No critical failures detected in the GGUF deployment path.")
    lines.extend(["", "## Final judgment", f"- Deployable under constraints: {final['deployable']}"])

    FINAL_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final report written: {FINAL_MD}")
    print(f"Final metrics written: {FINAL_JSON}")


if __name__ == "__main__":
    main()
