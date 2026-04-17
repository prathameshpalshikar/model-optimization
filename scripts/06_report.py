"""
PHASE 6: FINAL REPORT
Aggregate HF optimization phases and GGUF/llama.cpp deployment results.
"""

import json
import re
import time
from pathlib import Path

from utils import resolve_artifact_pointer

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
FINAL_MD = REPORTS_DIR / "final_report.md"
FINAL_JSON = REPORTS_DIR / "final_metrics.json"
SIZE_TARGET_MIB = 1230
MEMORY_TARGET_MB = 2048
THROUGHPUT_TARGET_TPS = 5.0


def latest_file(path: Path, pattern: str):
    def sort_key(item: Path):
        match = re.search(r"(\d{8}_\d{6})", item.stem)
        return match.group(1) if match else time.strftime("%Y%m%d_%H%M%S", time.localtime(item.stat().st_mtime))

    files = sorted(path.glob(pattern), key=sort_key, reverse=True)
    return files[0] if files else None


def load_json(path: Path):
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_exists_from_report(path_text: str | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if path.exists():
        return True
    if not path.is_absolute() and (ROOT / path).exists():
        return True
    return False


def summarize_pruning(selection: dict | None) -> str:
    if not selection:
        return "None"
    method = selection.get("pruning_type", "wanda_activation_aware_weight_pruning")
    target = selection.get("ratio_target")
    effective = selection.get("ratio_effective")
    passed = selection.get("quality_gate_passed")
    return f"{method}, target={target}, effective={effective}, quality_gate_passed={passed}"


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
    quantized_path = quant.get("quantized_gguf_path") if quant else None
    artifact_exists = artifact_exists_from_report(quantized_path)
    latest_gguf = resolve_artifact_pointer(ROOT / "artifacts" / "gguf" / "latest_gguf_path.txt")

    size_ok = bool(gguf_size_mb is not None and gguf_size_mb <= SIZE_TARGET_MIB)
    mem_ok = bool(runtime_mem_mb is not None and runtime_mem_mb <= MEMORY_TARGET_MB)
    stable_ok = bool(runtime and runtime.get("stable_inference", False))
    throughput_ok = bool(runtime_tps is not None and runtime_tps >= THROUGHPUT_TARGET_TPS)

    failure_analysis = []
    if not size_ok:
        failure_analysis.append(f"Final GGUF artifact exceeds the {SIZE_TARGET_MIB} MiB size target or size is missing.")
    if not artifact_exists:
        failure_analysis.append("Final GGUF path from the quantization report is not present on this machine.")
    if not mem_ok:
        failure_analysis.append(f"llama.cpp runtime memory exceeds the {MEMORY_TARGET_MB} MB target or memory is missing.")
    if not stable_ok:
        failure_analysis.append("llama.cpp multi-turn stability validation failed.")
    if not throughput_ok:
        failure_analysis.append(f"llama.cpp throughput is below {THROUGHPUT_TARGET_TPS} tokens/sec or was not measured.")

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
            "path": quantized_path,
            "path_exists": artifact_exists,
            "latest_pointer_resolves_to": str(latest_gguf) if latest_gguf else None,
            "size_mb": gguf_size_mb,
        },
        "runtime_validation": {
            "runtime": runtime.get("runtime") if runtime else None,
            "average_tokens_per_sec": runtime_tps,
            "peak_runtime_memory_mb": runtime_mem_mb,
            "stable_inference": runtime.get("stable_inference") if runtime else None,
        },
        "failure_analysis": failure_analysis,
        "deployable": all([size_ok, artifact_exists, mem_ok, stable_ok, throughput_ok]),
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
        f"- Pruning selection: {summarize_pruning(final['hf_pipeline']['pruning_selected'])}",
        f"- QAT-lite method: {qat.get('method', 'fake_quant_lora') if qat else None}",
        f"- QAT uses quantization simulation: {qat.get('uses_quantization_simulation', False) if qat else False}",
        "",
        "## Final deployable artifact",
        f"- Format: {final['deployable_artifact']['format']}",
        f"- Quantization: {final['deployable_artifact']['quantization']}",
        f"- Path: {final['deployable_artifact']['path']}",
        f"- Path exists: {final['deployable_artifact']['path_exists']}",
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
