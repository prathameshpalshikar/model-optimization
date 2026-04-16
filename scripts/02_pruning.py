"""
PHASE 2: LIGHT STRUCTURAL PRUNING
Apply conservative unstructured masking to attention/MLP weights (10-20%).
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_multilingual_eval_set

MODEL_ID = "Qwen/Qwen3.5-2B"
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "pruned"
REPORT_DIR = Path(__file__).parent.parent / "reports" / "pruning"


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def count_zeros(tensor: torch.Tensor) -> int:
    return int((tensor == 0).sum().item())


def prune_tensor_(tensor: torch.Tensor, ratio: float) -> tuple[int, int]:
    flat = tensor.detach().abs().view(-1)
    k = int(flat.numel() * ratio)
    if k <= 0:
        return 0, flat.numel()
    threshold = torch.kthvalue(flat, k).values
    mask = tensor.detach().abs() >= threshold
    tensor.mul_(mask)
    return flat.numel() - int(mask.sum().item()), flat.numel()


def run_generation_check(model, tokenizer) -> dict:
    checks = {}
    model.eval()
    for lang, prompts in get_multilingual_eval_set().items():
        prompt = prompts[0]
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            checks[lang] = {"ok": True, "preview": text[:200]}
        except Exception as exc:
            checks[lang] = {"ok": False, "error": str(exc)}
    return checks


def main():
    print("=" * 80)
    print("PHASE 2: STRUCTURAL PRUNING")
    print("=" * 80)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_slug()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)

    ratios = [0.10, 0.15, 0.20]
    best = None
    reports = []

    for ratio in ratios:
        candidate = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
        total_pruned = 0
        total_seen = 0
        for name, param in candidate.named_parameters():
            if not param.requires_grad or param.ndim < 2:
                continue
            if ("attn" in name) or ("mlp" in name):
                with torch.no_grad():
                    pruned, seen = prune_tensor_(param.data, ratio)
                    total_pruned += pruned
                    total_seen += seen

        quality = run_generation_check(candidate, tokenizer)
        quality_ok = all(v.get("ok", False) for v in quality.values())
        effective = (total_pruned / total_seen) if total_seen > 0 else 0.0
        report = {
            "ratio_target": ratio,
            "ratio_effective": effective,
            "pruned_params": total_pruned,
            "eligible_params": total_seen,
            "quality_gate_passed": quality_ok,
            "quality_check": quality,
        }
        reports.append(report)
        print(f"ratio={ratio:.2f} effective={effective:.4f} gate={quality_ok}")

        if quality_ok and best is None:
            best = (ratio, candidate, report)
        else:
            del candidate

    if best is None:
        print("No candidate passed the quality gate. Keeping original model at 10% report only.")
        best_ratio = 0.10
        best_model = model
        best_report = {"ratio_target": 0.10, "quality_gate_passed": False}
    else:
        best_ratio, best_model, best_report = best

    out_model_dir = ARTIFACT_DIR / f"pruned_{int(best_ratio*100)}pct_{stamp}"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)

    report_payload = {
        "phase": "pruning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": MODEL_ID,
        "selected": best_report,
        "all_candidates": reports,
        "artifact_dir": str(out_model_dir),
    }
    report_file = REPORT_DIR / f"pruning_report_{stamp}.json"
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    latest = ARTIFACT_DIR / "latest_pruned_path.txt"
    latest.write_text(str(out_model_dir), encoding="utf-8")

    print(f"Pruned model saved to: {out_model_dir}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
