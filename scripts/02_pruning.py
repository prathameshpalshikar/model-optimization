"""
PHASE 2: ACTIVATION-AWARE WANDA PRUNING
Prune attention/MLP weights using calibration activations and WANDA scores.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import assess_generation, get_multilingual_eval_set, write_artifact_pointer

MODEL_ID = "Qwen/Qwen3.5-2B"
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "pruned"
REPORT_DIR = Path(__file__).parent.parent / "reports" / "pruning"


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_ratios(raw: str) -> list[float]:
    ratios = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio < 0 or ratio >= 1:
            raise ValueError(f"Pruning ratio must be in [0, 1): {ratio}")
        ratios.append(ratio)
    return ratios or [0.10]


def is_prunable_linear(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lowered = name.lower()
    target_tokens = (
        "attn",
        "attention",
        "mlp",
        "ffn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    return any(token in lowered for token in target_tokens)


def normalize_calibration_prompt(row: dict) -> str | None:
    prompt = row.get("prompt") or row.get("instruction") or row.get("question") or row.get("input") or row.get("text")
    return str(prompt) if prompt else None


def load_calibration_prompts(path_text: str | None, limit: int) -> tuple[list[str], str]:
    if not path_text:
        return collect_builtin_calibration_prompts(limit), "builtin_multilingual_prompts"

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")

    prompts = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                prompt = normalize_calibration_prompt(json.loads(line))
                if prompt:
                    prompts.append(prompt)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("data", [])
        for item in items:
            prompt = normalize_calibration_prompt(item)
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No calibration prompts found in {path}")
    return prompts[:limit], str(path)


def collect_builtin_calibration_prompts(limit: int) -> list[str]:
    prompts = []
    for items in get_multilingual_eval_set().values():
        prompts.extend(items)
    return prompts[:limit]


def collect_activation_stats(model, tokenizer, prompts: list[str], max_length: int = 256) -> dict[str, dict[str, torch.Tensor | int]]:
    stats: dict[str, dict[str, torch.Tensor | int]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs):
            if not inputs:
                return
            activation = inputs[0].detach().float()
            if activation.ndim == 1:
                activation = activation.unsqueeze(0)
            activation = activation.reshape(-1, activation.shape[-1])
            current = stats.setdefault(
                name,
                {
                    "sum_sq": torch.zeros(activation.shape[-1], dtype=torch.float32),
                    "count": 0,
                },
            )
            current["sum_sq"] += activation.pow(2).sum(dim=0).cpu()
            current["count"] += activation.shape[0]

        return hook

    for name, module in model.named_modules():
        if is_prunable_linear(name, module):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            model(**inputs)

    for hook in hooks:
        hook.remove()
    return stats


def apply_wanda_pruning(model, activation_stats: dict[str, dict[str, torch.Tensor | int]], ratio: float) -> dict:
    total_pruned = 0
    total_seen = 0
    module_reports = {}

    for name, module in model.named_modules():
        if not is_prunable_linear(name, module):
            continue
        if name not in activation_stats:
            continue
        stat = activation_stats[name]
        count = int(stat["count"])
        if count <= 0:
            continue

        with torch.no_grad():
            weight = module.weight.data
            in_features = weight.shape[1]
            prune_per_row = int(in_features * ratio)
            if prune_per_row <= 0:
                continue

            activation_scale = torch.sqrt(stat["sum_sq"].to(weight.device) / max(1, count))
            metric = weight.detach().abs().float() * activation_scale.unsqueeze(0)
            threshold = torch.kthvalue(metric, prune_per_row, dim=1).values.unsqueeze(1)
            keep_mask = metric > threshold
            before_nonzero = int(torch.count_nonzero(weight).item())
            weight.mul_(keep_mask.to(dtype=weight.dtype))
            after_nonzero = int(torch.count_nonzero(weight).item())
            pruned = before_nonzero - after_nonzero
            seen = weight.numel()
            total_pruned += pruned
            total_seen += seen
            module_reports[name] = {
                "shape": list(weight.shape),
                "pruned_weights": pruned,
                "total_weights": seen,
                "effective_ratio": pruned / seen if seen else 0.0,
                "activation_samples": count,
            }

    return {
        "pruned_params": total_pruned,
        "eligible_params": total_seen,
        "ratio_effective": total_pruned / total_seen if total_seen else 0.0,
        "module_reports": module_reports,
    }


def run_generation_check(model, tokenizer) -> dict:
    checks = {}
    model.eval()
    for lang, prompts in get_multilingual_eval_set().items():
        prompt_index = 0
        prompt = prompts[prompt_index]
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assessment = assess_generation(text, lang, prompt_index=prompt_index)
            checks[lang] = {**assessment, "preview": text[:200]}
        except Exception as exc:
            checks[lang] = {"ok": False, "error": str(exc)}
    return checks


def main():
    parser = argparse.ArgumentParser(description="Activation-aware WANDA pruning for the pipeline.")
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", MODEL_ID))
    parser.add_argument("--ratios", default=os.environ.get("PRUNE_RATIOS", "0.10,0.15,0.20"))
    parser.add_argument("--calibration-samples", type=int, default=int(os.environ.get("PRUNE_CALIBRATION_SAMPLES", "8")))
    parser.add_argument("--calibration-file", default=os.environ.get("PRUNE_CALIBRATION_FILE"))
    parser.add_argument("--calibration-max-length", type=int, default=int(os.environ.get("PRUNE_CALIBRATION_MAX_LENGTH", "256")))
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 2: ACTIVATION-AWARE WANDA PRUNING")
    print("=" * 80)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_slug()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ratios = parse_ratios(args.ratios)
    calibration_prompts, calibration_source = load_calibration_prompts(args.calibration_file, args.calibration_samples)
    best = None
    reports = []

    for ratio in ratios:
        candidate = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float32)
        activation_stats = collect_activation_stats(
            candidate,
            tokenizer,
            calibration_prompts,
            max_length=args.calibration_max_length,
        )
        prune_result = apply_wanda_pruning(candidate, activation_stats, ratio)

        quality = run_generation_check(candidate, tokenizer)
        quality_ok = all(v.get("ok", False) for v in quality.values())
        report = {
            "pruning_type": "wanda_activation_aware_weight_pruning",
            "target_scope": "attention_and_mlp_weight_tensors",
            "shape_change": False,
            "ratio_target": ratio,
            "ratio_effective": prune_result["ratio_effective"],
            "pruned_params": prune_result["pruned_params"],
            "eligible_params": prune_result["eligible_params"],
            "calibration_prompts": len(calibration_prompts),
            "calibration_source": calibration_source,
            "module_reports": prune_result["module_reports"],
            "quality_gate_passed": quality_ok,
            "quality_check": quality,
        }
        reports.append(report)
        print(f"ratio={ratio:.2f} effective={prune_result['ratio_effective']:.4f} gate={quality_ok}")

        if quality_ok and best is None:
            best = (ratio, candidate, report)
        else:
            del candidate

    if best is None:
        print("No candidate passed the quality gate. Keeping original model at 10% report only.")
        best_ratio = 0.10
        best_model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float32)
        best_report = {
            "pruning_type": "none_fallback",
            "ratio_target": 0.10,
            "quality_gate_passed": False,
        }
    else:
        best_ratio, best_model, best_report = best

    out_model_dir = ARTIFACT_DIR / f"pruned_{int(best_ratio*100)}pct_{stamp}"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)

    report_payload = {
        "phase": "pruning",
        "method": "wanda_activation_aware_weight_pruning",
        "notes": "WANDA scores abs(weight) by calibration activation scale before pruning. Tensor shapes are preserved for compatibility with the Qwen3.5 architecture.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": args.model_id,
        "calibration_samples": len(calibration_prompts),
        "calibration_source": calibration_source,
        "calibration_max_length": args.calibration_max_length,
        "selected": best_report,
        "all_candidates": reports,
        "artifact_dir": str(out_model_dir),
    }
    report_file = REPORT_DIR / f"pruning_report_{stamp}.json"
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    latest = ARTIFACT_DIR / "latest_pruned_path.txt"
    write_artifact_pointer(latest, out_model_dir)

    print(f"Pruned model saved to: {out_model_dir}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
