"""
PHASE 4: QAT-LITE
Generate small multilingual recovery data and run short parameter-efficient fine-tuning.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from utils import get_multilingual_eval_set, resolve_artifact_pointer, write_artifact_pointer

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "qat_lite"
REPORT_DIR = Path(__file__).parent.parent / "reports" / "qat_lite"
PRUNED_LATEST = Path(__file__).parent.parent / "artifacts" / "pruned" / "latest_pruned_path.txt"


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def fake_quantize_groupwise_symmetric(weight: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Group-wise symmetric fake quantization with STE for QAT-style training."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    if weight.ndim != 2:
        return weight

    qmax = (2 ** (bits - 1)) - 1
    original_shape = weight.shape
    in_features = original_shape[1]
    pad = (group_size - (in_features % group_size)) % group_size
    padded = F.pad(weight.float(), (0, pad)) if pad else weight.float()
    grouped = padded.reshape(original_shape[0], -1, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    quant = torch.clamp(torch.round(grouped / scale), -qmax - 1, qmax)
    dequant = (quant * scale).reshape(original_shape[0], -1)
    if pad:
        dequant = dequant[:, :in_features]
    dequant = dequant.to(dtype=weight.dtype)
    return weight + (dequant - weight).detach()


class FakeQuantLinear(nn.Linear):
    """Linear layer that simulates low-bit weight quantization during forward passes."""

    def __init__(self, source: nn.Linear, bits: int = 4, group_size: int = 128):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.weight.data.copy_(source.weight.data)
        if source.bias is not None:
            self.bias.data.copy_(source.bias.data)
        self.bits = bits
        self.group_size = group_size

    def forward(self, input):
        quantized_weight = fake_quantize_groupwise_symmetric(self.weight, bits=self.bits, group_size=self.group_size)
        return F.linear(input, quantized_weight, self.bias)


def is_qat_target_linear(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lowered = name.lower()
    target_tokens = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "attn", "ffn", "mlp")
    return any(token in lowered for token in target_tokens)


def replace_module(root: nn.Module, dotted_name: str, replacement: nn.Module):
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def inject_fake_quant_linear(model: nn.Module, bits: int = 4, group_size: int = 128) -> int:
    replacements = []
    for name, module in model.named_modules():
        if is_qat_target_linear(name, module):
            replacements.append((name, module))
    for name, module in replacements:
        replace_module(model, name, FakeQuantLinear(module, bits=bits, group_size=group_size))
    return len(replacements)

SUPERVISED_RECOVERY_ANSWERS = {
    "english": [
        "The capital of France is Paris.",
        "Relativity says space, time, and gravity depend on motion and mass rather than being fixed for everyone.",
        "A small robot learned to care for a garden and discovered that helping living things gave its work meaning.",
        "Plants use chlorophyll to capture sunlight and convert carbon dioxide and water into glucose through photosynthesis.",
        "Regular exercise supports heart health, strength, mood, sleep, and weight management.",
    ],
    "hindi": [
        "भारत की राजधानी नई दिल्ली है।",
        "2 + 2 का उत्तर 4 है।",
    ],
    "marathi": [
        "महाराष्ट्राची राजधानी मुंबई आहे.",
        "शिक्षणामुळे ज्ञान, आत्मविश्वास आणि चांगल्या संधी मिळतात.",
    ],
    "telugu": [
        "తెలుగు భారతదేశంలో ప్రధానంగా మాట్లాడే భాష.",
        "నీరు జీవానికి అవసరం; అది ఆరోగ్యం, వ్యవసాయం మరియు పర్యావరణానికి ముఖ్యమైనది.",
    ],
}

def build_synthetic_samples(target_count: int = 1200) -> list[dict]:
    prompts = get_multilingual_eval_set()
    templates = [
        "Answer clearly in the same language: {prompt}",
        "Give a concise factual response: {prompt}",
        "Instruction: {prompt}",
    ]
    rows = []
    for lang, items in prompts.items():
        answers = SUPERVISED_RECOVERY_ANSWERS.get(lang, [])
        for idx, prompt in enumerate(items):
            answer = answers[idx] if idx < len(answers) else ""
            if not answer:
                continue
            for _ in range(max(1, target_count // 60)):
                prompt_text = random.choice(templates).format(prompt=prompt)
                rows.append({"prompt": prompt_text, "answer": answer})
    while len(rows) < target_count:
        lang = random.choice(list(prompts.keys()))
        idx = random.randrange(len(prompts[lang]))
        answer = SUPERVISED_RECOVERY_ANSWERS[lang][idx]
        prompt_text = random.choice(templates).format(prompt=prompts[lang][idx])
        rows.append({"prompt": prompt_text, "answer": answer})
    random.shuffle(rows)
    return rows[:target_count]


def normalize_dataset_row(row: dict) -> dict | None:
    prompt = row.get("prompt") or row.get("instruction") or row.get("question") or row.get("input")
    answer = row.get("answer") or row.get("response") or row.get("output") or row.get("completion")
    text = row.get("text")
    if prompt and answer:
        return {"prompt": str(prompt), "answer": str(answer)}
    if text:
        return {"prompt": "Continue:", "answer": str(text)}
    return None


def load_supervised_rows(dataset_path: str | None, fallback_count: int) -> tuple[list[dict], str]:
    if not dataset_path:
        return build_synthetic_samples(target_count=fallback_count), "synthetic_fallback"

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    rows = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                normalized = normalize_dataset_row(json.loads(line))
                if normalized:
                    rows.append(normalized)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("data", [])
        for item in items:
            normalized = normalize_dataset_row(item)
            if normalized:
                rows.append(normalized)

    if not rows:
        raise ValueError(f"No prompt/answer rows found in dataset: {path}")
    random.shuffle(rows)
    return rows, str(path)


def main():
    parser = argparse.ArgumentParser(description="QAT-ready LoRA recovery fine-tuning.")
    parser.add_argument("--dataset", default=os.environ.get("QAT_DATASET"))
    parser.add_argument("--target-count", type=int, default=int(os.environ.get("QAT_TARGET_COUNT", "600")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("QAT_MAX_STEPS", "10")))
    parser.add_argument("--qat-mode", choices=["fake_quant_lora", "lora"], default=os.environ.get("QAT_MODE", "fake_quant_lora"))
    parser.add_argument("--quant-bits", type=int, default=int(os.environ.get("QAT_QUANT_BITS", "4")))
    parser.add_argument("--group-size", type=int, default=int(os.environ.get("QAT_GROUP_SIZE", "128")))
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("QAT_LORA_R", "8")))
    args_cli = parser.parse_args()

    print("=" * 80)
    print("PHASE 4: QAT-LITE")
    print("=" * 80)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_slug()
    random.seed(42)

    model_path_obj = resolve_artifact_pointer(PRUNED_LATEST)
    model_path = str(model_path_obj) if model_path_obj is not None else "Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")

    fake_quant_modules = 0
    if args_cli.qat_mode == "fake_quant_lora":
        fake_quant_modules = inject_fake_quant_linear(model, bits=args_cli.quant_bits, group_size=args_cli.group_size)

    lora_cfg = LoraConfig(
        r=args_cli.lora_r,
        lora_alpha=args_cli.lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    rows, dataset_source = load_supervised_rows(args_cli.dataset, fallback_count=args_cli.target_count)
    dataset = Dataset.from_list(rows)

    def tokenize_fn(batch):
        prefixes = [f"{prompt}\nResponse:" for prompt in batch["prompt"]]
        suffix = tokenizer.eos_token or ""
        full_texts = [f"{prefix} {answer}{suffix}" for prefix, answer in zip(prefixes, batch["answer"])]
        out = tokenizer(full_texts, truncation=True, max_length=256, padding="max_length")
        labels = []
        for input_ids, attention_mask, prefix in zip(out["input_ids"], out["attention_mask"], prefixes):
            label = list(input_ids)
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            for idx in range(min(prefix_len, len(label))):
                label[idx] = -100
            for idx, mask in enumerate(attention_mask):
                if mask == 0:
                    label[idx] = -100
            labels.append(label)
        out["labels"] = labels
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "answer"])
    out_dir = ARTIFACT_DIR / f"qat_lite_{stamp}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="no",
        max_steps=args_cli.max_steps,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    train_result = trainer.train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    merged_dir = ARTIFACT_DIR / f"qat_lite_merged_{stamp}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    report = {
        "phase": "qat_lite",
        "method": args_cli.qat_mode,
        "uses_quantization_simulation": args_cli.qat_mode == "fake_quant_lora",
        "true_qat_ready": args_cli.qat_mode == "fake_quant_lora",
        "notes": "With --dataset, this runs LoRA recovery while target Linear layers see fake-quantized weights. Without --dataset, it uses a small synthetic fallback and should be treated as a mechanics test, not quality QAT.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_model": model_path,
        "synthetic_samples": len(rows),
        "dataset_source": dataset_source,
        "sample_type": "prompt_response_pairs",
        "fake_quant_modules": fake_quant_modules,
        "quant_bits": args_cli.quant_bits if args_cli.qat_mode == "fake_quant_lora" else None,
        "group_size": args_cli.group_size if args_cli.qat_mode == "fake_quant_lora" else None,
        "epochs": 1,
        "max_steps": args_cli.max_steps,
        "adapter_artifact_dir": str(out_dir),
        "merged_artifact_dir": str(merged_dir),
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
    }
    report_file = REPORT_DIR / f"qat_lite_report_{stamp}.json"
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    write_artifact_pointer(ARTIFACT_DIR / "latest_qat_path.txt", out_dir)
    write_artifact_pointer(ARTIFACT_DIR / "latest_qat_merged_path.txt", merged_dir)
    print(f"QAT-lite adapter saved to: {out_dir}")
    print(f"Merged HF model saved to: {merged_dir}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
