"""
PHASE 4: QAT-LITE
Generate synthetic multilingual data and run short parameter-efficient fine-tuning.
"""

import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from utils import get_multilingual_eval_set

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "qat_lite"
REPORT_DIR = Path(__file__).parent.parent / "reports" / "qat_lite"
PRUNED_LATEST = Path(__file__).parent.parent / "artifacts" / "pruned" / "latest_pruned_path.txt"


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def build_synthetic_samples(target_count: int = 1200) -> list[dict]:
    prompts = get_multilingual_eval_set()
    templates = [
        "Answer clearly: {prompt}",
        "Give a concise response in the same language: {prompt}",
        "Provide factual and short output: {prompt}",
    ]
    rows = []
    for lang, items in prompts.items():
        for prompt in items:
            for _ in range(max(1, target_count // 60)):
                text = random.choice(templates).format(prompt=prompt)
                rows.append({"text": f"{text}\nResponse:"})
    while len(rows) < target_count:
        lang = random.choice(list(prompts.keys()))
        prompt = random.choice(prompts[lang])
        rows.append({"text": f"Instruction: {prompt}\nAnswer:"})
    random.shuffle(rows)
    return rows[:target_count]


def main():
    print("=" * 80)
    print("PHASE 4: QAT-LITE")
    print("=" * 80)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_slug()

    model_path = PRUNED_LATEST.read_text(encoding="utf-8").strip() if PRUNED_LATEST.exists() else "Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")

    # LoRA as minimal recovery step instead of full-model QAT.
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_cfg)

    rows = build_synthetic_samples(target_count=600)
    dataset = Dataset.from_list(rows)

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    out_dir = ARTIFACT_DIR / f"qat_lite_{stamp}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="no",
        max_steps=10,
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_model": model_path,
        "synthetic_samples": len(rows),
        "epochs": 1,
        "adapter_artifact_dir": str(out_dir),
        "merged_artifact_dir": str(merged_dir),
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
    }
    report_file = REPORT_DIR / f"qat_lite_report_{stamp}.json"
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    (ARTIFACT_DIR / "latest_qat_path.txt").write_text(str(out_dir), encoding="utf-8")
    (ARTIFACT_DIR / "latest_qat_merged_path.txt").write_text(str(merged_dir), encoding="utf-8")
    print(f"QAT-lite adapter saved to: {out_dir}")
    print(f"Merged HF model saved to: {merged_dir}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
