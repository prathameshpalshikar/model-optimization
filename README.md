# Edge LLM Optimization Pipeline

## 1. Objective

This project reduces model size and runtime memory for a multilingual language model intended for memory-constrained CPU edge devices.

Starting point:
- Approximately 2B parameter model

Target constraints:
- Model size around 1.2 GiB or lower for the current Q4_K_M target
- Runtime memory around 2 GiB or lower
- CPU-only inference

## 2. Approach

The optimization pipeline follows a strict sequence:

1. Baseline measurement
2. Activation-aware WANDA pruning over attention/MLP Linear weights
3. QAT-ready LoRA recovery with optional fake 4-bit quantization simulation
4. GGUF export and configurable llama.cpp post-training quantization
5. Runtime validation with llama.cpp
6. Final reporting
7. Optional GGUF comparison against a reference model such as Unsloth

Each step is validated before proceeding.

## 3. Repository Structure

scripts/
- 01_baseline.py
- 02_pruning.py
- 03_quantize.py
- 04_qat_lite.py
- 05_runtime_opt.py
- 06_report.py
- 07_compare_gguf.py
- utils.py

reports/
- performance comparisons and pipeline reports

artifacts/
- lightweight latest-artifact pointers and small metadata files

## 4. Key Design Decisions

- No distillation dataset is currently used.
- Pruning uses WANDA scores: `abs(weight) * activation_scale`, where activation scale is collected from calibration prompts.
- Pruning currently preserves tensor shapes for Qwen3.5 compatibility; it prunes scalar weights inside attention/MLP Linear tensors.
- QAT-lite can run in `fake_quant_lora` mode, where target Linear layers simulate group-wise 4-bit quantized weights during LoRA recovery training.
- If `--dataset` is not provided, QAT uses answer-bearing synthetic fallback data. This verifies mechanics but is not enough for high-quality QAT.
- llama.cpp `llama-quantize` performs the deployable GGUF post-training quantization.
- CPU-first deployment target.
- llama.cpp is used for final inference benchmarking.

## 5. Important Note on Model Files

Large model files are not included in this repository.

Excluded:
- *.gguf
- *.safetensors
- checkpoints and binaries

Reason:
- GitHub size limits
- Models are treated as artifacts, not source code

To reproduce:
- Download or generate models locally.
- Place them in ignored local artifact/model folders.
- Keep latest-artifact pointer files portable by using repo-relative paths when possible.

## 6. How to Run

From the repository root:

```powershell
python scripts/01_baseline.py
python scripts/02_pruning.py
python scripts/04_qat_lite.py
python scripts/03_quantize.py
python scripts/05_runtime_opt.py
python scripts/06_report.py
```

Useful knobs:

```powershell
python scripts/02_pruning.py --ratios 0.10,0.15 --calibration-samples 8
python scripts/04_qat_lite.py --qat-mode fake_quant_lora --dataset path\to\train.jsonl --max-steps 200
python scripts/03_quantize.py --quant-type Q4_K_M
```

After restoring or generating GGUF files, compare against a reference GGUF:

```powershell
python scripts/07_compare_gguf.py --candidate path\to\your.gguf --reference path\to\unsloth.gguf
```

## 7. Benchmarking

Final models are tested using llama.cpp CPU inference.

Metrics:
- tokens/sec
- RAM usage
- multilingual output quality
- GGUF size, metadata, tensor count, and tensor quantization-type distribution against a reference model

## 8. Limitations

- WANDA pruning is real activation-aware pruning, but because tensor shapes are preserved, it does not guarantee GGUF size or CPU speed improvements.
- Quantization introduces quality degradation.
- Fake-quant LoRA recovery is QAT-ready but still needs representative data for high-quality output recovery.
- No large-scale fine-tuning dataset is included.
- Performance depends on CPU capability and llama.cpp version.

## 9. Future Work

- Add a representative multilingual supervised dataset for true QAT or better recovery fine-tuning.
- Add perplexity and task-level quality gates.
- Explore structured pruning only if speed/size reductions are required before quantization.
- Evaluate mixed precision and alternative GGUF quantization strategies.
