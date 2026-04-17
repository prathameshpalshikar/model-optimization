# Environment Setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Recommended cache location (optional):

```powershell
$env:HF_HOME = "$PWD\.hf_cache"
```

Optional reproducibility pin for llama.cpp downloads:

```powershell
$env:LLAMA_CPP_RELEASE_TAG = "bXXXX"
```

GGUF notes:

```powershell
# scripts/03_quantize.py will auto-download prebuilt llama.cpp CPU binaries
# and clone llama.cpp source for the GGUF converter if they are missing.
#
# Final deployable artifact:
#   artifacts/gguf/*Q4_K_M.gguf
#
# Runtime validation:
#   scripts/05_runtime_opt.py
```

Run the full pipeline:

```powershell
python scripts/01_baseline.py
python scripts/02_pruning.py
python scripts/04_qat_lite.py
python scripts/03_quantize.py
python scripts/05_runtime_opt.py
python scripts/06_report.py
```

Run the optimization phases with explicit real-method settings:

```powershell
# WANDA activation-aware pruning with calibration prompts
python scripts/02_pruning.py --ratios 0.10,0.15,0.20 --calibration-samples 8

# QAT-ready LoRA recovery. Provide JSONL rows with prompt/answer for real training data.
python scripts/04_qat_lite.py --qat-mode fake_quant_lora --dataset data\qat_train.jsonl --max-steps 200

# If no dataset is available, the same QAT mechanics run on synthetic fallback data.
python scripts/04_qat_lite.py --qat-mode fake_quant_lora --max-steps 10

# Deployable post-training GGUF quantization through llama.cpp.
python scripts/03_quantize.py --quant-type Q4_K_M
```

Accepted QAT dataset row fields:

```json
{"prompt": "Question or instruction", "answer": "Target answer"}
```

Aliases are also accepted: `instruction`, `question`, or `input` for the prompt; `response`, `output`, or `completion` for the answer.

Compare a generated GGUF with a reference GGUF:

```powershell
python scripts/07_compare_gguf.py --candidate path\to\your.gguf --reference path\to\reference.gguf
```
