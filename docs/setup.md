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
