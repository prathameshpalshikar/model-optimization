Project Title
# Edge LLM Optimization Pipeline
1. Objective
This project focuses on reducing the size and runtime memory of a multilingual language model for deployment on memory-constrained edge devices.

Starting point:
- ~2B parameter model

Target constraints:
- Model size ≤ 1GB
- Runtime memory ≤ 2GB
- CPU-only inference
2. Approach
The optimization pipeline follows a strict sequence:

1. Baseline measurement
2. Structural pruning (10–20%)
3. 4-bit quantization
4. Quantization-aware fine-tuning (QAT-lite)
5. Runtime optimization (KV cache + context control)

Each step is validated before proceeding.
3. Repository Structure
scripts/
- 01_baseline.py
- 02_pruning.py
- 03_quantize.py
- 04_qat_lite.py
- 05_runtime_opt.py
- 06_report.py

reports/
- performance comparisons

configs/
- model + experiment configs
4. Key Design Decisions
- No distillation dataset used
- Synthetic data used for QAT-lite
- CPU-first deployment target
- llama.cpp used for final inference benchmarking
5. Important Note on Model Files
Large model files are NOT included in this repository.

Excluded:
- *.gguf
- *.safetensors
- checkpoints and binaries

Reason:
- GitHub size limits (max ~2GB per file)
- Models are treated as artifacts, not source code

To reproduce:
- Download or generate models locally
- Place them in /models or /outputs (ignored by Git)
6. How to Run
1. Clone repo
2. Install dependencies
3. Run scripts in order:

python scripts/01_baseline.py
python scripts/02_pruning.py
...
7. Benchmarking
Final models are tested using llama.cpp (CPU inference).

Metrics:
- tokens/sec
- RAM usage
- multilingual output quality
8. Limitations
- Quantization introduces quality degradation
- No large-scale fine-tuning
- Performance depends on CPU capability
9. Future Work
- Better multilingual recovery after quantization
- Explore mixed precision strategies
- Evaluate alternative pruning methods