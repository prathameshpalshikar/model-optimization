# GGUF Benchmark: Local Quantized Model vs Unsloth Q4_K_M

Generated: 2026-04-18

## Models Compared

| Role | Path | Size |
|---|---|---:|
| Candidate | `artifacts/gguf/model_Q4_K_M_20260416_174022.gguf` | 1215.359 MiB |
| Reference | `C:\Users\prath\Documents\unsloth-qwen3.5\models\qwen2b\Qwen3.5-2B-Q4_K_M.gguf` | 1221.500 MiB |

Candidate size delta:

- Candidate is 6.141 MiB smaller than the Unsloth reference.
- Candidate is 99.497% of the Unsloth reference size.

Important local note:

- `artifacts/gguf/latest_gguf_path.txt` still points to `C:\Users\Admin\Documents\model-optimization\artifacts\gguf\model_Q4_K_M_20260416_174022.gguf`.
- Benchmarks below used the explicit local candidate path, not the stale latest pointer.

## Metadata Comparison

Metadata comparison report:

- `reports/benchmarks/gguf_compare_20260418_232737.json`

Shared properties:

| Field | Candidate | Unsloth |
|---|---:|---:|
| GGUF version | 3 | 3 |
| Architecture | `qwen35` | `qwen35` |
| Tensor count | 320 | 320 |
| `general.file_type` | 15 | 15 |
| Quantization version | 2 | 2 |

Model metadata differences:

| Field | Candidate | Unsloth |
|---|---|---|
| `general.name` | `Qat_Lite_Merged_20260416_165810` | `Qwen3.5-2B` |
| `general.size_label` | `1.9B` | `2B` |
| `general.quantized_by` | not present | `Unsloth` |
| Base model metadata | sparse | includes Qwen base-model org/repo/license metadata |

Tensor quantization mix:

| Tensor type | Candidate | Unsloth |
|---|---:|---:|
| `F32` | 133 | 133 |
| `Q4_K` | 162 | 98 |
| `Q5_K` | 0 | 36 |
| `Q6_K` | 25 | 17 |
| `Q8_0` | 0 | 36 |

Interpretation:

- The candidate and Unsloth files are the same architecture and tensor count, but not the same quantization recipe.
- The candidate uses a simpler mix: mostly `Q4_K` plus some `Q6_K`.
- Unsloth keeps selected tensors at `Q5_K` and `Q8_0`, likely preserving more precision in sensitive tensors.
- The candidate is slightly smaller, consistent with the heavier use of `Q4_K`.

## Runtime Benchmark

Runtime binary:

- `vendor/llama.cpp-bin/llama-bench.exe`

llama.cpp build:

- Build commit: `23b8cc499`
- Build number: `8838`
- Backend: CPU
- CPU: Intel(R) Core(TM) i7-6600U CPU @ 2.60GHz

Benchmark settings:

| Setting | Value |
|---|---:|
| Threads | 8 |
| GPU layers | 0 |
| Prompt tokens | 128 |
| Generated tokens | 32 |
| Repetitions | 1 |
| Batch size | 2048 |
| Micro-batch size | 512 |
| KV cache type K | `f16` |
| KV cache type V | `f16` |
| mmap | enabled |
| Flash attention | disabled |

Commands:

```powershell
vendor\llama.cpp-bin\llama-bench.exe -m artifacts\gguf\model_Q4_K_M_20260416_174022.gguf -t 8 -ngl 0 -p 128 -n 32 -r 1 -o json
vendor\llama.cpp-bin\llama-bench.exe -m C:\Users\prath\Documents\unsloth-qwen3.5\models\qwen2b\Qwen3.5-2B-Q4_K_M.gguf -t 8 -ngl 0 -p 128 -n 32 -r 1 -o json
```

Throughput results:

| Metric | Candidate | Unsloth | Difference |
|---|---:|---:|---:|
| Prompt processing | 11.798 tok/s | 9.183 tok/s | Candidate +28.47% |
| Token generation | 2.677 tok/s | 2.692 tok/s | Candidate -0.56% |

Raw model size reported by llama-bench:

| Model | llama-bench model size |
|---|---:|
| Candidate | 1,263,435,008 bytes |
| Unsloth | 1,269,873,920 bytes |

Interpretation:

- The candidate is materially faster in prompt processing on this CPU-only benchmark.
- Decode/generation speed is effectively tied; Unsloth is only about 0.56% faster.
- The candidate is slightly smaller both by filesystem size and by llama-bench reported model size.
- The speed result is plausible because the candidate uses more `Q4_K` tensors, while Unsloth retains more `Q5_K` and `Q8_0` tensors.

## Memory Footprint Benchmark

Memory benchmark artifact:

- `reports/benchmarks/gguf_memory_20260419.json`

Measurement method:

- Each model was launched separately through `llama-bench.exe`.
- The benchmark workload was the same CPU-only run used above: 8 threads, 0 GPU layers, 128 prompt tokens, 32 generated tokens, 1 repetition, mmap enabled.
- While `llama-bench.exe` was running, Windows process counters were sampled every 200 ms.
- The recorded memory values are the peak observed values for that process during the run.

Equivalent benchmark command shape:

```powershell
vendor\llama.cpp-bin\llama-bench.exe -m <model.gguf> -t 8 -ngl 0 -p 128 -n 32 -r 1 -o json
```

Measured memory footprint:

| Metric | Candidate | Unsloth | Difference |
|---|---:|---:|---:|
| Peak working set | 1919.688 MB | 1828.719 MB | Candidate +90.969 MB (+4.98%) |
| Peak private bytes | 1034.629 MB | 937.711 MB | Candidate +96.918 MB (+10.34%) |
| Peak paged memory | 1034.629 MB | 937.711 MB | Candidate +96.918 MB (+10.34%) |
| Elapsed benchmark time | 27.134 s | 25.166 s | Candidate +1.968 s |

Metric meaning:

- Peak working set is the best practical RAM-footprint signal here. It measures the process-resident memory pages observed by Windows.
- Peak private bytes measures memory committed privately by the benchmark process. This excludes shared mappings and is useful for seeing process-owned allocation pressure.
- Peak paged memory matched private bytes in this run, so it does not add a separate signal for these two measurements.

Interpretation:

- The candidate is smaller on disk, but it used more resident memory during this benchmark run.
- Candidate peak working set was about 4.98% higher than the Unsloth reference.
- Candidate private committed memory was about 10.34% higher than the Unsloth reference.
- This can happen even when the GGUF file is smaller because runtime memory is affected by tensor layout, quantization mix, mmap page residency, scratch buffers, KV cache behavior, and backend scheduling.
- The candidate's heavier `Q4_K` mix reduces file size, but Unsloth's mixed `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` layout may produce a different memory-residency pattern under llama.cpp.

Benchmark caveats:

- This was a single repetition, so the result should be treated as an initial memory-footprint snapshot.
- The sampling interval was 200 ms, so very short allocation spikes could be missed.
- These are Windows process counters, not full-system before/after RAM readings.
- For final reporting, rerun with `-r 3` or `-r 5` and compare median peak memory, average throughput, and standard deviation.

## Generation Smoke Test

Attempted prompt:

```text
What is the capital of France? Answer in one sentence.
```

Both models loaded and began generating through `llama-cli.exe`. The first visible generated content for both began with reasoning text:

```text
[Start thinking]
Thinking Process:
```

The `llama-cli.exe` runs then entered interactive conversation mode and timed out while waiting for further input. This means the smoke test confirms loading and initial generation, but it is not a clean automated quality benchmark.

For future automated quality checks, use one of these instead:

- `llama-completion.exe` for non-interactive completion.
- A wrapper that sends `/exit` after generation.
- `llama-perplexity.exe` against a fixed evaluation text file.

## Current Conclusion

The candidate model is competitive with the Unsloth Q4_K_M reference on CPU runtime:

- Smaller by 6.141 MiB.
- Faster prompt processing in this run.
- Practically tied on generation throughput.
- Higher observed runtime memory footprint in this single memory benchmark: +90.969 MB peak working set and +96.918 MB peak private bytes.

The remaining open question is quality. Because Unsloth uses a richer quantization mix (`Q5_K` and `Q8_0` on selected tensors), quality/perplexity may favor Unsloth even though speed and size are close. The next benchmark should measure perplexity and task-level answer quality on the same prompts or dataset.
