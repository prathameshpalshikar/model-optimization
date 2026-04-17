"""
Compare a locally produced GGUF against a reference GGUF.

The script always reports file/metadata/tensor-layout differences. If a
llama.cpp `llama-cli` binary is available, it also runs both models through the
same prompts and records tokens/sec plus process RSS.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

try:
    import psutil
except Exception:  # psutil is only needed for peak RSS measurement.
    psutil = None

from utils import resolve_artifact_pointer


ROOT = Path(__file__).parent.parent
DEFAULT_CANDIDATE_POINTER = ROOT / "artifacts" / "gguf" / "latest_gguf_path.txt"
DEFAULT_UNSLOTH_GGUF = Path(
    r"C:\Users\prath\Documents\unsloth-qwen3.5\models\qwen2b\Qwen3.5-2B-Q4_K_M.gguf"
)
REPORT_DIR = ROOT / "reports" / "benchmarks"
LLAMA_BIN = ROOT / "vendor" / "llama.cpp-bin"

GGUF_VALUE_TYPES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

GGML_TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "TQ1_0",
    32: "TQ2_0",
}

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain quantization for edge inference in two short sentences.",
    "Write one sentence about why compact language models matter.",
]


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of GGUF file")
    return data


def u32(handle) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def i32(handle) -> int:
    return struct.unpack("<i", read_exact(handle, 4))[0]


def u64(handle) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_str(handle) -> str:
    length = u64(handle)
    return read_exact(handle, length).decode("utf-8", errors="replace")


def read_scalar(handle, value_type: int):
    if value_type == 0:
        return struct.unpack("<B", read_exact(handle, 1))[0]
    if value_type == 1:
        return struct.unpack("<b", read_exact(handle, 1))[0]
    if value_type == 2:
        return struct.unpack("<H", read_exact(handle, 2))[0]
    if value_type == 3:
        return struct.unpack("<h", read_exact(handle, 2))[0]
    if value_type == 4:
        return u32(handle)
    if value_type == 5:
        return i32(handle)
    if value_type == 6:
        return struct.unpack("<f", read_exact(handle, 4))[0]
    if value_type == 7:
        return bool(struct.unpack("<?", read_exact(handle, 1))[0])
    if value_type == 8:
        return read_str(handle)
    if value_type == 10:
        return u64(handle)
    if value_type == 11:
        return struct.unpack("<q", read_exact(handle, 8))[0]
    if value_type == 12:
        return struct.unpack("<d", read_exact(handle, 8))[0]
    raise ValueError(f"Unsupported GGUF scalar value type: {value_type}")


def read_value(handle, value_type: int):
    if value_type != 9:
        return read_scalar(handle, value_type)

    element_type = u32(handle)
    length = u64(handle)
    values = [read_scalar(handle, element_type) for _ in range(length)]
    if len(values) > 16:
        return {
            "array_type": GGUF_VALUE_TYPES.get(element_type, str(element_type)),
            "len": len(values),
            "head": values[:8],
        }
    return values


def parse_gguf(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        version = u32(handle)
        tensor_count = u64(handle)
        metadata_count = u64(handle)

        metadata = {}
        for _ in range(metadata_count):
            key = read_str(handle)
            value_type = u32(handle)
            metadata[key] = read_value(handle, value_type)

        tensor_types = collections.Counter()
        first_tensors = []
        for _ in range(tensor_count):
            name = read_str(handle)
            dims = [u64(handle) for _ in range(u32(handle))]
            tensor_type = GGML_TENSOR_TYPES.get(u32(handle), "UNKNOWN")
            _offset = u64(handle)
            tensor_types[tensor_type] += 1
            if len(first_tensors) < 12:
                first_tensors.append({"name": name, "dims": dims, "type": tensor_type})

    selected_metadata = {}
    for key in sorted(metadata):
        if key.startswith("general.") or key.startswith("qwen35.") or key.startswith("qwen3_5."):
            selected_metadata[key] = metadata[key]
        elif key in {"tokenizer.ggml.model", "tokenizer.chat_template"}:
            value = metadata[key]
            selected_metadata[key] = value[:240] + "..." if isinstance(value, str) and len(value) > 240 else value

    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mib": path.stat().st_size / (1024**2),
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "selected_metadata": selected_metadata,
        "tensor_type_counts": dict(tensor_types),
        "first_tensors": first_tensors,
    }


def find_llama_cli(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    path_from_env = shutil.which("llama-cli.exe") or shutil.which("llama-cli")
    if path_from_env:
        return Path(path_from_env)

    matches = sorted(LLAMA_BIN.rglob("llama-cli.exe")) if LLAMA_BIN.exists() else []
    return matches[0] if matches else None


def parse_tok_per_sec(text: str) -> float | None:
    import re

    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s+tok/s", text)
    return float(matches[-1]) if matches else None


def run_llama_cli(
    llama_cli: Path,
    model_path: Path,
    prompt: str,
    threads: int,
    ctx_size: int,
    max_tokens: int,
) -> dict:
    args = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-t",
        str(threads),
        "-c",
        str(ctx_size),
        "-n",
        str(max_tokens),
        "--temp",
        "0",
        "--no-display-prompt",
        "-p",
        prompt,
    ]
    start = time.time()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss_mb = None

    if psutil is not None:
        proc = psutil.Process(process.pid)
        peak = 0.0
        while process.poll() is None:
            try:
                peak = max(peak, proc.memory_info().rss / (1024**2))
            except psutil.Error:
                pass
            time.sleep(0.1)
        peak_rss_mb = peak

    stdout, stderr = process.communicate()
    elapsed = time.time() - start
    combined = stdout + "\n" + stderr
    return {
        "returncode": process.returncode,
        "elapsed_sec": elapsed,
        "tok_per_sec": parse_tok_per_sec(combined),
        "peak_rss_mb": peak_rss_mb,
        "stdout_preview": stdout[:500],
        "stderr_tail": stderr[-1500:],
    }


def benchmark_model(
    name: str,
    model_path: Path,
    llama_cli: Path | None,
    threads: int,
    ctx_size: int,
    max_tokens: int,
) -> dict:
    result = {"name": name, "metadata": parse_gguf(model_path), "runtime": None}
    if llama_cli is None:
        result["runtime"] = {"skipped": True, "reason": "llama-cli not found"}
        return result

    prompt_runs = []
    for prompt in PROMPTS:
        prompt_runs.append(
            {
                "prompt": prompt,
                **run_llama_cli(llama_cli, model_path, prompt, threads, ctx_size, max_tokens),
            }
        )
    speeds = [run["tok_per_sec"] for run in prompt_runs if run.get("tok_per_sec") is not None]
    peaks = [run["peak_rss_mb"] for run in prompt_runs if run.get("peak_rss_mb") is not None]
    result["runtime"] = {
        "skipped": False,
        "threads": threads,
        "ctx_size": ctx_size,
        "max_tokens": max_tokens,
        "average_tok_per_sec": sum(speeds) / len(speeds) if speeds else None,
        "peak_rss_mb": max(peaks) if peaks else None,
        "prompt_runs": prompt_runs,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local GGUF against a reference GGUF.")
    parser.add_argument("--candidate", help="Your generated GGUF. Defaults to artifacts/gguf/latest_gguf_path.txt")
    parser.add_argument("--reference", default=str(DEFAULT_UNSLOTH_GGUF), help="Reference GGUF, e.g. Unsloth Q4_K_M.")
    parser.add_argument("--llama-cli", help="Path to llama-cli.exe. Defaults to PATH or vendor/llama.cpp-bin.")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads. Default uses os/psutil CPU count.")
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--metadata-only", action="store_true", help="Skip runtime inference even if llama-cli exists.")
    args = parser.parse_args()

    candidate = Path(args.candidate) if args.candidate else resolve_artifact_pointer(DEFAULT_CANDIDATE_POINTER)
    reference = Path(args.reference)
    if candidate is None:
        raise SystemExit("No candidate path supplied and artifacts/gguf/latest_gguf_path.txt is missing or empty.")
    if not candidate.exists():
        raise SystemExit(f"Candidate GGUF does not exist: {candidate}")
    if not reference.exists():
        raise SystemExit(f"Reference GGUF does not exist: {reference}")

    cpu_count = psutil.cpu_count() if psutil is not None else None
    threads = args.threads if args.threads > 0 else max(1, cpu_count or 1)
    llama_cli = None if args.metadata_only else find_llama_cli(args.llama_cli)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llama_cli": str(llama_cli) if llama_cli else None,
        "candidate": benchmark_model("candidate", candidate, llama_cli, threads, args.ctx_size, args.max_tokens),
        "reference": benchmark_model("reference", reference, llama_cli, threads, args.ctx_size, args.max_tokens),
    }

    cand_size = report["candidate"]["metadata"]["size_mib"]
    ref_size = report["reference"]["metadata"]["size_mib"]
    report["comparison"] = {
        "candidate_size_mib": cand_size,
        "reference_size_mib": ref_size,
        "candidate_minus_reference_mib": cand_size - ref_size,
        "candidate_size_pct_of_reference": (cand_size / ref_size * 100) if ref_size else None,
        "tensor_type_counts_equal": report["candidate"]["metadata"]["tensor_type_counts"]
        == report["reference"]["metadata"]["tensor_type_counts"],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / f"gguf_compare_{timestamp_slug()}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written: {output}")
    print(json.dumps(report["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
