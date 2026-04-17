"""
PHASE 3: GGUF EXPORT + Q4_K_M QUANTIZATION
Convert the HF checkpoint to GGUF and quantize it for llama.cpp.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

from utils import resolve_artifact_pointer, write_artifact_pointer

ROOT = Path(__file__).parent.parent
REPORT_DIR = ROOT / "reports" / "quantization"
ARTIFACT_DIR = ROOT / "artifacts" / "gguf"
VENDOR_DIR = ROOT / "vendor"
LLAMA_SRC = VENDOR_DIR / "llama.cpp"
LLAMA_BIN = VENDOR_DIR / "llama.cpp-bin"
QAT_MERGED_LATEST = ROOT / "artifacts" / "qat_lite" / "latest_qat_merged_path.txt"
PRUNED_LATEST = ROOT / "artifacts" / "pruned" / "latest_pruned_path.txt"
LLAMA_CPP_RELEASE_TAG = os.environ.get("LLAMA_CPP_RELEASE_TAG")


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def run_checked(args: list[str], cwd: Path | None = None):
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed


def resolve_input_model() -> str:
    qat_model = resolve_artifact_pointer(QAT_MERGED_LATEST)
    if qat_model is not None and qat_model.exists():
        return str(qat_model)
    pruned_model = resolve_artifact_pointer(PRUNED_LATEST)
    if pruned_model is not None and pruned_model.exists():
        return str(pruned_model)
    return "Qwen/Qwen3.5-2B"


def ensure_llama_cpp_source():
    if LLAMA_SRC.exists():
        return
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    args = ["git", "clone", "--depth", "1"]
    if LLAMA_CPP_RELEASE_TAG:
        args.extend(["--branch", LLAMA_CPP_RELEASE_TAG])
    args.extend(["https://github.com/ggml-org/llama.cpp.git", str(LLAMA_SRC)])
    run_checked(args)


def latest_cpu_zip_asset() -> tuple[str, str]:
    release_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    if LLAMA_CPP_RELEASE_TAG:
        release_url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_RELEASE_TAG}"
    response = requests.get(release_url, timeout=60)
    response.raise_for_status()
    release = response.json()
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if re.search(r"bin-win-cpu-x64\.zip$", name):
            return name, asset["browser_download_url"]
    raise RuntimeError("Could not find a Windows CPU llama.cpp binary zip in the latest release.")


def ensure_llama_cpp_binaries():
    llama_cli = LLAMA_BIN / "llama-cli.exe"
    llama_quantize = LLAMA_BIN / "llama-quantize.exe"
    if llama_cli.exists() and llama_quantize.exists():
        return

    LLAMA_BIN.mkdir(parents=True, exist_ok=True)
    asset_name, asset_url = latest_cpu_zip_asset()
    zip_path = LLAMA_BIN / asset_name
    if not zip_path.exists():
        with requests.get(asset_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(LLAMA_BIN)


def find_binary(name: str) -> Path:
    matches = sorted(LLAMA_BIN.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Missing required llama.cpp binary: {name}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Export HF checkpoint to GGUF and run llama.cpp quantization.")
    parser.add_argument("--quant-type", default=os.environ.get("GGUF_QUANT_TYPE", "Q4_K_M"))
    parser.add_argument("--input-model", default=os.environ.get("QUANT_INPUT_MODEL"))
    args_cli = parser.parse_args()

    print("=" * 80)
    print(f"PHASE 3: GGUF EXPORT + {args_cli.quant_type} QUANTIZATION")
    print("=" * 80)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_slug()

    input_model = args_cli.input_model or resolve_input_model()
    ensure_llama_cpp_source()
    ensure_llama_cpp_binaries()

    converter = LLAMA_SRC / "convert_hf_to_gguf.py"
    llama_quantize = find_binary("llama-quantize.exe")
    raw_gguf = ARTIFACT_DIR / f"model_f16_{stamp}.gguf"
    quantized_gguf = ARTIFACT_DIR / f"model_{args_cli.quant_type}_{stamp}.gguf"

    print(f"Input HF model: {input_model}")
    print("Converting HF checkpoint to GGUF...")
    convert_result = run_checked(
        [
            sys.executable,
            str(converter),
            input_model,
            "--outtype",
            "f16",
            "--outfile",
            str(raw_gguf),
        ],
        cwd=ROOT,
    )

    print(f"Quantizing GGUF to {args_cli.quant_type}...")
    quant_result = run_checked(
        [
            str(llama_quantize),
            str(raw_gguf),
            str(quantized_gguf),
            args_cli.quant_type,
        ],
        cwd=ROOT,
    )

    report = {
        "phase": "gguf_quantization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_model": input_model,
        "llama_cpp_source_dir": str(LLAMA_SRC),
        "llama_cpp_bin_dir": str(LLAMA_BIN),
        "llama_cpp_release_tag": LLAMA_CPP_RELEASE_TAG or "latest",
        "raw_gguf_path": str(raw_gguf),
        "raw_gguf_size_mb": file_size_mb(raw_gguf),
        "quantized_gguf_path": str(quantized_gguf),
        "quantized_gguf_size_mb": file_size_mb(quantized_gguf),
        "quantization_method": args_cli.quant_type,
        "quantization_impl": "llama.cpp llama-quantize post-training GGUF quantization",
        "quantization_is_applied": True,
        "conversion_stdout_tail": convert_result.stdout[-4000:],
        "conversion_stderr_tail": convert_result.stderr[-4000:],
        "quantize_stdout_tail": quant_result.stdout[-4000:],
        "quantize_stderr_tail": quant_result.stderr[-4000:],
    }
    report_file = REPORT_DIR / f"quantization_report_{stamp}.json"
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    write_artifact_pointer(ARTIFACT_DIR / "latest_gguf_path.txt", quantized_gguf)
    write_artifact_pointer(ARTIFACT_DIR / "latest_raw_gguf_path.txt", raw_gguf)

    print(f"Raw GGUF saved to: {raw_gguf}")
    print(f"Quantized GGUF saved to: {quantized_gguf}")
    print(f"Quantized size: {file_size_mb(quantized_gguf):.2f} MB")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
