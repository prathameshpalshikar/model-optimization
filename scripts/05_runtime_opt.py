"""
PHASE 5: LLAMA.CPP RUNTIME VALIDATION
Validate the GGUF artifact with llama.cpp under edge-style settings.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import psutil

from utils import get_multilingual_eval_set, resolve_artifact_pointer, write_artifact_pointer

ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = ROOT / "artifacts" / "runtime"
REPORT_DIR = ROOT / "reports" / "runtime"
GGUF_LATEST = ROOT / "artifacts" / "gguf" / "latest_gguf_path.txt"
LLAMA_BIN = ROOT / "vendor" / "llama.cpp-bin"


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def find_binary(name: str) -> Path:
    matches = sorted(LLAMA_BIN.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Missing llama.cpp binary: {name}")
    return matches[0]

def parse_tok_per_sec(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+tok/s", text)
    return float(match.group(1)) if match else None


def safe_json_load_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def build_jsonl_prompt(questions: dict[str, list[str]]) -> str:
    """
    Build a single prompt that asks the model to return JSONL only.

    Expected output format (one JSON object per line, no surrounding text):
      {"lang":"english","idx":1,"answer":"..."}
      {"lang":"hindi","idx":1,"answer":"..."}
      ...
    """
    lines = []
    lines.append("You are a multilingual assistant.")
    lines.append("Answer the provided questions in the language specified per line.")
    lines.append("Return ONLY JSONL: one JSON object per line, no markdown, no extra text.")
    lines.append("JSON schema per line: {\"lang\":<lang>,\"idx\":<1-based integer>,\"answer\":<string>}")
    lines.append("Keep each answer under 20 tokens.")
    lines.append("")
    for lang in ["english", "hindi", "marathi", "telugu"]:
        qlist = questions.get(lang, [])
        for idx, q in enumerate(qlist, 1):
            lines.append(f"{lang.upper()}[{idx}]: {q}")
    lines.append("")
    return "\n".join(lines)


def run_llama(prompt: str, gguf_path: str, ctx_size: int = 2048, max_tokens: int = 48, threads: int | None = None) -> dict:
    llama_cli = find_binary("llama-cli.exe")
    if threads is None:
        threads = max(1, psutil.cpu_count() or 1)
    args = [
        str(llama_cli),
        "-m",
        gguf_path,
        "-t",
        str(threads),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "-c",
        str(ctx_size),
        "--temp",
        "0",
        "--no-display-prompt",
    ]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps = psutil.Process(process.pid)
    peak_rss = 0.0
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, ps.memory_info().rss / (1024 ** 2))
        except psutil.Error:
            pass
        time.sleep(0.2)
    stdout, stderr = process.communicate()
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "peak_rss_mb": peak_rss,
        "tok_per_sec": parse_tok_per_sec(stdout + "\n" + stderr),
    }


def main():
    print("=" * 80)
    print("PHASE 5: LLAMA.CPP RUNTIME VALIDATION")
    print("=" * 80)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_slug()

    gguf_path_obj = resolve_artifact_pointer(GGUF_LATEST)
    if gguf_path_obj is None or not gguf_path_obj.exists():
        raise FileNotFoundError(f"Missing GGUF artifact pointer or model file: {GGUF_LATEST}")
    gguf_path = str(gguf_path_obj)
    context_cap = 2048
    sliding_window = 256
    runtime_samples = {}
    speeds = []
    failures = []
    peak_rss = 0.0

    # Phase 5 was slow primarily because `llama-cli.exe` is spawned once per prompt,
    # which re-loads the full GGUF model each time. This implementation reduces
    # process/model reload overhead by running only two generations total:
    #   1) a JSONL multi-question generation
    #   2) a short multi-turn stability generation
    multilingual = get_multilingual_eval_set()
    expected_counts = {
        "english": len(multilingual.get("english", [])),
        "hindi": len(multilingual.get("hindi", [])),
        "marathi": len(multilingual.get("marathi", [])),
        "telugu": len(multilingual.get("telugu", [])),
    }

    runtime_samples = {
        lang: [{"ok": False} for _ in range(cnt)] for lang, cnt in expected_counts.items() if cnt > 0
    }

    questions = {lang: multilingual.get(lang, []) for lang in ["english", "hindi", "marathi", "telugu"]}
    jsonl_prompt = build_jsonl_prompt(questions)

    print("Generating multilingual runtime samples (single JSONL invocation)...")
    cpu_threads = max(1, psutil.cpu_count() or 1)
    # JSONL for all prompts; keep max_tokens modest since each answer is short.
    result1 = run_llama(jsonl_prompt, gguf_path, ctx_size=context_cap, max_tokens=220, threads=cpu_threads)
    peak_rss = max(peak_rss, result1["peak_rss_mb"])
    if result1["returncode"] != 0:
        failures.append({"multilingual": True, "error": result1["stderr"][-500:]})
        return_obj = None
    else:
        if result1["tok_per_sec"] is not None:
            speeds.append(result1["tok_per_sec"])
        return_obj = result1["stdout"]

    if return_obj is not None:
        parsed = 0
        for raw_line in return_obj.splitlines():
            obj = safe_json_load_line(raw_line)
            if obj is None:
                continue
            lang = obj.get("lang")
            idx = obj.get("idx")
            answer = obj.get("answer")
            if not isinstance(lang, str) or not isinstance(idx, int) or not isinstance(answer, str):
                continue
            if lang not in runtime_samples:
                continue
            if idx < 1 or idx > len(runtime_samples[lang]):
                continue
            runtime_samples[lang][idx - 1] = {"ok": True, "answer": answer}
            parsed += 1
        # If the model didn't emit all JSONL lines, keep failures/fair signal in output.
        if parsed < sum(cnt for cnt in expected_counts.values()):
            failures.append({"multilingual_jsonl_incomplete": True, "parsed_lines": parsed})

    print("Running shortened multi-turn stability harness (single invocation)...")
    history = "System: Reply briefly and stay on topic."
    turns = 2
    for turn in range(1, turns + 1):
        user_msg = f"\nUser: Summarize turn {turn} in one short sentence and include one Hindi word."
        history = (history + user_msg)[-context_cap * 4 :]

    multi_turn_prompt = (
        history
        + "\nAssistant (Turn 1):\n"
        + "Assistant (Turn 2):\n"
        + "Provide the two assistant responses in order, each under 20 tokens."
    )

    result2 = run_llama(multi_turn_prompt, gguf_path, ctx_size=context_cap, max_tokens=48, threads=cpu_threads)
    peak_rss = max(peak_rss, result2["peak_rss_mb"])
    if result2["returncode"] != 0:
        failures.append({"multiturn": True, "error": result2["stderr"][-500:]})
    else:
        if result2["tok_per_sec"] is not None:
            speeds.append(result2["tok_per_sec"])

    report = {
        "phase": "llama_cpp_runtime",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gguf_model_path": gguf_path,
        "gguf_size_mb": Path(gguf_path).stat().st_size / (1024 ** 2),
        "runtime": "llama.cpp",
        "backend": "cpu",
        "context_cap_tokens": context_cap,
        "sliding_window_tokens": sliding_window,
        "sliding_window_applied": False,
        "runtime_notes": "Context is capped with -c. sliding_window_tokens is a target policy value only unless mapped to a supported llama.cpp CLI flag.",
        "stable_inference": len(failures) == 0,
        "failures": failures,
        "peak_runtime_memory_mb": peak_rss,
        "average_tokens_per_sec": sum(speeds) / len(speeds) if speeds else None,
        "multilingual_samples": runtime_samples,
    }
    report_file = REPORT_DIR / f"runtime_report_{stamp}.json"
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    write_artifact_pointer(ARTIFACT_DIR / "latest_runtime_report.txt", report_file)
    print(f"Runtime report saved to: {report_file}")


if __name__ == "__main__":
    main()
