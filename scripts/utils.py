"""
Shared utilities for model optimization pipeline.
"""

import os
import json
import gc
import re
import tracemalloc
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except Exception:
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# Memory & Resource Monitoring
# ============================================================================

def start_memory_tracking():
    """Start CPU memory tracking with tracemalloc."""
    tracemalloc.start()
    gc.collect()

def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB during execution."""
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)

def get_current_memory_mb() -> float:
    """Get current memory usage in MB."""
    if psutil is None:
        raise RuntimeError("psutil is required for current memory measurement")
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

def cleanup_resources():
    """Cleanup: clear cache, garbage collect, reset memory tracking."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.1)  # Brief pause to allow cleanup

# ============================================================================
# Artifact Path Handling
# ============================================================================

def write_artifact_pointer(pointer_file: str | Path, target_path: str | Path, root: Path = PROJECT_ROOT):
    """Write a portable latest-artifact pointer, using a repo-relative path when possible."""
    pointer = Path(pointer_file)
    target = Path(target_path)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = str(target.resolve().relative_to(root.resolve()))
    except ValueError:
        text = str(target)
    pointer.write_text(text, encoding="utf-8")

def resolve_artifact_pointer(pointer_file: str | Path, root: Path = PROJECT_ROOT) -> Path | None:
    """Resolve a latest-artifact pointer written as absolute, repo-relative, or pointer-relative path."""
    pointer = Path(pointer_file)
    if not pointer.exists():
        return None
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    candidate = Path(raw)
    candidates = [candidate] if candidate.is_absolute() else [root / candidate, pointer.parent / candidate, candidate]
    for item in candidates:
        if item.exists():
            return item.resolve()
    return candidates[0]

# ============================================================================
# Throughput Measurement
# ============================================================================

def measure_throughput(model, tokenizer, prompts: List[str], max_new_tokens: int = 100) -> Dict[str, float]:
    """
    Measure throughput (tokens/sec) on list of prompts.
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompts: List of prompts to test
        max_new_tokens: Max tokens to generate per prompt
        
    Returns:
        Dict with keys: 'throughput_tps', 'latency_per_token_ms', 'avg_new_tokens'
    """
    times = []
    token_counts = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        elapsed = time.time() - start
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        
        times.append(elapsed)
        token_counts.append(num_tokens)
    
    total_tokens = sum(token_counts)
    total_time = sum(times)
    
    return {
        "throughput_tps": total_tokens / total_time if total_time > 0 else 0,
        "latency_per_token_ms": (total_time / total_tokens * 1000) if total_tokens > 0 else 0,
        "avg_new_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "total_time_sec": total_time
    }

# ============================================================================
# Multilingual Evaluation
# ============================================================================

MULTILINGUAL_PROMPTS = {
    "english": [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot.",
        "How do plants convert sunlight into energy?",
        "What are the benefits of regular exercise?"
    ],
    "hindi": [
        "भारत की राजधानी कौन सी है?",
        "2 + 2 का उत्तर क्या है?",
    ],
    "marathi": [
        "महाराष्ट्राचे मुख्यालय कोणते आहे?",
        "शिक्षीच्या महत्वाबद्दल सांगा.",
    ],
    "telugu": [
        "తెలుగు ప్రాంతం ఏ దేశంలో ఉంది?",
        "నీటి ప్రాముఖ్యత గురించి చెప్పండి.",
    ]
}

def get_multilingual_eval_set() -> Dict[str, List[str]]:
    """Return standard multilingual evaluation prompts."""
    return MULTILINGUAL_PROMPTS

EXPECTED_SHORT_ANSWERS = {
    "english": {
        0: ["paris"],
    },
    "hindi": {
        0: ["नई दिल्ली", "दिल्ली"],
        1: ["4", "चार"],
    },
    "marathi": {
        0: ["मुंबई"],
    },
    "telugu": {
        0: ["భారతదేశం", "ఇండియా", "india"],
    },
}

SCRIPT_RANGES = {
    "hindi": (0x0900, 0x097F),
    "marathi": (0x0900, 0x097F),
    "telugu": (0x0C00, 0x0C7F),
}

def has_repetition_loop(text: str, ngram: int = 4, max_repeats: int = 3) -> bool:
    """Detect simple repeated lines or repeated token n-grams in generated text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counts = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1
        if line_counts[line] >= max_repeats:
            return True

    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) < ngram * max_repeats:
        return False
    counts = {}
    for idx in range(0, len(tokens) - ngram + 1):
        key = tuple(tokens[idx : idx + ngram])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= max_repeats:
            return True
    return False

def contains_language_script(text: str, language: str) -> bool:
    """Return True when output contains the expected script for Indic-language prompts."""
    bounds = SCRIPT_RANGES.get(language)
    if bounds is None:
        return True
    start, end = bounds
    return any(start <= ord(char) <= end for char in text)

def assess_generation(text: str, language: str, prompt_index: int | None = None) -> Dict[str, Any]:
    """Score a lightweight generation gate without requiring a full benchmark suite."""
    checks = {
        "non_empty": bool(text.strip()),
        "has_expected_script": contains_language_script(text, language),
        "has_repetition_loop": has_repetition_loop(text),
        "contains_expected_answer": True,
    }
    expected = EXPECTED_SHORT_ANSWERS.get(language, {}).get(prompt_index)
    if expected:
        lowered = text.lower()
        checks["contains_expected_answer"] = any(answer.lower() in lowered for answer in expected)
        checks["expected_answers"] = expected
    checks["ok"] = (
        checks["non_empty"]
        and checks["has_expected_script"]
        and not checks["has_repetition_loop"]
        and checks["contains_expected_answer"]
    )
    return checks

def generate_and_save_outputs(model, tokenizer, output_file: str, 
                             max_new_tokens: int = 100) -> Dict[str, Any]:
    """
    Generate outputs on all multilingual prompts and save to file.
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        output_file: Path to save outputs
        max_new_tokens: Max tokens per generation
        
    Returns:
        Dict mapping language to list of (prompt, output) tuples
    """
    prompts = get_multilingual_eval_set()
    results = {}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTILINGUAL OUTPUT SAMPLES\n")
        f.write("=" * 80 + "\n\n")
        
        for lang, prompt_list in prompts.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"LANGUAGE: {lang.upper()}\n")
            f.write(f"{'=' * 80}\n\n")
            
            lang_outputs = []
            for i, prompt in enumerate(prompt_list):
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, 
                                           do_sample=False, temperature=0.0)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                f.write(f"Prompt {i+1}: {prompt}\n")
                f.write(f"Output:\n{decoded}\n")
                f.write("-" * 80 + "\n\n")
                
                lang_outputs.append((prompt, decoded))
            
            results[lang] = lang_outputs
    
    return results

# ============================================================================
# Model Configuration & Info
# ============================================================================

def get_model_size_mb(model) -> float:
    """Get model size in MB (sum of all parameters)."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming FP32 = 4 bytes per param, FP16 = 2 bytes
    size_mb = (total_params * 4) / (1024 ** 2)  # Conservative FP32 estimate
    return size_mb

def get_model_config_dict(model, tokenizer) -> Dict[str, Any]:
    """Extract model configuration into dict."""
    return {
        "model_architecture": model.config.model_type if hasattr(model.config, 'model_type') else "unknown",
        "hidden_size": model.config.hidden_size if hasattr(model.config, 'hidden_size') else None,
        "num_layers": model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else None,
        "num_attention_heads": model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else None,
        "vocab_size": model.config.vocab_size if hasattr(model.config, 'vocab_size') else None,
        "max_position_embeddings": model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else None,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "estimated_size_fp32_mb": get_model_size_mb(model),
    }

# ============================================================================
# Metrics Serialization
# ============================================================================

def save_metrics_json(metrics: Dict[str, Any], output_file: str):
    """Save metrics dict to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics_json(input_file: str) -> Dict[str, Any]:
    """Load metrics dict from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_outputs(before_file: str, after_file: str, output_file: str):
    """
    Compare two output files (qualitative assessment).
    This is a placeholder — actual comparison would be manual/visual.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("QUALITATIVE COMPARISON: BEFORE vs AFTER\n")
        f.write("=" * 80 + "\n")
        f.write("Note: Manual inspection required. This file documents the comparison process.\n\n")
        
        try:
            with open(before_file, 'r', encoding='utf-8') as bf:
                before_content = bf.read()
            with open(after_file, 'r', encoding='utf-8') as af:
                after_content = af.read()
            
            f.write("Files loaded successfully for comparison.\n")
            f.write(f"Before file size: {len(before_content)} chars\n")
            f.write(f"After file size: {len(after_content)} chars\n\n")
            f.write("Manual review recommended for:\n")
            f.write("  - Coherence and fluency\n")
            f.write("  - Multilingual quality\n")
            f.write("  - Presence of repetition loops\n")
            f.write("  - Language degradation\n")
        except Exception as e:
            f.write(f"Error during comparison: {e}\n")
