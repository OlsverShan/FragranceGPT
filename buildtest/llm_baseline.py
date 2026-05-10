"""
Baseline 2: LLM Zero-shot Baseline
====================================
Given Main Accords → LLM predicts Top/Middle/Base notes → compare with ground truth.

Supports: DeepSeek, OpenAI, Anthropic (auto-detect via env vars)
[LOCAL MODEL] Fine-tuned Qwen2.5-7B (free, no API key needed)

Requires: pip install openai

Run:
  export DEEPSEEK_API_KEY="sk-xxx"    # or OPENAI_API_KEY (API mode)
  python llm_baseline.py              # uses local model by default
"""

import os
import json
import numpy as np
from utils import load_data, preprocess, evaluate_single, evaluate_single_fuzzy, print_results

# ============================================================
# Local Fine-tuned Model (Qwen2.5-7B LoRA)
# ============================================================

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

def _get_local_model():
    """Lazy-load the fine-tuned model (loads once, reuses)."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is None:
        import sys
        sys.path.insert(0, '.')
        from finetune.inference import load_model, predict as ft_predict, parse_notes
        _LOCAL_MODEL, _LOCAL_TOKENIZER = load_model()
    return _LOCAL_MODEL, _LOCAL_TOKENIZER


def predict_local(accords):
    """Local fine-tuned model: accords → (top_set, mid_set, base_set)."""
    from finetune.inference import predict as ft_predict, parse_notes
    model, tokenizer = _get_local_model()
    output = ft_predict(model, tokenizer, ', '.join(accords))
    return parse_notes(output)


def evaluate_finetuned(df, sample_size=100):
    """Evaluate fine-tuned model on test sample."""
    print(f"  Model: Qwen2.5-7B-LoRA (local)  Samples: {sample_size}")
    sample = df.sample(n=sample_size, random_state=42)
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}

    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']:
            continue
        try:
            pred_top, pred_mid, pred_base = predict_local(row['accords'])
        except Exception as e:
            print(f"  [{i+1}/{sample_size}] Error: {e}")
            pred_top, pred_mid, pred_base = set(), set(), set()

        p_t, r_t, f_t = evaluate_single(pred_top, row['Top_clean'])
        p_m, r_m, f_m = evaluate_single(pred_mid, row['Middle_clean'])
        p_b, r_b, f_b = evaluate_single(pred_base, row['Base_clean'])
        results['top'].append({'precision': p_t, 'recall': r_t, 'f1': f_t})
        results['mid'].append({'precision': p_m, 'recall': r_m, 'f1': f_m})
        results['base'].append({'precision': p_b, 'recall': r_b, 'f1': f_b})

        pred_all = pred_top | pred_mid | pred_base
        true_all = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
        _, _, f_a = evaluate_single(pred_all, true_all)
        results['overall'].append({'precision': 0, 'recall': 0, 'f1': f_a})

        p_tf, r_tf, f_tf = evaluate_single_fuzzy(pred_top, row['Top_clean'])
        p_mf, r_mf, f_mf = evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
        p_bf, r_bf, f_bf = evaluate_single_fuzzy(pred_base, row['Base_clean'])
        results_fuzzy['top'].append({'precision': p_tf, 'recall': r_tf, 'f1': f_tf})
        results_fuzzy['mid'].append({'precision': p_mf, 'recall': r_mf, 'f1': f_mf})
        results_fuzzy['base'].append({'precision': p_bf, 'recall': r_bf, 'f1': f_bf})
        _, _, f_af = evaluate_single_fuzzy(pred_all, true_all)
        results_fuzzy['overall'].append({'precision': 0, 'recall': 0, 'f1': f_af})

        if (i + 1) % 20 == 0:
            f1_exact = np.mean([m['f1'] for m in results['overall']])
            f1_fuzzy = np.mean([m['f1'] for m in results_fuzzy['overall']])
            print(f"  [{i+1}/{sample_size}] Exact F1={f1_exact:.3f}  Fuzzy F1={f1_fuzzy:.3f}")

    return results, results_fuzzy


PROMPT_TEMPLATE = """You are a professional perfumer. Given the main accords of a fragrance, predict the most likely Top, Middle, and Base notes.

Main Accords: {accords}

Reply ONLY with a JSON object (no markdown, no explanation):
{{"top_notes": ["note1", "note2", "note3", "note4", "note5"], "middle_notes": ["note1", "note2", "note3", "note4", "note5"], "base_notes": ["note1", "note2", "note3", "note4", "note5"]}}

Include exactly 5 specific notes per layer. Use standardized note names in lowercase (e.g., "bergamot" not "Bergamot Essential Oil")."""


# # ============================================================
# # DeepSeek API (commented out — use local model instead)
# # ============================================================
#
# def get_client():
#     """Auto-detect provider from env vars and return (client, model, provider_name)."""
#     from openai import OpenAI
#
#     provider = os.environ.get("FRAGRANCE_PROVIDER", "").lower()
#
#     if not provider:
#         if os.environ.get("DEEPSEEK_API_KEY"):
#             provider = "deepseek"
#         elif os.environ.get("OPENAI_API_KEY"):
#             provider = "openai"
#         else:
#             return None, None, None
#
#     if provider == "deepseek":
#         client = OpenAI(
#             api_key=os.environ["DEEPSEEK_API_KEY"],
#             base_url="https://api.deepseek.com",
#         )
#         return client, "deepseek-chat", "deepseek"
#     elif provider == "openai":
#         client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#         return client, "gpt-4o", "openai"
#     else:
#         return None, None, None
#
#
# def predict(client, model, accords):
#     """Single LLM call: accords → (top_set, mid_set, base_set)."""
#     prompt = PROMPT_TEMPLATE.format(accords=", ".join(accords))
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#         temperature=0.3,
#         max_tokens=500,
#     )
#     raw = response.choices[0].message.content.strip()
#
#     if raw.startswith("```"):
#         raw = raw.split("\n", 1)[1]
#         if raw.endswith("```"):
#             raw = raw[:-3]
#         raw = raw.strip()
#
#     result = json.loads(raw)
#     return (
#         set(n.strip().lower() for n in result.get("top_notes", [])),
#         set(n.strip().lower() for n in result.get("middle_notes", [])),
#         set(n.strip().lower() for n in result.get("base_notes", [])),
#     )
#
#
# def evaluate_llm(df, sample_size=50):
#     client, model, provider = get_client()
#     if client is None:
#         print("  No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
#         return None
#
#     print(f"  Provider: {provider}  Model: {model}  Samples: {sample_size}")
#
#     sample = df.sample(n=sample_size, random_state=42)
#     results = {'top': [], 'mid': [], 'base': [], 'overall': []}
#     results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}
#
#     for i, (_, row) in enumerate(sample.iterrows()):
#         if not row['accords']:
#             continue
#
#         try:
#             pred_top, pred_mid, pred_base = predict(client, model, row['accords'])
#         except Exception as e:
#             print(f"  [{i+1}/{sample_size}] Error: {e}")
#             pred_top, pred_mid, pred_base = set(), set(), set()
#
#         results['top'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single(pred_top, row['Top_clean'])
#         )))
#         results['mid'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single(pred_mid, row['Middle_clean'])
#         )))
#         results['base'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single(pred_base, row['Base_clean'])
#         )))
#
#         pred_all = pred_top | pred_mid | pred_base
#         true_all = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
#         results['overall'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single(pred_all, true_all)
#         )))
#
#         results_fuzzy['top'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single_fuzzy(pred_top, row['Top_clean'])
#         )))
#         results_fuzzy['mid'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
#         )))
#         results_fuzzy['base'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single_fuzzy(pred_base, row['Base_clean'])
#         )))
#         results_fuzzy['overall'].append(dict(zip(
#             ['precision', 'recall', 'f1'],
#             evaluate_single_fuzzy(pred_all, true_all)
#         )))
#
#         if (i + 1) % 10 == 0:
#             running_f1 = np.mean([m['f1'] for m in results['overall']])
#             print(f"  [{i+1}/{sample_size}] running overall F1: {running_f1:.3f}")
#
#     return results, results_fuzzy


def main():
    print("=" * 60)
    print("  Baseline 2: Zero-shot Baseline")
    print("  [LOCAL] Fine-tuned Qwen2.5-7B (no API required)")
    print("=" * 60)

    print("\n[1/2] Loading data...")
    df = load_data()
    df = preprocess(df)
    print(f"  Loaded {len(df):,} perfumes")

    print("\n[2/2] Running local fine-tuned model evaluation...")
    results, results_fuzzy = evaluate_finetuned(df, sample_size=100)
    if results:
        print_results(results, "Fine-tuned Qwen2.5-7B (Exact Match)")
        print_results(results_fuzzy, "Fine-tuned Qwen2.5-7B (Fuzzy Match)")

        overall_f1 = np.mean([m['f1'] for m in results_fuzzy['overall']])
        print(f"\n  {'='*50}")
        print(f"  Cross-Baseline Comparison")
        print(f"  {'='*50}")
        print(f"  Random Guessing:     0.006 / 0.006")
        print(f"  Frequency Baseline:  0.299 / 0.327")
        print(f"  LLM Zero-shot:       0.337 / 0.393")
        print(f"  LLM + RAG:           0.406 / 0.443")
        print(f"  Fine-tuned Qwen-7B:  {np.mean([m['f1'] for m in results['overall']]):.3f} / {overall_f1:.3f}")
        print(f"")
        print(f"  # To use DeepSeek API instead, uncomment the section below:")
        print(f"  # export DEEPSEEK_API_KEY=sk-xxx && python llm_baseline.py")


# # ============================================================
# # DeepSeek API main() — uncomment to use
# # ============================================================
# def main():
#     print("=" * 60)
#     print("  Baseline 2: LLM Zero-shot Baseline")
#     print("=" * 60)
#
#     print("\n[1/2] Loading data...")
#     df = load_data()
#     df = preprocess(df)
#     print(f"  Loaded {len(df):,} perfumes")
#
#     print("\n[2/2] Running LLM evaluation...")
#     results, results_fuzzy = evaluate_llm(df, sample_size=50)
#     if results:
#         print_results(results, "LLM Zero-shot (Exact Match)")
#         print_results(results_fuzzy, "LLM Zero-shot (Fuzzy Match)")
#         print(f"\n  Frequency baseline = 0.299, Random = 0.006")


if __name__ == "__main__":
    main()
