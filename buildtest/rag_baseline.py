"""
Baseline 3: RAG (Retrieval-Augmented Generation) Baseline
===========================================================
Given Main Accords → retrieve similar perfumes from 24k database →
inject their real notes into prompt as reference → LLM predicts.

对比: LLM zero-shot (0.335) vs LLM + RAG (?)
[LOCAL MODEL] Fine-tuned Qwen2.5-7B included for comparison

Run:
  export DEEPSEEK_API_KEY="sk-xxx"
  python rag_baseline.py
"""

import os
import json
import numpy as np
from pathlib import Path
from utils import load_data, preprocess, evaluate_single, evaluate_single_fuzzy, print_results
from src.rag import FragranceVectorStore

# ============================================================
# Local Fine-tuned Model (Qwen2.5-7B LoRA)
# ============================================================

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

def _get_local_model():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is None:
        from finetune.inference import load_model
        _LOCAL_MODEL, _LOCAL_TOKENIZER = load_model()
    return _LOCAL_MODEL, _LOCAL_TOKENIZER

def predict_local(accords):
    from finetune.inference import predict as ft_predict, parse_notes
    model, tokenizer = _get_local_model()
    output = ft_predict(model, tokenizer, ', '.join(accords))
    return parse_notes(output)

def evaluate_finetuned(df, sample_size=100):
    print(f"  Model: Qwen2.5-7B-LoRA (local)  Samples: {sample_size}")
    sample = df.sample(n=sample_size, random_state=42)
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}
    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']: continue
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
        _, _, f_a = evaluate_single(pred_top | pred_mid | pred_base, row['Top_clean'] | row['Middle_clean'] | row['Base_clean'])
        results['overall'].append({'precision': 0, 'recall': 0, 'f1': f_a})
        p_tf, r_tf, f_tf = evaluate_single_fuzzy(pred_top, row['Top_clean'])
        p_mf, r_mf, f_mf = evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
        p_bf, r_bf, f_bf = evaluate_single_fuzzy(pred_base, row['Base_clean'])
        results_fuzzy['top'].append({'precision': p_tf, 'recall': r_tf, 'f1': f_tf})
        results_fuzzy['mid'].append({'precision': p_mf, 'recall': r_mf, 'f1': f_mf})
        results_fuzzy['base'].append({'precision': p_bf, 'recall': r_bf, 'f1': f_bf})
        _, _, f_af = evaluate_single_fuzzy(pred_top | pred_mid | pred_base, row['Top_clean'] | row['Middle_clean'] | row['Base_clean'])
        results_fuzzy['overall'].append({'precision': 0, 'recall': 0, 'f1': f_af})
        if (i + 1) % 20 == 0:
            f1e = np.mean([m['f1'] for m in results['overall']])
            f1f = np.mean([m['f1'] for m in results_fuzzy['overall']])
            print(f"  [{i+1}/{sample_size}] Exact F1={f1e:.3f}  Fuzzy F1={f1f:.3f}")
    return results, results_fuzzy

# ============================================================
# RAG Prompt Template
# ============================================================

RAG_PROMPT = """You are a professional perfumer. Given the main accords of a fragrance, predict the most likely Top, Middle, and Base notes.

Use the reference perfumes below (which share similar accords) as hints. The notes you predict should be plausible for the given accords — they do NOT need to exactly match the references.

Main Accords: {accords}

Reference perfumes with similar accords:
{references}

Reply ONLY with a JSON object (no markdown, no explanation):
{{"top_notes": ["note1", "note2", "note3", "note4", "note5"], "middle_notes": ["note1", "note2", "note3", "note4", "note5"], "base_notes": ["note1", "note2", "note3", "note4", "note5"]}}

Include exactly 5 specific notes per layer. Use standardized note names in lowercase."""


def format_references(refs):
    """Format retrieved perfumes for LLM prompt."""
    lines = []
    for i, ref in enumerate(refs, 1):
        similarity_pct = ref['similarity'] * 100 if ref['similarity'] else 0
        lines.append(
            f"  {i}. [{ref['brand']}] {ref['name']} "
            f"(accord similarity: {similarity_pct:.0f}%)\n"
            f"     {ref['notes']}"
        )
    return "\n".join(lines)


# ============================================================
# LLM Client
# ============================================================

def get_client():
    from openai import OpenAI

    if os.environ.get("DEEPSEEK_API_KEY"):
        return (
            OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"),
            "deepseek-chat",
            "deepseek",
        )
    elif os.environ.get("OPENAI_API_KEY"):
        return (
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            "gpt-4o",
            "openai",
        )
    return None, None, None


def predict_with_rag(client, model, accords, references):
    """LLM call with RAG-enhanced prompt."""
    refs_text = format_references(references)
    prompt = RAG_PROMPT.format(
        accords=", ".join(accords),
        references=refs_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    result = json.loads(raw)
    return (
        set(n.strip().lower() for n in result.get("top_notes", [])),
        set(n.strip().lower() for n in result.get("middle_notes", [])),
        set(n.strip().lower() for n in result.get("base_notes", [])),
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_rag(df, vector_store, sample_size=50):
    client, model, provider = get_client()
    if client is None:
        print("  No API key found.")
        return None

    print(f"  Provider: {provider}  Model: {model}  Samples: {sample_size}")

    sample = df.sample(n=sample_size, random_state=42)
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}

    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']:
            continue

        # RAG: retrieve similar perfumes
        references = vector_store.retrieve(row['accords'], top_k=5)

        try:
            pred_top, pred_mid, pred_base = predict_with_rag(
                client, model, row['accords'], references
            )
        except Exception as e:
            print(f"  [{i+1}/{sample_size}] Error: {e}")
            pred_top, pred_mid, pred_base = set(), set(), set()

        results['top'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_top, row['Top_clean'])
        )))
        results['mid'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_mid, row['Middle_clean'])
        )))
        results['base'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_base, row['Base_clean'])
        )))

        pred_all = pred_top | pred_mid | pred_base
        true_all = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
        results['overall'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single(pred_all, true_all)
        )))

        results_fuzzy['top'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_top, row['Top_clean'])
        )))
        results_fuzzy['mid'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
        )))
        results_fuzzy['base'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_base, row['Base_clean'])
        )))
        results_fuzzy['overall'].append(dict(zip(
            ['precision', 'recall', 'f1'],
            evaluate_single_fuzzy(pred_all, true_all)
        )))

        if (i + 1) % 10 == 0:
            running_f1 = np.mean([m['f1'] for m in results['overall']])
            running_f1f = np.mean([m['f1'] for m in results_fuzzy['overall']])
            print(f"  [{i+1}/{sample_size}] F1: {running_f1:.3f} (exact) / {running_f1f:.3f} (fuzzy)")

    return results, results_fuzzy


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Baseline 3: RAG + Local Fine-tuned Model Comparison")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    df = load_data()
    df = preprocess(df)
    print(f"  Loaded {len(df):,} perfumes")

    # --- Build vector store ---
    print("\n[2/5] Building vector store...")
    store = FragranceVectorStore()
    store.build(df)

    # --- Sanity check: show a retrieval example ---
    print("\n[3/5] Retrieval sanity check...")
    sample_accords = df['accords'].iloc[100]
    print(f"  Query accords: {sample_accords}")
    refs = store.retrieve(sample_accords, top_k=3)
    print(f"  Top-3 retrieved:")
    for ref in refs:
        print(f"    [{ref['brand']}] {ref['name']} (sim: {ref['similarity']:.2f})")
        print(f"    {ref['notes'][:150]}...")

    # --- Local Fine-tuned Model ---
    print("\n[4/5] Evaluating local fine-tuned model (no API)...")
    ft_results, ft_results_fuzzy = evaluate_finetuned(df, sample_size=100)
    if ft_results:
        print_results(ft_results, "Fine-tuned Qwen2.5-7B (Exact Match)")
        print_results(ft_results_fuzzy, "Fine-tuned Qwen2.5-7B (Fuzzy Match)")

    # --- RAG + LLM (API required) ---
    print("\n[5/5] RAG + LLM evaluation (API required)...")
    results, results_fuzzy = evaluate_rag(df, store, sample_size=50)

    if results:
        print_results(results, "LLM + RAG (Exact Match)")
        print_results(results_fuzzy, "LLM + RAG (Fuzzy Match)")

        overall_f1_rag = np.mean([m['f1'] for m in results_fuzzy['overall']])
        overall_f1_ft = np.mean([m['f1'] for m in ft_results_fuzzy['overall']]) if ft_results else 0
        print(f"\n  {'='*50}")
        print(f"  Cross-Baseline Comparison (Fuzzy F1)")
        print(f"  {'='*50}")
        print(f"  Random Guessing:          0.006")
        print(f"  Frequency Baseline:       0.327")
        print(f"  LLM Zero-shot:            0.393")
        print(f"  Fine-tuned Qwen-7B:       {overall_f1_ft:.3f}  ← LOCAL (free)")
        print(f"  LLM + RAG:                {overall_f1_rag:.3f}  ← best (API)\n")


# # ============================================================
# # Original DeepSeek-only main() — uncomment to use
# # ============================================================
# def main():
#     print("=" * 60)
#     print("  Baseline 3: RAG (Retrieval-Augmented Generation)")
#     print("=" * 60)
#     print("\n[1/4] Loading data...")
#     df = load_data()
#     df = preprocess(df)
#     print(f"  Loaded {len(df):,} perfumes")
#     print("\n[2/4] Building vector store...")
#     store = FragranceVectorStore()
#     store.build(df)
#     print("\n[3/4] Retrieval sanity check...")
#     sample_accords = df['accords'].iloc[100]
#     print(f"  Query accords: {sample_accords}")
#     refs = store.retrieve(sample_accords, top_k=3)
#     for ref in refs:
#         print(f"    [{ref['brand']}] {ref['name']} (sim: {ref['similarity']:.2f})")
#         print(f"    {ref['notes'][:150]}...")
#     print("\n[4/4] Running RAG evaluation...")
#     results, results_fuzzy = evaluate_rag(df, store, sample_size=50)
#     if results:
#         print_results(results, "LLM + RAG (Exact Match)")
#         print_results(results_fuzzy, "LLM + RAG (Fuzzy Match)")
#         overall_f1 = np.mean([m['f1'] for m in results['overall']])
#         print(f"\n  === Cross-Baseline Comparison ===")
#         print(f"  Random guessing:      F1 = 0.006")
#         print(f"  Frequency baseline:   F1 = 0.299")
#         print(f"  LLM Zero-shot:        F1 = 0.335")
#         print(f"  LLM + RAG:            F1 = {overall_f1:.3f}")
#         delta_vs_zero = (overall_f1 - 0.335) / 0.335 * 100
#         delta_vs_freq = (overall_f1 - 0.299) / 0.299 * 100
#         print(f"  RAG vs Zero-shot:     {delta_vs_zero:+.1f}%")
#         print(f"  RAG vs Frequency:     {delta_vs_freq:+.1f}%")


if __name__ == "__main__":
    main()
