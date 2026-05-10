"""
LLM-as-Judge Evaluation Framework
===================================
Uses an LLM judge to score fragrance predictions on 4 dimensions:
  1. Accord Accuracy  — Do predicted notes match the given accords?
  2. Layer Correctness — Are notes in the right layer (top/mid/base)?
  3. Coherence        — Do the three layers form a logical progression?
  4. Professionalism   — Would a real perfumer find this plausible?

This is a different evaluation paradigm from token-level F1:
  - F1 measures exact string match with ground truth
  - LLM-as-Judge measures QUALITY of the prediction itself

A good prediction can have low F1 (creative but valid notes) or
high F1 but low Coherence (correct notes in wrong layers).

Run:
  export DEEPSEEK_API_KEY="sk-xxx"
  python judge_baseline.py
"""

import os
import json
import numpy as np
from openai import OpenAI

from utils import load_data, preprocess, evaluate_single
from rag_baseline import FragranceVectorStore, format_references, get_client


# ============================================================
# Judge Prompt
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are a professional perfumer with 20 years of experience, serving as a judge in a fragrance evaluation competition. You will receive a fragrance brief (main accords) and a set of predicted notes (Top/Middle/Base). Score the prediction on each dimension from 1 (terrible) to 5 (perfect).

Scoring Rubric:

1. Accord Accuracy (1-5):
   How well do the predicted notes align with the main accords?
   5 = Every note strongly supports the accords
   3 = Most notes fit, some are off-target
   1 = Notes have little to do with the accords

2. Layer Correctness (1-5):
   Are notes placed in the correct volatility layer?
   Top notes should be: citrus, light fruits, green herbs, aldehydes (high volatility)
   Middle notes should be: florals, spices, medium fruits (medium volatility)
   Base notes should be: woods, resins, musks, vanilla, patchouli (low volatility)
   5 = All 15 notes in correct layers
   3 = 2-3 notes misplaced
   1 = Many notes in wrong layers

3. Coherence (1-5):
   Do the three layers form a logical, harmonious fragrance progression?
   5 = Beautiful arc from top through heart to base
   3 = Layers work but feel disconnected
   1 = Notes clash across layers, no coherent story

4. Professionalism (1-5):
   Would a working perfumer consider this a viable formula?
   5 = Could be a commercial fragrance formula
   3 = Reasonable attempt, some odd choices
   1 = Amateur or nonsensical combination

Output ONLY a JSON object:
{
  "accord_accuracy": <1-5>,
  "layer_correctness": <1-5>,
  "coherence": <1-5>,
  "professionalism": <1-5>,
  "overall": <1-5>,
  "brief_comment": "<1 sentence explaining the score>"
}"""


def build_judge_prompt(accords, pred_top, pred_mid, pred_base):
    """Build the judge evaluation prompt."""
    return f"""Fragrance Brief:
  Main Accords: {', '.join(accords)}

Predicted Formula:
  Top notes:    {json.dumps(sorted(pred_top) if pred_top else [])}
  Middle notes: {json.dumps(sorted(pred_mid) if pred_mid else [])}
  Base notes:   {json.dumps(sorted(pred_base) if pred_base else [])}

Please score this prediction."""


def judge_single(client, model, accords, pred_top, pred_mid, pred_base):
    """Run LLM judge on a single prediction."""
    prompt = build_judge_prompt(accords, pred_top, pred_mid, pred_base)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "overall": 0}


def evaluate_with_judge(df, vector_store, sample_size=30):
    """
    Full pipeline: predict with RAG → judge predictions with LLM.
    Also judges the GROUND TRUTH as a reference baseline.
    """
    client, model, provider = get_client()
    if client is None:
        print("  No API key found.")
        return None

    print(f"  Judge model: {provider}/{model}")
    print(f"  Samples: {sample_size} (each judged twice: prediction + ground truth)")

    sample = df.sample(n=sample_size, random_state=42)

    judge_scores_pred = []     # LLM judge scores for predictions
    judge_scores_truth = []    # LLM judge scores for ground truth
    f1_scores = []             # Token-level F1 (for correlation analysis)

    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']:
            continue

        accords = row['accords']
        true_top = row['Top_clean']
        true_mid = row['Middle_clean']
        true_base = row['Base_clean']

        # --- RAG prediction ---
        references = vector_store.retrieve(accords, top_k=5)
        refs_text = format_references(references)

        pred_prompt = f"""You are a professional perfumer. Given the main accords, predict Top/Middle/Base notes.
Main Accords: {', '.join(accords)}
Reference perfumes:
{refs_text}
Reply ONLY with JSON: {{"top_notes": [...5 notes...], "middle_notes": [...5 notes...], "base_notes": [...5 notes...]}}
Use lowercase standardized note names."""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": pred_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3, max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:])
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            pred = json.loads(raw)
            pred_top = set(n.strip().lower() for n in pred.get("top_notes", []))
            pred_mid = set(n.strip().lower() for n in pred.get("middle_notes", []))
            pred_base = set(n.strip().lower() for n in pred.get("base_notes", []))
        except Exception as e:
            print(f"  [{i+1}/{sample_size}] Predict error: {e}")
            continue

        # --- LLM Judge: score prediction ---
        judge_pred = judge_single(client, model, accords, pred_top, pred_mid, pred_base)

        # --- LLM Judge: score ground truth ---
        judge_truth = judge_single(client, model, accords, true_top, true_mid, true_base)

        # --- Token F1 ---
        pred_all = pred_top | pred_mid | pred_base
        true_all = true_top | true_mid | true_base
        _, _, f1 = evaluate_single(pred_all, true_all)

        judge_scores_pred.append(judge_pred)
        judge_scores_truth.append(judge_truth)
        f1_scores.append(f1)

        if (i + 1) % 10 == 0:
            avg_pred = np.mean([s.get("overall", 0) for s in judge_scores_pred if "overall" in s])
            avg_truth = np.mean([s.get("overall", 0) for s in judge_scores_truth if "overall" in s])
            avg_f1 = np.mean(f1_scores)
            print(f"  [{i+1}/{sample_size}] Judge(Pred)={avg_pred:.1f}  "
                  f"Judge(Truth)={avg_truth:.1f}  F1={avg_f1:.3f}")

    return {
        "judge_pred": judge_scores_pred,
        "judge_truth": judge_scores_truth,
        "f1_scores": f1_scores,
        "sample_size": len(f1_scores),
    }


# ============================================================
# Analysis
# ============================================================

def analyze_results(results):
    """Compute statistics and correlations."""
    pred_scores = results["judge_pred"]
    truth_scores = results["judge_truth"]
    f1_scores = results["f1_scores"]

    print(f"\n{'='*60}")
    print(f"  LLM-as-Judge Results (n={results['sample_size']})")
    print(f"{'='*60}")

    # Average scores per dimension
    dims = ["accord_accuracy", "layer_correctness", "coherence", "professionalism", "overall"]
    print(f"\n  {'Dimension':<22} {'Prediction':>10} {'Ground Truth':>12} {'Gap':>8}")
    print(f"  {'-'*52}")
    for dim in dims:
        pred_vals = [s.get(dim, 0) for s in pred_scores if dim in s]
        truth_vals = [s.get(dim, 0) for s in truth_scores if dim in s]
        if pred_vals and truth_vals:
            avg_p = np.mean(pred_vals)
            avg_t = np.mean(truth_vals)
            gap = avg_t - avg_p
            print(f"  {dim:<22} {avg_p:>8.2f}/5 {avg_t:>10.2f}/5 {gap:>+7.2f}")

    # Overall comparison
    pred_overall = [s.get("overall", 0) for s in pred_scores if "overall" in s]
    truth_overall = [s.get("overall", 0) for s in truth_scores if "overall" in s]
    avg_pred = np.mean(pred_overall)
    avg_truth = np.mean(truth_overall)
    print(f"\n  Judge Overall: Prediction = {avg_pred:.2f}/5, Ground Truth = {avg_truth:.2f}/5")
    print(f"  Gap (Truth - Pred): {avg_truth - avg_pred:+.2f}")

    # Correlation: does F1 predict judge score?
    if len(f1_scores) > 5 and len(pred_overall) == len(f1_scores):
        corr = np.corrcoef(f1_scores, pred_overall)[0, 1]
        print(f"\n  Correlation: F1 vs Judge Score = {corr:.3f}")
        if corr > 0.7:
            print("  → F1 and LLM Judge strongly agree (both measure 'correctness')")
        elif corr > 0.4:
            print("  → Moderate agreement: F1 and Judge capture different but related aspects")
        else:
            print("  → Weak correlation: Judge captures QUALITY beyond exact match")

    # Sample comments
    print(f"\n  Sample Judge Comments (on predictions):")
    for i, s in enumerate(pred_scores[:5]):
        comment = s.get("brief_comment", "N/A")
        overall = s.get("overall", "?")
        f1 = f1_scores[i] if i < len(f1_scores) else 0
        print(f"    [{overall}/5, F1={f1:.2f}] {comment}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  LLM-as-Judge Evaluation")
    print("=" * 60)

    print("\n[1/3] Loading data + vector store...")
    df = load_data()
    df = preprocess(df)
    store = FragranceVectorStore()
    store.build(df)

    print("\n[2/3] Evaluating: RAG prediction → LLM Judge scoring...")
    print("  (Each sample: 1 predict call + 2 judge calls = 3 API calls)")
    results = evaluate_with_judge(df, store, sample_size=30)

    if results:
        print("\n[3/3] Analysis...")
        analyze_results(results)

        # Save raw results
        with open("judge_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\n  Raw results saved to judge_results.json")


if __name__ == "__main__":
    main()
