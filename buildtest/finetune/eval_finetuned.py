"""
Evaluate the fine-tuned model on the test set.
Computes both exact and fuzzy F1, compares against baselines.

Usage after training:
  python finetune/eval_finetuned.py
"""
import json
import sys
import numpy as np
sys.path.insert(0, '.')

from src.data import load_data, preprocess
from src.evaluation import evaluate_single, evaluate_single_fuzzy, print_results
from finetune.inference import load_model, predict, parse_notes


def evaluate(model, tokenizer, df, sample_size=100):
    """Evaluate fine-tuned model on a test sample."""
    sample = df.sample(n=sample_size, random_state=42)
    results_exact = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}

    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']:
            continue

        accords_text = ', '.join(row['accords'])
        output = predict(model, tokenizer, accords_text)
        pred_top, pred_mid, pred_base = parse_notes(output)

        # Exact match
        p_t, r_t, f_t = evaluate_single(pred_top, row['Top_clean'])
        p_m, r_m, f_m = evaluate_single(pred_mid, row['Middle_clean'])
        p_b, r_b, f_b = evaluate_single(pred_base, row['Base_clean'])
        results_exact['top'].append({'precision': p_t, 'recall': r_t, 'f1': f_t})
        results_exact['mid'].append({'precision': p_m, 'recall': r_m, 'f1': f_m})
        results_exact['base'].append({'precision': p_b, 'recall': r_b, 'f1': f_b})
        pred_all = pred_top | pred_mid | pred_base
        true_all = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
        _, _, f_a = evaluate_single(pred_all, true_all)
        results_exact['overall'].append({'precision': 0, 'recall': 0, 'f1': f_a})

        # Fuzzy match
        p_tf, r_tf, f_tf = evaluate_single_fuzzy(pred_top, row['Top_clean'])
        p_mf, r_mf, f_mf = evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
        p_bf, r_bf, f_bf = evaluate_single_fuzzy(pred_base, row['Base_clean'])
        results_fuzzy['top'].append({'precision': p_tf, 'recall': r_tf, 'f1': f_tf})
        results_fuzzy['mid'].append({'precision': p_mf, 'recall': r_mf, 'f1': f_mf})
        results_fuzzy['base'].append({'precision': p_bf, 'recall': r_bf, 'f1': f_bf})
        _, _, f_af = evaluate_single_fuzzy(pred_all, true_all)
        results_fuzzy['overall'].append({'precision': 0, 'recall': 0, 'f1': f_af})

        if (i + 1) % 20 == 0:
            f1_exact = np.mean([m['f1'] for m in results_exact['overall']])
            f1_fuzzy = np.mean([m['f1'] for m in results_fuzzy['overall']])
            print(f"  [{i+1}/{sample_size}] Exact F1={f1_exact:.3f}  Fuzzy F1={f1_fuzzy:.3f}")

    return results_exact, results_fuzzy


def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading test data...")
    df = load_data()
    df = preprocess(df)
    df = df[df['Top_clean'].apply(len) > 0]
    df = df[df['Middle_clean'].apply(len) > 0]
    df = df[df['Base_clean'].apply(len) > 0]

    print(f"Evaluating on 100 samples...")
    results_exact, results_fuzzy = evaluate(model, tokenizer, df, sample_size=100)

    print_results(results_exact, "Fine-tuned Qwen2.5-7B (Exact Match)")
    print_results(results_fuzzy, "Fine-tuned Qwen2.5-7B (Fuzzy Match)")

    overall_exact = np.mean([m['f1'] for m in results_exact['overall']])
    overall_fuzzy = np.mean([m['f1'] for m in results_fuzzy['overall']])

    print(f"\n  {'='*50}")
    print(f"  Comparison with Baselines")
    print(f"  {'='*50}")
    print(f"  Random Guessing:     0.006 / 0.006")
    print(f"  Frequency Baseline:  0.299 / 0.327")
    print(f"  LLM Zero-shot:       0.337 / 0.393")
    print(f"  LLM + RAG:           0.406 / 0.443  ← previous best")
    print(f"  Fine-tuned Qwen-7B:  {overall_exact:.3f} / {overall_fuzzy:.3f}  ← NEW")

    # Save results
    output = {
        "model": "Qwen2.5-7B-Instruct-LoRA",
        "exact_match": {
            "overall_f1": overall_exact,
            "top": float(np.mean([m['f1'] for m in results_exact['top']])),
            "mid": float(np.mean([m['f1'] for m in results_exact['mid']])),
            "base": float(np.mean([m['f1'] for m in results_exact['base']])),
        },
        "fuzzy_match": {
            "overall_f1": overall_fuzzy,
            "top": float(np.mean([m['f1'] for m in results_fuzzy['top']])),
            "mid": float(np.mean([m['f1'] for m in results_fuzzy['mid']])),
            "base": float(np.mean([m['f1'] for m in results_fuzzy['base']])),
        },
    }
    with open("finetune/eval_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to finetune/eval_results.json")


if __name__ == "__main__":
    main()
