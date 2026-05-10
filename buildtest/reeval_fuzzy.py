"""
Re-evaluate RAG baseline with fuzzy note matching.
Compares old F1 (exact match) vs new F1 (fuzzy: rules + synonyms + edit distance).

Run: export DEEPSEEK_API_KEY="sk-xxx" && python reeval_fuzzy.py
"""

import os, json, numpy as np
from openai import OpenAI
from utils import load_data, preprocess, evaluate_single, evaluate_single_fuzzy
from rag_baseline import FragranceVectorStore, format_references

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)

print("Loading data...")
df = load_data()
df = preprocess(df)
store = FragranceVectorStore()
store.build(df)

sample = df.sample(n=50, random_state=42)

results_old = {'top': [], 'mid': [], 'base': [], 'overall': []}
results_new = {'top': [], 'mid': [], 'base': [], 'overall': []}

print(f"Running RAG predictions on {len(sample)} samples...")
for i, (_, row) in enumerate(sample.iterrows()):
    if not row['accords']:
        continue

    # RAG retrieval
    refs = store.retrieve(row['accords'], top_k=5)
    refs_text = format_references(refs)

    prompt = f"""You are a professional perfumer. Given the main accords, predict Top/Middle/Base notes.
Main Accords: {', '.join(row['accords'])}
Reference perfumes:
{refs_text}
Reply ONLY with JSON: {{"top_notes": [...5 notes...], "middle_notes": [...5 notes...], "base_notes": [...5 notes...]}}
Use lowercase standardized note names."""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
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
    pred_top = [n.strip().lower() for n in pred.get("top_notes", [])]
    pred_mid = [n.strip().lower() for n in pred.get("middle_notes", [])]
    pred_base = [n.strip().lower() for n in pred.get("base_notes", [])]

    # OLD evaluation
    p_t, r_t, f_t = evaluate_single(pred_top, row['Top_clean'])
    p_m, r_m, f_m = evaluate_single(pred_mid, row['Middle_clean'])
    p_b, r_b, f_b = evaluate_single(pred_base, row['Base_clean'])
    results_old['top'].append({'precision': p_t, 'recall': r_t, 'f1': f_t})
    results_old['mid'].append({'precision': p_m, 'recall': r_m, 'f1': f_m})
    results_old['base'].append({'precision': p_b, 'recall': r_b, 'f1': f_b})
    results_old['overall'].append({'precision': 0, 'recall': 0, 'f1': 0})
    p_all = set(pred_top + pred_mid + pred_base)
    t_all = row['Top_clean'] | row['Middle_clean'] | row['Base_clean']
    _, _, f_a = evaluate_single(p_all, t_all)
    results_old['overall'][-1] = {'precision': 0, 'recall': 0, 'f1': f_a}

    # NEW evaluation (fuzzy)
    p_tf, r_tf, f_tf = evaluate_single_fuzzy(pred_top, row['Top_clean'])
    p_mf, r_mf, f_mf = evaluate_single_fuzzy(pred_mid, row['Middle_clean'])
    p_bf, r_bf, f_bf = evaluate_single_fuzzy(pred_base, row['Base_clean'])
    results_new['top'].append({'precision': p_tf, 'recall': r_tf, 'f1': f_tf})
    results_new['mid'].append({'precision': p_mf, 'recall': r_mf, 'f1': f_mf})
    results_new['base'].append({'precision': p_bf, 'recall': r_bf, 'f1': f_bf})
    results_new['overall'].append({'precision': 0, 'recall': 0, 'f1': 0})
    p_all_f = set(pred_top + pred_mid + pred_base)
    _, _, f_af = evaluate_single_fuzzy(p_all_f, t_all)
    results_new['overall'][-1] = {'precision': 0, 'recall': 0, 'f1': f_af}

    if (i + 1) % 10 == 0:
        f1_old = np.mean([m['f1'] for m in results_old['overall']])
        f1_new = np.mean([m['f1'] for m in results_new['overall']])
        print(f"  [{i+1}/50] Old F1={f1_old:.3f}  New F1={f1_new:.3f}  Delta={f1_new-f1_old:+.3f}")


print(f"\n{'='*60}")
print(f"  Comparison: Exact Match vs Fuzzy Matching")
print(f"{'='*60}")
for layer in ['top', 'mid', 'base', 'overall']:
    old_f1 = np.mean([m['f1'] for m in results_old[layer]])
    new_f1 = np.mean([m['f1'] for m in results_new[layer]])
    old_p = np.mean([m['precision'] for m in results_old[layer]])
    new_p = np.mean([m['precision'] for m in results_new[layer]])
    old_r = np.mean([m['recall'] for m in results_old[layer]])
    new_r = np.mean([m['recall'] for m in results_new[layer]])
    delta = new_f1 - old_f1
    print(f"  {layer:<10} F1: {old_f1:.3f} → {new_f1:.3f} ({delta:+.3f})  "
          f"P: {old_p:.3f}→{new_p:.3f}  R: {old_r:.3f}→{new_r:.3f}")

overall_old = np.mean([m['f1'] for m in results_old['overall']])
overall_new = np.mean([m['f1'] for m in results_new['overall']])
pct = (overall_new - overall_old) / overall_old * 100
print(f"\n  Overall F1 improvement: {overall_old:.3f} → {overall_new:.3f} (+{pct:.1f}%)")
