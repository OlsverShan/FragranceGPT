"""
Baseline 4: RAG + Few-shot Baseline
=====================================
RAG retrieval (top-5 similar perfumes) + curated few-shot examples →
LLM predicts Top/Middle/Base notes.

[LOCAL MODEL] Fine-tuned Qwen2.5-7B comparison included

Run:
  export DEEPSEEK_API_KEY="sk-xxx"
  python fewshot_baseline.py
"""

import os
import json
import numpy as np
from utils import load_data, preprocess, evaluate_single, evaluate_single_fuzzy, print_results
from rag_baseline import FragranceVectorStore, format_references, get_client

# ============================================================
# Local Fine-tuned Model
# ============================================================
_LOCAL_MODEL = None; _LOCAL_TOKENIZER = None
def _get_local_model():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is None:
        from finetune.inference import load_model
        _LOCAL_MODEL, _LOCAL_TOKENIZER = load_model()
    return _LOCAL_MODEL, _LOCAL_TOKENIZER
def predict_local(accords):
    from finetune.inference import predict as ft_predict, parse_notes
    model, tokenizer = _get_local_model()
    return parse_notes(ft_predict(model, tokenizer, ', '.join(accords)))
def evaluate_finetuned(df, sample_size=100):
    print(f"  Model: Qwen2.5-7B-LoRA (local)  Samples: {sample_size}")
    sample = df.sample(n=sample_size, random_state=42)
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}
    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']: continue
        try: ptop, pmid, pbase = predict_local(row['accords'])
        except: ptop, pmid, pbase = set(), set(), set()
        pt,rt,ft = evaluate_single(ptop, row['Top_clean']); pm,rm,fm = evaluate_single(pmid, row['Middle_clean']); pb,rb,fb = evaluate_single(pbase, row['Base_clean'])
        results['top'].append({'precision': pt, 'recall': rt, 'f1': ft})
        results['mid'].append({'precision': pm, 'recall': rm, 'f1': fm})
        results['base'].append({'precision': pb, 'recall': rb, 'f1': fb})
        _,_,fa = evaluate_single(ptop|pmid|pbase, row['Top_clean']|row['Middle_clean']|row['Base_clean'])
        results['overall'].append({'precision': 0, 'recall': 0, 'f1': fa})
        ptf,rtf,ftf = evaluate_single_fuzzy(ptop, row['Top_clean']); pmf,rmf,fmf = evaluate_single_fuzzy(pmid, row['Middle_clean']); pbf,rbf,fbf = evaluate_single_fuzzy(pbase, row['Base_clean'])
        results_fuzzy['top'].append({'precision': ptf, 'recall': rtf, 'f1': ftf})
        results_fuzzy['mid'].append({'precision': pmf, 'recall': rmf, 'f1': fmf})
        results_fuzzy['base'].append({'precision': pbf, 'recall': rbf, 'f1': fbf})
        _,_,faf = evaluate_single_fuzzy(ptop|pmid|pbase, row['Top_clean']|row['Middle_clean']|row['Base_clean'])
        results_fuzzy['overall'].append({'precision': 0, 'recall': 0, 'f1': faf})
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{sample_size}] Exact F1={np.mean([m['f1'] for m in results['overall']]):.3f}  Fuzzy F1={np.mean([m['f1'] for m in results_fuzzy['overall']]):.3f}")
    return results, results_fuzzy

# ============================================================
# Curated Few-shot Examples
# ============================================================
# Selected from the 24k dataset: high rating count (>10k votes),
# diverse accord coverage, representative note mappings.
# Each example MUST have 5 notes per layer (pad if needed) to
# establish a consistent output format for the LLM.

FEWSHOT_EXAMPLES = [
    {
        "accords": "citrus, woody, fresh, fruity, aromatic",
        "top_notes": ["sicilian lemon", "apple", "cedar", "bellflower", "green notes"],
        "middle_notes": ["jasmine", "white rose", "bamboo", "lily-of-the-valley", "freesia"],
        "base_notes": ["amber", "musk", "cedar", "sandalwood", "white musk"],
    },
    {
        "accords": "aromatic, warm spicy, lavender, woody, fresh spicy",
        "top_notes": ["cardamom", "bergamot", "lavender", "pink pepper", "lemon"],
        "middle_notes": ["lavender", "virginia cedar", "bergamot", "geranium", "clary sage"],
        "base_notes": ["vetiver", "caraway", "tonka bean", "patchouli", "cedar"],
    },
    {
        "accords": "musky, powdery, white floral, citrus, floral",
        "top_notes": ["bergamot", "african orange flower", "osmanthus", "neroli", "mandarin orange"],
        "middle_notes": ["musk", "amber", "jasmine", "rose", "orange blossom"],
        "base_notes": ["patchouli", "vetiver", "vanilla", "sandalwood", "white musk"],
    },
    {
        "accords": "floral, citrus, fresh, woody, fresh spicy",
        "top_notes": ["yuzu", "pomegranate", "bergamot", "lemon", "ice"],
        "middle_notes": ["peony", "magnolia", "lotus", "lily-of-the-valley", "jasmine"],
        "base_notes": ["musk", "amber", "mahogany", "sandalwood", "cedar"],
    },
    {
        "accords": "woody, floral, sweet, powdery, amber",
        "top_notes": ["pomegranate", "persimmon", "green notes", "bergamot", "pink pepper"],
        "middle_notes": ["black orchid", "lotus", "champaca", "jasmine", "rose"],
        "base_notes": ["amber", "mahogany", "black violet", "musk", "vanilla"],
    },
    {
        "accords": "sweet, patchouli, fruity, warm spicy, caramel",
        "top_notes": ["bergamot", "mandarin orange", "coconut", "melon", "pineapple"],
        "middle_notes": ["honey", "apricot", "peach", "red berries", "rose"],
        "base_notes": ["patchouli", "vanilla", "caramel", "chocolate", "tonka bean"],
    },
]

# ============================================================
# Prompt Templates
# ============================================================

def build_fewshot_prompt(accords, references, n_examples=5):
    """Build prompt with few-shot examples + RAG references."""

    # Few-shot section
    fewshot_lines = []
    for i, ex in enumerate(FEWSHOT_EXAMPLES[:n_examples], 1):
        fewshot_lines.append(f"Example {i}:")
        fewshot_lines.append(f"  Accords: {ex['accords']}")
        fewshot_lines.append(f"  Top notes:    {json.dumps(ex['top_notes'])}")
        fewshot_lines.append(f"  Middle notes: {json.dumps(ex['middle_notes'])}")
        fewshot_lines.append(f"  Base notes:   {json.dumps(ex['base_notes'])}")
        fewshot_lines.append("")
    fewshot_text = "\n".join(fewshot_lines)

    # RAG references section
    refs_text = format_references(references)

    prompt = f"""You are a professional perfumer. Given the main accords of a fragrance, predict the most likely Top, Middle, and Base notes.

Here are {n_examples} examples of correct accord-to-notes mappings from real perfumes:

{fewshot_text}
Now predict for a new fragrance. Use the reference perfumes below (similar accords) as additional hints:

Main Accords: {', '.join(accords)}

Reference perfumes:
{refs_text}

Reply ONLY with a JSON object (no markdown, no explanation):
{{"top_notes": ["note1", "note2", "note3", "note4", "note5"],
  "middle_notes": ["note1", "note2", "note3", "note4", "note5"],
  "base_notes":   ["note1", "note2", "note3", "note4", "note5"]}}

Include exactly 5 specific notes per layer. Use lowercase standardized note names."""

    return prompt


def predict_fewshot(client, model, accords, references, n_examples=5):
    """LLM call with RAG + Few-shot prompt."""
    prompt = build_fewshot_prompt(accords, references, n_examples)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
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

def evaluate_fewshot(df, vector_store, sample_size=50, n_examples=5):
    client, model, provider = get_client()
    if client is None:
        print("  No API key found.")
        return None

    print(f"  Provider: {provider}  Model: {model}")
    print(f"  Few-shot examples: {n_examples}  RAG top-K: 5  Samples: {sample_size}")

    sample = df.sample(n=sample_size, random_state=42)
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}

    for i, (_, row) in enumerate(sample.iterrows()):
        if not row['accords']:
            continue

        references = vector_store.retrieve(row['accords'], top_k=5)

        try:
            pred_top, pred_mid, pred_base = predict_fewshot(
                client, model, row['accords'], references, n_examples
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
    print("  Baseline 4: RAG + Few-shot")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_data()
    df = preprocess(df)
    print(f"  Loaded {len(df):,} perfumes")

    print("\n[2/4] Loading vector store...")
    store = FragranceVectorStore()
    store.build(df)  # skips if already built

    print("\n[3/4] Few-shot examples preview...")
    for i, ex in enumerate(FEWSHOT_EXAMPLES[:3], 1):
        print(f"  Ex {i}: accords=[{ex['accords'][:60]}...] "
              f"→ top={ex['top_notes'][:2]}... base={ex['base_notes'][:2]}...")

    print("\n[4/4] Running RAG + Few-shot evaluation...")
    results, results_fuzzy = evaluate_fewshot(df, store, sample_size=50, n_examples=5)

    if results:
        print_results(results, "RAG + Few-shot (Exact Match)")
        print_results(results_fuzzy, "RAG + Few-shot (Fuzzy Match)")

        overall_f1 = np.mean([m['f1'] for m in results['overall']])
        print(f"\n  {'='*50}")
        print(f"  Cross-Baseline Comparison")
        print(f"  {'='*50}")
        baselines = [
            ("Random guessing",      0.006),
            ("Frequency baseline",   0.299),
            ("LLM Zero-shot",        0.335),
            ("LLM + RAG",            0.402),
            ("LLM + RAG + Few-shot", overall_f1),
        ]
        for name, f1 in baselines:
            delta = ""
            if len(baselines) > 1 and f1 > 0.006:
                prev = baselines[baselines.index((name, f1)) - 1][1]
                if prev > 0.01:
                    pct = (f1 - prev) / prev * 100
                    delta = f"  (Δ {pct:+.1f}%)"
            print(f"  {name:<25} F1 = {f1:.3f}{delta}")


if __name__ == "__main__":
    main()
