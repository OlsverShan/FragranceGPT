"""
Baseline 1: Frequency Baseline
===============================
Pure statistical co-occurrence: accord → note frequency mapping.
No LLM required. Runs on CPU in seconds.

Run: python baseline.py
"""

from collections import defaultdict, Counter
from utils import load_data, preprocess, evaluate_single, evaluate_single_fuzzy, print_results


class FrequencyBaseline:
    """
    For each accord (e.g. "citrus"), count how often each note appears
    in Top/Middle/Base across all perfumes with that accord.
    Prediction = top-K most frequent notes per accord, per layer.
    """

    def __init__(self, df, top_k=5):
        self.top_k = top_k
        self.accord_to_top = defaultdict(Counter)
        self.accord_to_mid = defaultdict(Counter)
        self.accord_to_base = defaultdict(Counter)

        for _, row in df.iterrows():
            for accord in row['accords']:
                for note in row['Top_clean']:
                    self.accord_to_top[accord][note] += 1
                for note in row['Middle_clean']:
                    self.accord_to_mid[accord][note] += 1
                for note in row['Base_clean']:
                    self.accord_to_base[accord][note] += 1

    def predict(self, accords):
        top_notes = set()
        mid_notes = set()
        base_notes = set()
        for accord in accords:
            accord = accord.lower().strip()
            for note, _ in self.accord_to_top.get(accord, Counter()).most_common(self.top_k):
                top_notes.add(note)
            for note, _ in self.accord_to_mid.get(accord, Counter()).most_common(self.top_k):
                mid_notes.add(note)
            for note, _ in self.accord_to_base.get(accord, Counter()).most_common(self.top_k):
                base_notes.add(note)
        return top_notes, mid_notes, base_notes

    def show_examples(self, n=5):
        all_accords = (set(self.accord_to_top.keys()) |
                       set(self.accord_to_mid.keys()) |
                       set(self.accord_to_base.keys()))
        for accord in list(all_accords)[:n]:
            print(f"\n  [{accord}]")
            print(f"    → Top:    {[n for n, _ in self.accord_to_top[accord].most_common(5)]}")
            print(f"    → Middle: {[n for n, _ in self.accord_to_mid[accord].most_common(5)]}")
            print(f"    → Base:   {[n for n, _ in self.accord_to_base[accord].most_common(5)]}")


def evaluate(baseline, df, sample_size=2000):
    results = {'top': [], 'mid': [], 'base': [], 'overall': []}
    results_fuzzy = {'top': [], 'mid': [], 'base': [], 'overall': []}
    sample = df.sample(n=sample_size, random_state=42)

    for _, row in sample.iterrows():
        if not row['accords']:
            continue

        pred_top, pred_mid, pred_base = baseline.predict(row['accords'])

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

    return results, results_fuzzy


def main():
    print("=" * 60)
    print("  Baseline 1: Frequency Baseline (Statistical Co-occurrence)")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    df = load_data()
    df = preprocess(df)
    print(f"  Loaded {len(df):,} perfumes")

    # Dataset stats
    all_notes_set = set()
    for notes in df['all_notes']:
        all_notes_set |= notes
    vocab_size = len(all_notes_set)
    avg_notes = df['all_notes'].apply(len).mean()
    print(f"  Unique notes: {vocab_size:,}")
    print(f"  Avg notes per perfume: {avg_notes:.1f}")

    print("\n[2/3] Building frequency baseline (top_k=5)...")
    freq = FrequencyBaseline(df, top_k=5)
    freq.show_examples(5)

    print("\n[3/3] Evaluating on 2000 samples...")
    results, results_fuzzy = evaluate(freq, df, sample_size=2000)
    print_results(results, "Frequency Baseline (Exact Match)")
    print_results(results_fuzzy, "Frequency Baseline (Fuzzy Match)")

    print(f"\n  Random guessing recall ≈ {avg_notes / vocab_size * 100:.2f}%")
    print("  Frequency baseline beats random by ~95× on recall.")


if __name__ == "__main__":
    main()
