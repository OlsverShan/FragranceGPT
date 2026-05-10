"""
Convert 24k fragrance dataset into instruction fine-tuning format.
Output: finetune/train.jsonl and finetune/eval.jsonl

Run: python finetune/prepare_data.py
"""
import json
import sys
sys.path.insert(0, '.')
from src.data import load_data, preprocess


def format_sample(row):
    """Convert one perfume row to instruction format."""
    accords = ', '.join(row['accords'])

    top = ', '.join(sorted(row['Top_clean'])) if row['Top_clean'] else '(none)'
    mid = ', '.join(sorted(row['Middle_clean'])) if row['Middle_clean'] else '(none)'
    base = ', '.join(sorted(row['Base_clean'])) if row['Base_clean'] else '(none)'

    output = f"Top: {top}\nMiddle: {mid}\nBase: {base}"

    return {
        "instruction": "You are a professional perfumer. Given the main accords of a fragrance, predict the Top, Middle, and Base notes. List exactly 5 notes per layer, using standardized lowercase names.",
        "input": accords,
        "output": output,
    }


def main():
    print("Loading data...")
    df = load_data()
    df = preprocess(df)

    # Filter: only use perfumes with notes in all 3 layers
    df = df[df['Top_clean'].apply(len) > 0]
    df = df[df['Middle_clean'].apply(len) > 0]
    df = df[df['Base_clean'].apply(len) > 0]
    print(f"  Filtered to {len(df)} perfumes with complete notes")

    # Shuffle and split: 95% train, 5% eval
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(len(df) * 0.95)

    train_df = df.iloc[:split]
    eval_df = df.iloc[split:]

    for name, subset in [("train.jsonl", train_df), ("eval.jsonl", eval_df)]:
        path = f"finetune/{name}"
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in subset.iterrows():
                sample = format_sample(row)
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"  Saved {len(subset)} samples to {path}")

    # Show a few examples
    print("\nExample training samples:")
    for i in range(3):
        s = format_sample(train_df.iloc[i])
        print(f"\n  --- Sample {i+1} ---")
        print(f"  Input:  {s['input']}")
        print(f"  Output: {s['output'][:120]}...")


if __name__ == "__main__":
    main()
