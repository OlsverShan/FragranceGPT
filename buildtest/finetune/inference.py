"""
Run inference with a fine-tuned fragrance model.
Loads the QLoRA adapter and generates note predictions.

Usage:
  python finetune/inference.py --accords "citrus, woody, aromatic, fresh spicy, amber"
"""
import argparse
import glob
import torch
from unsloth import FastLanguageModel


# Auto-detect: try 7B first, fall back to 3B
import glob
_candidates = glob.glob("./finetune/fragrance-qwen-*b-final")
MODEL_PATH = _candidates[0] if _candidates else "./finetune/fragrance-qwen-7b-final"
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_model(model_path=MODEL_PATH):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict(model, tokenizer, accords, instruction=None):
    """Generate notes prediction for given accords."""
    if instruction is None:
        instruction = "You are a professional perfumer. Given the main accords of a fragrance, predict the Top, Middle, and Base notes. List exactly 5 notes per layer, using standardized lowercase names."

    prompt = PROMPT_TEMPLATE.format(instruction=instruction, input=accords)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def parse_notes(text):
    """Parse model output into Top/Mid/Base note sets."""
    top, mid, base = set(), set(), set()
    for line in text.split('\n'):
        line = line.strip().lower()
        if line.startswith('top:'):
            top = {n.strip() for n in line[4:].split(',') if n.strip() and n.strip() != '(none)'}
        elif line.startswith('middle:'):
            mid = {n.strip() for n in line[7:].split(',') if n.strip() and n.strip() != '(none)'}
        elif line.startswith('base:'):
            base = {n.strip() for n in line[5:].split(',') if n.strip() and n.strip() != '(none)'}
    return top, mid, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accords', type=str, required=True,
                        help='Comma-separated accords, e.g. "citrus, woody, aromatic"')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to fine-tuned model')
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    print(f"\nAccords: {args.accords}")
    print("-" * 50)
    result = predict(model, tokenizer, args.accords)
    print(result)

    top, mid, base = parse_notes(result)
    print(f"\nParsed:")
    print(f"  Top:    {sorted(top)}")
    print(f"  Middle: {sorted(mid)}")
    print(f"  Base:   {sorted(base)}")


if __name__ == "__main__":
    main()
