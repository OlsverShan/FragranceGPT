"""
Fine-tune Qwen2.5-7B-Instruct on fragrance accord→notes prediction.
Uses Unsloth + QLoRA 4-bit — runs on free Colab T4 (16GB VRAM).

Usage (Colab / cloud GPU):
  1. pip install unsloth
  2. python finetune/train.py

Or open finetune/train_colab.ipynb in Colab.
"""
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments


# ============================================================
# Config — auto-adapts to available VRAM
# ============================================================

def get_model_config():
    """Choose model and batch settings based on available VRAM."""
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if vram_gb >= 20:
        print(f"  VRAM: {vram_gb:.1f}GB → 7B model, batch=8 (high-end)")
        return {
            "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "max_seq_length": 1024,
            "batch_size": 8, "grad_accum": 2,
            "save_name": "fragrance-qwen-7b",
        }
    elif vram_gb >= 14:
        print(f"  VRAM: {vram_gb:.1f}GB → 7B model, batch=4")
        return {
            "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "max_seq_length": 1024,
            "batch_size": 4, "grad_accum": 4,
            "save_name": "fragrance-qwen-7b",
        }
    elif vram_gb >= 7:
        print(f"  VRAM: {vram_gb:.1f}GB → 7B model, batch=1 (tight fit)")
        return {
            "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "max_seq_length": 512,
            "batch_size": 1, "grad_accum": 8,
            "save_name": "fragrance-qwen-7b",
        }
    else:
        print(f"  VRAM: {vram_gb:.1f}GB → 3B model, batch=2 (safe)")
        return {
            "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "max_seq_length": 512,
            "batch_size": 2, "grad_accum": 4,
            "save_name": "fragrance-qwen-3b",
        }

CFG = get_model_config()

MODEL_NAME = CFG["model_name"]
MAX_SEQ_LENGTH = CFG["max_seq_length"]
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = CFG["batch_size"]
GRAD_ACCUM = CFG["grad_accum"]
OUTPUT_DIR = f"./finetune/{CFG['save_name']}-lora"
SAVE_DIR = f"./finetune/{CFG['save_name']}-final"


# ============================================================
# Prompt Template
# ============================================================

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

EOS_TOKEN = "<|im_end|>"


def formatting_func(examples):
    """Convert dataset examples to the instruction prompt format."""
    texts = []
    for instruction, inp, out in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=out)
        text += EOS_TOKEN
        texts.append(text)
    return texts


# ============================================================
# Main
# ============================================================

def main():
    # Load training data
    print("Loading training data...")
    train_data = []
    with open("finetune/train.jsonl", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"  Train samples: {len(train_data)}")

    eval_data = []
    with open("finetune/eval.jsonl", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))
    print(f"  Eval samples: {len(eval_data)}")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # Load model with Unsloth 4-bit
    # Note: RTX 50-series (Blackwell, sm_120) requires Unsloth >= 2025.3
    print(f"\nLoading model: {MODEL_NAME}")
    print(f"  Compute capability: {torch.cuda.get_device_capability()}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        if "bitsandbytes" in str(e).lower() or "CUDA" in str(e):
            print(f"\n  Blackwell 4-bit issue detected: {e}")
            print("  Falling back to 8-bit loading...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=False,
                load_in_8bit=True,
            )
        else:
            raise

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Train
    print("\n" + "=" * 60)
    print("  Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}  Batch: {BATCH_SIZE}×{GRAD_ACCUM}")
    print(f"  LR: {LEARNING_RATE}  LoRA rank: {LORA_R}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    print(f"\nSaving final model to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
