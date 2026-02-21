#!/usr/bin/env python3
"""Fine-tuning LoRA d'un expert Phi-2 à partir d'un fichier JSONL."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_ID = "microsoft/phi-2"
MAX_SEQ_LENGTH = 512

LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

tokenizer = None


def format_sample(sample):
    global tokenizer
    prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}{tokenizer.eos_token}"
    return {"text": prompt}


def tokenize_sample(sample):
    global tokenizer
    return tokenizer(
        sample["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_attention_mask=True,
    )

def load_dataset(path: Path) -> Dataset:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{path.name}: {len(rows)} exemples chargés")
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    dataset = dataset.map(tokenize_sample, batched=True, remove_columns=["text"])
    return dataset

def train_lora_expert(data_path: Path, output_dir: Path, epochs: int = 3):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_path)

    print("Chargement du modèle de base en bf16…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    final_dir = output_dir / "final_lora_adapters"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adaptateur LoRA sauvegardé dans {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_lora_expert(data_path, output_dir, args.epochs)


if __name__ == "__main__":
    main()
