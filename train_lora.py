import argparse
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def formatting_prompts_func(examples, tokenizer, instruction_key="instruction", response_key="response") -> List[str]:
    """
    Assemble instruction + response into a single prompt string compatible with Phi-2 fine-tuning.
    """
    instructions = examples[instruction_key]
    responses = examples[response_key]
    output_texts = []
    for instruction, response in zip(instructions, responses):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="microsoft/phi-2")
    parser.add_argument("--cache_dir", default="models/phi-2")
    parser.add_argument("--dataset_path", default="data/photo_learning_dataset.jsonl")
    parser.add_argument("--output_dir", default="lora_runs/phi2-photo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(dataset_path))

    def tokenize_function(examples):
        texts = formatting_prompts_func(examples, tokenizer)
        return tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding="max_length",
        )

    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        weight_decay=0.01,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
