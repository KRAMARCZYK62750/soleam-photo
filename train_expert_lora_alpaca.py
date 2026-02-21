import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


def format_example(inst: str, resp: str) -> str:
    return "### Instruction:\n" + inst.strip() + "\n\n### Réponse:\n" + resp.strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="JSONL avec instruction/response")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_model", default="microsoft/phi-2")
    args = parser.parse_args()

    base_model_id = args.base_model
    print(f"📂 Dataset : {args.data_path}")
    print(f"💾 Output  : {args.output_dir}")
    print(f"🧠 Base    : {base_model_id}")

    dataset = load_dataset("json", data_files=str(args.data_path), split="train")

    def build_text(example):
        inst = example.get("instruction", "")
        resp = example.get("response", "")
        return {"text": format_example(inst, resp)}

    dataset = dataset.map(build_text)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=1024, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

    final_dir = Path(args.output_dir) / "final_lora_adapters"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"✅ Adaptateur LoRA sauvegardé dans {final_dir}")


if __name__ == "__main__":
    main()
