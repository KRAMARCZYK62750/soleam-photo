import argparse
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen1.5-1.8B",
    )
    return parser

def format_example(example):
    instr = example["instruction"].strip()
    resp = example["response"].strip()
    text = f"### Instruction:\n{instr}\n\n### Réponse:\n{resp}"
    return {"text": text}

def main():
    args = build_argparser().parse_args()

    print(f"📂 Dataset : {args.data_path}")
    print(f"💾 Output  : {args.output_dir}")
    print(f"🧠 Base    : {args.base_model}")

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.map(format_example)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    if torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = {"": "mps"}
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto"

    print("🔌 Chargement du modèle Qwen…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=False,
        fp16=(dtype == torch.float16),
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    final_dir = args.output_dir.rstrip("/") + "/final_lora_adapters"
    print(f"✅ Sauvegarde LoRA dans {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

if __name__ == "__main__":
    main()
