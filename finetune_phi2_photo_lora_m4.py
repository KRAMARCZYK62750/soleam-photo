import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

MODEL_ID = "microsoft/phi-2"
DATASET_PATH = "/Users/fredericmaryline/projects/soleam-photo/data/photo_learning_dataset.jsonl"
OUTPUT_DIR = "./phi2_photo_lora_output"
LOGS_DIR = "./phi2_photo_lora_logs"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
MAX_SEQ_LENGTH = 512

BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32


def format_prompt(example, local_tokenizer):
    instruction = example["instruction"]
    response = example["response"]
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{local_tokenizer.eos_token}"
    return {"text": text}


def main():
    print(f"Chargement du modèle : {MODEL_ID} en {DTYPE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    expanded_data_path = os.path.expanduser(DATASET_PATH)
    print(f"Chargement du dataset depuis : {expanded_data_path}")
    dataset = load_dataset("json", data_files={"train": expanded_data_path})
    train_dataset = dataset["train"].map(
        lambda ex: format_prompt(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGS_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="adamw_torch",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        fp16=(DTYPE == torch.float16),
        bf16=(DTYPE == torch.bfloat16),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("\n🚀 Démarrage du Fine-Tuning LoRA sur Mac M4/MPS...")
    trainer.train()

    final_adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapters")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"\n✅ Fine-tuning terminé. Adaptateurs LoRA sauvegardés dans : {final_adapter_path}")


if __name__ == "__main__":
    main()
