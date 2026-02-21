import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="Nom logique de l'agent (ex: dev_raw)")
    parser.add_argument("--question", type=str, required=True, help="Question utilisateur")
    args = parser.parse_args()

    base_model_id = "Qwen/Qwen1.5-1.8B"
    lora_dir = f"models/qwen_lora_agent_{args.agent}/final_lora_adapters"

    print(f"🔌 Base: {base_model_id}")
    print(f"🎯 LoRA: {lora_dir}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=None,
    )

    model = PeftModel.from_pretrained(model, lora_dir)
    model.to(device)
    model.eval()

    prompt = f"### Instruction:\n{args.question}\n\n### Réponse:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "### Réponse:" in text:
        answer = text.split("### Réponse:", 1)[1].strip()
    else:
        answer = text.strip()

    print("\n🧠 Réponse de l'agent :\n")
    print(answer)

if __name__ == "__main__":
    main()
