import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "/Users/fredericmaryline/.cache/huggingface/hub/models--microsoft--phi-2/snapshots/ef382358ec9e382308935a992d908de099b64c23"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="Suffixe de l'agent (ex: dev_raw_v2)")
    parser.add_argument("--question", type=str, required=True, help="Question utilisateur")
    args = parser.parse_args()

    base_model_id = BASE_MODEL_PATH
    lora_path = f"models/phi2_lora_agent_{args.agent}/final_lora_adapters"

    print(f"🔌 Base (local): {base_model_id}")
    print(f"🎯 LoRA: {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
    )

    model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)
    model.eval()

    prompt = f"""### Instruction:\n{args.question}\n\n### Réponse:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
