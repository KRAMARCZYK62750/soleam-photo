import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    print(f"🔌 Chargement du modèle fusionné : {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    prompt = f"Instruction: {args.question}\nRéponse:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\n🧠 Réponse du modèle fusionné :\n")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
