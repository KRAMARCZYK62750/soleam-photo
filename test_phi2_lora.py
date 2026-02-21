import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "microsoft/phi-2"
ADAPTERS_PATH = "./phi2_photo_lora_output/final_lora_adapters"
DTYPE = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32


def load_model_with_adapters():
    print(f"Chargement du modèle de base ({MODEL_ID})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Chargement des adaptateurs LoRA depuis {ADAPTERS_PATH}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTERS_PATH)
    peft_model.eval()
    return peft_model, tokenizer


def generate_response(instruction, model, tokenizer):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_full = tokenizer.decode(output[0], skip_special_tokens=True)
    parts = response_full.split("### Response:")
    if len(parts) < 2:
        return "Impossible d'extraire la réponse."
    return parts[1].strip()


def main():
    model, tokenizer = load_model_with_adapters()
    tests = [
        "J'ai pris une photo de nuit mais elle est trop sombre, quels réglages vérifier ?",
        "Qu'est-ce que l'espace négatif ?",
        "Comment créer un flou d'arrière-plan avec un objectif f/5.6 ?",
    ]
    for instruction in tests:
        print(f"\n[Q] {instruction}")
        print(f"[R] {generate_response(instruction, model, tokenizer)}")


if __name__ == "__main__":
    main()
