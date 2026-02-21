import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base, lora, out):
    print("🔌 Chargement du modèle de base…")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)

    print("🎯 Application du LoRA…")
    model = PeftModel.from_pretrained(model, lora)

    print("🔧 Fusion des poids LoRA dans le modèle…")
    model = model.merge_and_unload()

    print(f"💾 Sauvegarde du modèle fusionné dans : {out}")
    model.save_pretrained(out)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(out)

    print("✅ Fusion terminée !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model")
    parser.add_argument("--lora_path")
    parser.add_argument("--output")
    args = parser.parse_args()

    merge_lora(args.base_model, args.lora_path, args.output)
