#!/usr/bin/env python3
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "microsoft/phi-2"
LORA_BASE_PATH = "./models"

AGENT_MAP = {
    "phi2_lora_agent_expo_iso": ["iso", "ettr", "vitesse", "ouverture", "histogramme", "zone", "dynamique"],
    "phi2_lora_agent_focale_distance": ["focale", "compression", "distance", "85mm", "24mm"],
    "phi2_lora_agent_lumiere_wb": ["balance", "wb", "kelvin", "lumière", "tungstene", "dominante"],
    "phi2_lora_agent_autofocus_net": ["af", "collimateur", "mise au point", "eye", "diffraction"],
    "phi2_lora_agent_composition": ["règle", "cadre", "ligne", "fond", "tension"],
    "phi2_lora_agent_gestion_mouvement": ["filé", "stabilisation", "pose longue", "freeze", "trépied"],
    "phi2_lora_agent_boitiers_capteurs": ["capteur", "hybride", "plein format", "iso invariance"],
    "phi2_lora_agent_objectifs_parc": ["objectif", "mtf", "piqué", "aberration", "macro"],
    "phi2_lora_agent_accessoires": ["trépied", "filtre", "pola", "softbox", "accessoire"],
    "phi2_lora_agent_studio": [
        "studio", "éclairage", "softbox", "beauty dish", "fond blanc",
        "clamshell", "rim light", "high key", "low key", "gélatine"
    ],
    "phi2_lora_agent_maintenance": ["nettoyage", "capteur", "backup", "batterie"],
    "phi2_lora_agent_portrait": ["portrait", "studio", "model", "peau"],
    "phi2_lora_agent_paysage": ["paysage", "gnd", "hyperfocale", "panorama"],
    "phi2_lora_agent_street": ["street", "reportage", "candid", "foule"],
    "phi2_lora_agent_macro_produit": ["macro", "produit", "focus stacking", "halo"],
    "phi2_lora_agent_architecture": [
        "verticales", "lignes droites", "déformation", "tilt-shift",
        "immobilier", "hdr", "lumière mixte", "perspective",
        "alignement", "géométrie", "obliques", "lignes de fuite"
    ],
    "phi2_lora_agent_astrophotographie": ["astro", "voie", "étoile", "pollution"],
    "phi2_lora_agent_evenementiel": ["mariage", "concert", "événement"],
    "phi2_lora_agent_dev_raw": ["raw", "clarté", "dehaze", "curves"],
    "phi2_lora_agent_couleur": ["couleur", "lut", "grading", "film"],
    "phi2_lora_agent_retouche_loc": ["retouche", "dodge", "burn", "halo"],
    "phi2_lora_agent_bruit_net": ["bruit", "netteté", "denoise", "output"],
    "phi2_lora_agent_tirage_format": [
        "tirage", "dpi", "print", "format", "impression", "résolution",
        "interpolation", "agrandissement", "1m", "300", "export"
    ],
    "phi2_lora_agent_papier_encres": ["papier", "fine art", "ic c"],
    "phi2_lora_agent_prepresse": ["softproof", "pré presse", "profil"],
    "phi2_lora_agent_portfolio": ["portfolio", "sélection", "rythme"],
    "phi2_lora_agent_diffusion_web": ["instagram", "web", "export"],
    "phi2_lora_agent_catalogage_tags": ["catalogue", "tags", "iptc"],
    "phi2_lora_agent_coaching": ["coaching", "progression", "exercice"],
    "phi2_lora_agent_critique_photo": ["critique", "analyse", "commentaire"],
    "phi2_lora_agent_references": ["référence", "photographe", "histoire"],
}

base_model = None
tokenizer = None
current_adapter = None


def load_base_model():
    global base_model, tokenizer
    if base_model is None:
        print("Chargement du modèle base…")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token


def route_query(text: str) -> str:
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    best_agent, best_score = None, -1
    for agent, keywords in AGENT_MAP.items():
        score = len(words.intersection(set(k.lower() for k in keywords)))
        if score > best_score:
            best_agent, best_score = agent, score
    if best_agent is None:
        best_agent = "phi2_lora_agent_expo_iso"
    print(f"Requête routée vers: {best_agent}")
    return os.path.join(LORA_BASE_PATH, best_agent, "final_lora_adapters")


def generate_response(query: str) -> str:
    global current_adapter, base_model, tokenizer
    load_base_model()
    lora_path = route_query(query)
    if current_adapter != lora_path:
        print(f"Chargement adaptateur {lora_path}")
        base_model = PeftModel.from_pretrained(base_model, lora_path)
        current_adapter = lora_path

    prompt = f"### Instruction:\n{query}\n\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_clean = response.split("### Response:")[-1].strip()

    agent_name = os.path.basename(os.path.dirname(lora_path))
    if agent_name in ["phi2_lora_agent_portrait", "phi2_lora_agent_macro_produit"]:
        response_clean += "\n\n***\n💡 ADN SOLÉAM : privilégiez une lumière ciselée de studio et un halo optique maîtrisé."

    print(response_clean)
    return response_clean


if __name__ == "__main__":
    questions = [
        "Quels réglages ISO pour éviter le bruit en basse lumière ?",
        "Comment corriger les verticales en photo d'architecture ?",
        "Quelle valeur de dpi pour un tirage fine art ?",
        "Quelles références pour un portrait studio dramatique ?",
    ]
    for q in questions:
        print("\n=== Question ===")
        print(q)
        print("--- Réponse ---")
        print(generate_response(q))
