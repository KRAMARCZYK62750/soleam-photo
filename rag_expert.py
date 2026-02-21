import json
from pathlib import Path
from difflib import SequenceMatcher
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(var, None)

BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

AGENT_FILE_MAP = {
    "dev_raw": "agent_dev_raw.jsonl",
    "couleur": "agent_couleur.jsonl",
    "concert_expo": "agent_concert_expo.jsonl",
    "papier_tirage": "agent_papier_tirage.jsonl",
    "astro": "agent_astro.jsonl",
    "flash_reportage": "agent_flash_reportage.jsonl",
    "composition": "agent_composition.jsonl",
    "moodboard": "agent_moodboard.jsonl",
    "post_traitement": "agent_post_traitement.jsonl",
    "macro": "agent_macro.jsonl",
    "studio": "agent_studio.jsonl",
    "studio_light": "agent_studio_light.jsonl",
    "makeup": "agent_makeup.jsonl",
    "impression_grand_format": "agent_impression_grand_format.jsonl",
    "scenographie": "agent_scenographie.jsonl",
    "lumiere_wb": "agent_lumiere_wb.jsonl",
    "objectifs_parc": "agent_objectifs_parc.jsonl",
    "diffusion_web": "agent_diffusion_web.jsonl",
    "style_orelia": "agent_style_orelia.jsonl",
}

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"🔌 Chargement du modèle chat : {BASE_MODEL_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map=None,
        )
        _model.to(DEVICE)
        _model.eval()
    return _tokenizer, _model


def load_agent_dataset(agent_name: str):
    if agent_name not in AGENT_FILE_MAP:
        raise ValueError(f"Agent inconnu: {agent_name}")

    path = Path(AGENT_FILE_MAP[agent_name])
    if not path.exists():
        raise FileNotFoundError(f"Fichier dataset introuvable: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def retrieve_examples(agent_name: str, question: str, k: int = 5):
    dataset = load_agent_dataset(agent_name)
    scored = []
    for row in dataset:
        instr = row.get("instruction", "")
        score = similarity(question, instr)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in scored[:k]]
    return top


def generate_with_model(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or pad_id
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_prompt(agent_name: str, question: str, examples: list[dict]) -> str:
    exemples_txt = []
    for i, ex in enumerate(examples, 1):
        instr = ex["instruction"]
        resp = ex["response"]
        exemples_txt.append(
            f"Exemple {i} - Question : {instr}\nExemple {i} - Réponse : {resp}"
        )
    exemples_block = "\n\n".join(exemples_txt)

    prompt = f"""Tu es un expert en développement RAW et traitement d'images pour un photographe.
Tu dois répondre UNIQUEMENT à partir des exemples ci-dessous, qui sont des extraits de ton manuel interne.
Tu n'as PAS le droit d'inventer des concepts techniques qui ne figurent pas dans ces exemples, ni de parler de modèles physiques complexes
(rayonnement du soleil, modèles mathématiques, etc.).

Si les exemples ne suffisent pas pour répondre précisément, tu dois le dire clairement :
"À partir de ces exemples, je ne peux pas répondre avec certitude."
puis proposer au maximum une piste PRUDENTE, en restant dans le cadre du développement RAW photo.

----------------- EXEMPLES -----------------

{exemples_block}

----------------- QUESTION -----------------

Question : {question}

----------------- CONSIGNES DE RÉPONSE -----------------
- Réponds en français.
- Donne 3 à 6 points concrets (liste numérotée).
- Reste strictement dans le contexte du développement RAW (curseurs, hautes lumières, exposition, etc.).
- N'utilise pas de jargon pseudo-physique qui n'apparaît pas dans les exemples.
- Ne répète pas les exemples mot à mot, adapte-les à la question.

Réponse :
"""
    return prompt


def ask_agent(agent_name: str, question: str) -> str:
    examples = retrieve_examples(agent_name, question, k=3)
    if not examples:
        return "Je n'ai trouvé aucun exemple pour cet agent."

    best = examples[0]
    parts = []
    parts.append("Voici une réponse basée sur ton manuel interne :\n")
    parts.append(best["response"].strip())

    others = examples[1:]
    if others:
        parts.append("\n\nAutres points utiles liés au sujet :")
        for ex in others:
            parts.append(f"- {ex['response'].strip()}")

    return "\n".join(parts).strip()


def run_style_agent(text: str) -> str:
    examples = retrieve_examples("style_orelia", text, k=3)
    exemples_txt = []
    for i, ex in enumerate(examples, 1):
        instr = ex["instruction"]
        resp = ex["response"]
        exemples_txt.append(
            f"Exemple {i} - Instruction : {instr}\nExemple {i} - Réponse : {resp}"
        )
    exemples_block = "\n\n".join(exemples_txt)

    prompt = f"""Tu es Orélia, styliste éditoriale pour un studio photo. Ta mission : lisser la prose fournie tout en restant fidèle au sens et aux données techniques.
Tu dois appliquer une pédagogie douce, un ton magazine maîtrisé et aucune invention.

----------------- EXEMPLES DE STYLE -----------------
{exemples_block}
----------------------------------------------------

Texte à retravailler :
{text}

Réécriture fluide :
"""
    return generate_with_model(prompt, max_new_tokens=400)


def ask_agent_polished(agent_key: str, question: str) -> str:
    """
    1) Demande à l'agent technique (dev_raw, couleur, etc.)
    2) Passe la réponse dans l'agent de style Orélia pour la lisser
    """
    brute = ask_agent(agent_key, question)
    return run_style_agent(
        f"Réécris ce texte de façon plus fluide, pédagogue et agréable à lire, "
        f"sans ajouter de nouvelles informations ni détails techniques, "
        f"et sans changer le sens.\n\nTexte :\n{brute}"
    ).strip()


if __name__ == "__main__":
    # Exemple 1 : dev_raw
    print("\n💬 Question (dev_raw) : Comment récupérer un ciel cramé en RAW ?\n")
    print("🧠 Réponse (RAG brut) :\n")
    print(ask_agent("dev_raw", "Comment récupérer un ciel cramé en RAW ?"))

    # Exemple 2 : concert_expo
    print("\n" + "=" * 80 + "\n")
    print("💬 Question (concert_expo) : Comment régler l'exposition en concert rock ?\n")
    print("🧠 Réponse (RAG brut) :\n")
    print(ask_agent("concert_expo", "Comment régler l'exposition en concert rock ?"))

    # Exemple 3 : papier_tirage
    print("\n" + "=" * 80 + "\n")
    print("💬 Question (papier_tirage) : Quel papier choisir pour un portrait doux ?\n")
    print("🧠 Réponse (RAG brut) :\n")
    print(ask_agent("papier_tirage", "Quel papier choisir pour un portrait doux ?"))
