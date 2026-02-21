#!/usr/bin/env python3
# orelia_cli.py
"""
Petit client CLI pour Orélia (mode RAG pur).

Usage simple :
    python3 orelia_cli.py "Ta question"

Usage interactif (sans argument) :
    python3 orelia_cli.py
    > Ta question...
"""

import argparse
import sys

from rag_expert import ask_agent

try:
    from orelia_vector_store import format_profile_context, search_profiles

    VECTOR_SEARCH_AVAILABLE = True
    VECTOR_SEARCH_ERROR = None
except Exception as exc:  # pragma: no cover - best-effort optional dependency
    VECTOR_SEARCH_AVAILABLE = False
    VECTOR_SEARCH_ERROR = str(exc)


# --- Router très simple par mots-clés ---


def route_agent(question: str) -> str:
    """Retourne le nom de l'agent le plus probable en fonction de la question."""

    q = question.lower()

    # Astrophotographie
    if any(
        k in q
        for k in (
            "voie lactée",
            "voie lactee",
            "astrophotographie",
            "astro",
            "nébuleuse",
            "nebuleuse",
            "galaxie",
            "andromède",
            "andromede",
            "bortle",
            "monture équatoriale",
            "monture equatoriale",
            "star adventurer",
            "perséides",
            "perseides",
            "aurores boréales",
            "aurores boreales",
        )
    ):
        return "astro"

    # Flash / reportage
    if any(
        k in q
        for k in (
            "flash",
            "synchro lente",
            "synchro rideau",
            "rideau arrière",
            "rideau arriere",
            "rebond",
            "fill-in",
            "fill in",
            "hss",
            "high speed sync",
            "godox",
            "profoto",
            "ttl",
        )
    ):
        return "flash_reportage"

    # Moodboard / ambiance visuelle
    if any(
        k in q
        for k in (
            "moodboard",
            "ambiance visuelle",
            "palette",
            "univers visuel",
            "références visuelles",
            "inspiration",
        )
    ):
        return "moodboard"

    # Scénographie / décor
    if any(
        k in q
        for k in (
            "scenographie",
            "scénographie",
            "décor",
            "deco",
            "plateau",
            "fond de scène",
            "fond de scene",
            "palette lumière",
            "palette lumiere",
            "fumée",
            "fumee",
            "décor club",
            "deco club",
        )
    ):
        return "scenographie"

    # Concert / scène
    if any(
        k in q
        for k in (
            "concert",
            "scène",
            "scene",
            "live",
            "festival",
            "bar",
            "pub",
            "club",
            "boîte",
            "boite",
        )
    ):
        return "concert_expo"

    # Papier / tirage
    if any(
        k in q
        for k in ("papier", "canson", "hahnemühle", "hahnemuhle", "tirage", "fine art")
    ):
        return "papier_tirage"

    # WB / couleur
    if any(
        k in q
        for k in (
            "balance des blancs",
            "wb ",
            "dominante",
            "température de couleur",
            "temperature de couleur",
        )
    ):
        return "lumiere_wb"

    # Studio lighting setups
    if any(
        k in q
        for k in (
            "setup lumière",
            "setup lumiere",
            "lighting",
            "rembrandt",
            "high key",
            "low key",
            "clamshell",
            "rim light",
            "gélatine",
            "gelatine",
            "gels",
            "softbox",
            "parapluie",
        )
    ):
        return "studio_light"

    # Maquillage
    if any(
        k in q
        for k in (
            "maquillage",
            "makeup",
            "make-up",
            "teint",
            "fond de teint",
            "fard",
            "eyeliner",
            "rouge à lèvres",
            "rouge a levres",
        )
    ):
        return "makeup"

    # Composition / cadrage
    if any(
        k in q
        for k in (
            "composition",
            "règle des tiers",
            "regle des tiers",
            "cadrage",
            "lignes directrices",
            "lignes de fuite",
            "horizon",
        )
    ):
        return "composition"

    # Post-traitement
    if any(
        k in q
        for k in (
            "post-traitement",
            "post traitement",
            "retouche",
            "lightroom",
            "capture one",
            "courbe",
            "hsl",
            "export",
        )
    ):
        return "post_traitement"

    # Macro
    if any(
        k in q
        for k in (
            "macro",
            "rapport 1:1",
            "1:1",
            "focus stacking",
            "insecte",
            "macro photographie",
        )
    ):
        return "macro"

    # Studio (général)
    if any(
        k in q
        for k in (
            "studio",
            "portrait studio",
            "fond studio",
            "fond blanc",
            "fond noir",
            "setup studio",
        )
    ):
        return "studio"

    # Impression grand format
    if any(
        k in q
        for k in (
            "grand format",
            "tirage grand format",
            "dpi",
            "épreuvage",
            "fine art",
            "profil icc",
            "calibration écran",
            "bâche",
            "bache",
            "dibond",
            "akilux",
            "backlit",
            "caisson lumineux",
            "expo 1,5 m",
            "expo 1.5 m",
            "100 x 150",
            "1m50",
        )
    ):
        return "impression_grand_format"

    # Parc optique / focales
    if any(
        k in q for k in ("focale", "parc optique", "24-70", "70-200", "35mm", "50mm", "85mm")
    ):
        return "objectifs_parc"

    # Par défaut : développement RAW
    return "dev_raw"


def fetch_profile_context(question: str, *, limit: int = 3) -> str | None:
    """Retourne un bloc texte décrivant les profils Qdrant les plus proches."""

    global VECTOR_SEARCH_AVAILABLE, VECTOR_SEARCH_ERROR

    if not VECTOR_SEARCH_AVAILABLE:
        return None

    try:
        hits = search_profiles(question, limit=limit)
    except Exception as exc:  # pragma: no cover - dépendant de Qdrant
        VECTOR_SEARCH_AVAILABLE = False
        VECTOR_SEARCH_ERROR = str(exc)
        return None

    if not hits:
        return None

    return format_profile_context(hits)


def main():
    parser = argparse.ArgumentParser(description="Orélia CLI (RAG photo).")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question à poser à Orélia (si vide → mode interactif)",
    )

    args = parser.parse_args()

    # Mode interactif si pas d’argument
    if not args.question:
        print("🎧 Orélia CLI – mode interactif (Ctrl+C pour quitter)\n")
        while True:
            try:
                q = input("❓ Question : ").strip()
                if not q:
                    continue

                agent = route_agent(q)
                print(f"\n🤖 Agent sélectionné : {agent}\n")

                profile_context = fetch_profile_context(q)
                if profile_context:
                    print("📚 Contextes Orélia (Qdrant) :\n")
                    print(profile_context)
                    print()

                answer = ask_agent(agent, q)
                print("🧠 Réponse :\n")
                print(answer)
                print("\n" + "=" * 80 + "\n")

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Fin de la session Orélia.")
                break

    # Mode one-shot avec argument
    else:
        question = args.question.strip()
        agent = route_agent(question)
        print(f"🤖 Agent sélectionné : {agent}\n")

        profile_context = fetch_profile_context(question)
        if profile_context:
            print("📚 Contextes Orélia (Qdrant) :\n")
            print(profile_context)
            print()

        answer = ask_agent(agent, question)
        print("🧠 Réponse :\n")
        print(answer)


if __name__ == "__main__":
    main()
