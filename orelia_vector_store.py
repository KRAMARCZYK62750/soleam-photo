"""Helpers for managing Orélia's Qdrant profile store and searching embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "orelia_profiles"
MODEL_NAME = "distiluse-base-multilingual-cased-v2"
QDRANT_URL = "http://localhost:6333"

PROFILES = [
    {
        "id": 1,
        "text": (
            "Profil impression grand format : bâches, backlit, profils ICC, résolutions dpi selon la distance"
            " de vision, choix des supports rigides ou textiles, gestion du contrecollage et du vernis."
        ),
        "tag": "impression_grand_format",
    },
    {
        "id": 2,
        "text": (
            "Profil studio lumière : portrait corporate, set Rembrandt, clamshell et split lighting,"
            " softbox 90 cm, réflecteurs argent/blanc, fond papier et gels CTO/CTB pour contrôler l'ambiance."
        ),
        "tag": "studio",
    },
    {
        "id": 3,
        "text": (
            "Profil astrophotographie : Voie lactée, Perséides, règle des 500, stacking long pose,"
            " suivi équatorial Star Adventurer, balance des blancs autour de 3800 K."
        ),
        "tag": "astro",
    },
    {
        "id": 4,
        "text": (
            "Profil food & stylisme culinaire : textures croustillantes, stylisme rustique, gestion de la condensation,"
            " fonds bois/ardoise, éclairage latéral pour les matières, props cohérents (lin, céramique)."
        ),
        "tag": "food",
    },
    {
        "id": 5,
        "text": (
            "Profil mariage & reportage : gestion de la balance des blancs mixte (église/salle),"
            " ISO auto 1600-6400, flash fill-in discret, storytelling sur trois chansons, liste des immanquables."
        ),
        "tag": "mariage",
    },
]


@dataclass
class ProfileHit:
    id: int
    score: float
    tag: str | None
    text: str | None


@lru_cache()
def get_embed_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache()
def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def _build_points(model: SentenceTransformer, profiles: Sequence[dict]) -> List[PointStruct]:
    texts = [p["text"] for p in profiles]
    embeddings = model.encode(texts).tolist()
    return [
        PointStruct(
            id=profiles[i]["id"],
            vector=embeddings[i],
            payload={"text": profiles[i]["text"], "tag": profiles[i]["tag"]},
        )
        for i in range(len(profiles))
    ]


def ensure_profiles_collection(
    *, client: QdrantClient | None = None, model: SentenceTransformer | None = None
) -> None:
    client = client or get_qdrant_client()
    model = model or get_embed_model()

    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        current_dim = info.config.params.vectors.size
        expected_dim = model.get_sentence_embedding_dimension()
        if current_dim == expected_dim:
            return
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )

    client.upsert(collection_name=COLLECTION_NAME, points=_build_points(model, PROFILES))


def search_profiles(
    question: str,
    *,
    limit: int = 3,
    client: QdrantClient | None = None,
    model: SentenceTransformer | None = None,
) -> List[ProfileHit]:
    client = client or get_qdrant_client()
    model = model or get_embed_model()
    ensure_profiles_collection(client=client, model=model)

    query_vec = model.encode(question).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=limit,
        with_payload=True,
    )

    hits: List[ProfileHit] = []
    for r in results:
        payload = r.payload or {}
        hits.append(
            ProfileHit(
                id=r.id,
                score=float(r.score),
                tag=payload.get("tag"),
                text=payload.get("text"),
            )
        )
    return hits


def format_profile_context(hits: Iterable[ProfileHit]) -> str:
    parts = []
    for hit in hits:
        label = hit.tag or "profil"
        snippet = hit.text or ""
        parts.append(f"[{label}] score={hit.score:.3f} → {snippet}")
    return "\n".join(parts)
