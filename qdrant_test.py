from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "orelia_test"
embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

client = QdrantClient(url="http://localhost:6333")

# 1) (Re)création de la collection avec la bonne dimension
if client.collection_exists(COLLECTION):
    print(f"ℹ️ Collection '{COLLECTION}' existante, suppression…")
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=EMBED_DIM,
        distance=Distance.COSINE,
    ),
)
print(f"🆕 Collection '{COLLECTION}' créée (dimension {EMBED_DIM}).")

# 2) Profils à indexer
profiles = [
    {
        "id": 1,
        "text": "Profil spécialisé impression grand format : bâches, panneaux, backlit, résolutions optimales et profils ICC.",
        "tag": "impression_grand_format",
    },
    {
        "id": 2,
        "text": "Profil studio : lumière Rembrandt, portraits corporate, gestion des softbox et réflecteurs.",
        "tag": "studio",
    },
    {
        "id": 3,
        "text": "Profil astrophotographie : Perséides, Voie lactée, réglages longue pose et stacking d’images.",
        "tag": "astro",
    },
]

texts = [p["text"] for p in profiles]
vectors = embed_model.encode(texts).tolist()

points = [
    PointStruct(
        id=profiles[i]["id"],
        vector=vectors[i],
        payload={"text": profiles[i]["text"], "tag": profiles[i]["tag"]},
    )
    for i in range(len(profiles))
]

client.upsert(collection_name=COLLECTION, points=points)
print("✅ Points insérés dans Qdrant.")

# 3) Recherche
query_text = "Quelle résolution choisir pour une bâche publicitaire grand format ?"
query_vector = embed_model.encode(query_text).tolist()

results = client.search(
    collection_name=COLLECTION,
    query_vector=query_vector,
    limit=3,
    with_payload=True,
)

print("\n🔍 Résultats de la recherche :")
for r in results:
    payload = r.payload or {}
    print(f"- id={r.id}, score={r.score:.4f}, tag={payload.get('tag')}, text={payload.get('text')}")
