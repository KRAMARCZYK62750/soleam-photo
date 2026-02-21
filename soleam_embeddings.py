from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np

COLLECTION = "soleam_profiles"
client = QdrantClient(url="http://localhost:6333")

# Charger le modèle
print("🤖 Chargement du modèle d'embeddings...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Profils photographiques
profiles = [
    {
        "id": 1,
        "nom": "Studio Pro Print",
        "description": "Spécialiste impression grand format, bâches publicitaires 4x3m, panneaux d'exposition.",
        "tags": ["impression", "grand format", "CMJN"]
    },
    {
        "id": 2,
        "nom": "Food Style Photo",
        "description": "Photographie culinaire haut de gamme, stylisme gastronomique. Clients : restaurants étoilés.",
        "tags": ["culinaire", "food", "restaurant"]
    }
]

# Générer embeddings
texts = [p["description"] for p in profiles]
embeddings = model.encode(texts, show_progress_bar=True)

print(f"✓ Embeddings générés : dimension {embeddings.shape[1]}")

# Créer collection
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)

# Insérer
points = [
    PointStruct(
        id=p["id"],
        vector=embeddings[i].tolist(),
        payload=p
    )
    for i, p in enumerate(profiles)
]
client.upsert(collection_name=COLLECTION, points=points)

# Test
query = "restaurant gastronomique"
results = client.search(
    collection_name=COLLECTION,
    query_vector=model.encode(query).tolist(),
    limit=2
)

print(f"\n🔎 Recherche '{query}':")
for r in results:
    print(f"→ {r.payload['nom']} (score: {r.score:.3f})")
