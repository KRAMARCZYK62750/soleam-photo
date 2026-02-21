#!/usr/bin/env python3
# orelia_brain_map.py

import numpy as np
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

COLLECTION = "soleam_profiles"


def fetch_vectors_from_qdrant():
    client = QdrantClient(url="http://localhost:6333")

    print(f"🧠 Lecture des vecteurs dans la collection '{COLLECTION}'...")

    all_vectors = []
    all_labels = []

    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            with_vectors=True,
            with_payload=True,
            offset=next_offset,
        )

        if not points:
            break

        for p in points:
            vec = p.vector
            if isinstance(vec, dict):
                vec = list(vec.values())[0]

            all_vectors.append(vec)
            payload = p.payload or {}
            label = payload.get("nom") or payload.get("description") or payload.get("tag") or str(p.id)
            all_labels.append(label)

        if next_offset is None:
            break

    if not all_vectors:
        print("⚠️ Aucun vecteur trouvé dans la collection.")
        return None, None

    vectors = np.array(all_vectors, dtype=float)
    return vectors, all_labels


def plot_brain_map(vectors, labels):
    if vectors.shape[0] < 2:
        print("⚠️ Pas assez de points pour une projection (min 2).")
        return

    print("📉 Projection en 2D avec PCA...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vectors)

    x = coords_2d[:, 0]
    y = coords_2d[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.8)

    for xi, yi, label in zip(x, y, labels):
        plt.text(xi + 0.02, yi + 0.02, label, fontsize=9)

    plt.title("🧠 Carte 2D du 'cerveau' d’Orélia (profils Soléam)")
    plt.xlabel("Composante 1 (PCA)")
    plt.ylabel("Composante 2 (PCA)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    vectors, labels = fetch_vectors_from_qdrant()
    if vectors is not None:
        print(f"✅ {vectors.shape[0]} vecteurs chargés, dimension = {vectors.shape[1]}")
        plot_brain_map(vectors, labels)
