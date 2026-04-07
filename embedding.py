from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")

texts = dataset["text"][:1000]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

def search(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0]]
    return results

query = "Your bank account has been suspended"
results = search(query)

for i, res in enumerate(results):
    print(f"{i+1}. {res}\n")