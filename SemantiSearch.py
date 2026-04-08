from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

ds = load_dataset("fka/prompts.chat", split="train[:2000]")

texts = [item["prompt"] + " " + item.get("response", "") for item in ds]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

query = "how to learn programming"

query_embedding = model.encode([query])

k = 5
distances, indices = index.search(np.array(query_embedding), k)

for i in indices[0]:
    print(texts[i])
    print()