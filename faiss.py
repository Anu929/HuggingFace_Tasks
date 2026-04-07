from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Artificial intelligence is transforming the world",
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is widely used for data science",
    "FAISS helps in fast similarity search",
    "Vector databases store embeddings efficiently"
]

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents).astype("float32")

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

while True:
    query = input("Enter your query: ")
    if query == "exit":
        break

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, 3)

    print("\nTop Results:")
    for i in indices[0]:
        print(documents[i])
    print()