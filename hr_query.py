"""import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policy_documents")

query_text = input("Enter your query: ")
query_embedding = model.encode([query_text]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    where={"department": "HR"},
    include=["distances", "documents", "metadatas"],
)

for doc_id, distance, document, metadata in zip(
    results["ids"][0], results["distances"][0], results["documents"][0], results["metadatas"][0]
):
    print(f"\n[{doc_id}] (distance: {distance:.4f}) department={metadata['department']}")
    print(document)"""
