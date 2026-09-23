import chromadb
import ollama
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policy_documents")

conversation_history = []

print("Policy chatbot ready. Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_query = input("You: ")
    if user_query.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    query_embedding = model.encode([user_query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas"],
    )
    retrieved_docs = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    context_blocks = [
        f"[{i}] (source: {meta['source']}, department: {meta['department']})\n{doc}"
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas), start=1)
    ]
    retrieved_context = "\n\n".join(context_blocks)

    system_message = (
        "You are a helpful policy chatbot. Answer the user's question using only the "
        "context below. Cite sources inline using their bracketed number, e.g. [1]. "
        "If the context doesn't contain the answer, say so explicitly.\n\n"
        f"Context:\n{retrieved_context}"
    )

    conversation_history.append({"role": "user", "content": user_query})

    response = ollama.chat(model='llama2', messages=[
        {"role": "system", "content": system_message},
        *conversation_history,
    ])

    answer = response['message']['content']
    print(f"Bot: {answer}\n")

    print("Sources:")
    for i, meta in enumerate(retrieved_metadatas, start=1):
        print(f"  [{i}] {meta['source']} (department: {meta['department']})")
    print()

    conversation_history.append({"role": "assistant", "content": answer})
