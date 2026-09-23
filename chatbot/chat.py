import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Embedding model used to turn text into vectors. Must be the SAME model that was
# used to embed the documents in ingest_documents.py/pdf_chunker.py, otherwise the
# query vectors won't be comparable to the stored document vectors.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connects to the on-disk vector database created by ingest_documents.py.
# PersistentClient means the data survives between runs (unlike chromadb.EphemeralClient,
# which only keeps data in memory for the current process).
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policy_documents")

# Keeps the running back-and-forth of the conversation (as role/content dicts) so the
# LLM has memory of earlier turns. Only real user/assistant messages go here - the
# retrieved document context is injected fresh each turn via system_message instead,
# so it doesn't pile up and bloat the conversation on every turn.
conversation_history = []

print("Policy chatbot ready. Type 'exit' or 'quit' to end the conversation.\n")

while True:
    user_query = input("You: ")
    if user_query.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Embed the user's question, then find the 3 most similar document chunks in
    # ChromaDB (nearest neighbor search over the stored embeddings).
    query_embedding = model.encode([user_query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas"],
    )
    # collection.query() returns lists-of-lists (one inner list per input query embedding).
    # Since we only sent one query, we take index [0] to get the results for it.
    retrieved_docs = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    # Number each retrieved chunk ([1], [2], [3]...) and tag it with where it came from,
    # so the model can cite its sources and we can print a matching Sources list below.
    context_blocks = [
        f"[{i}] (source: {meta['source']}, department: {meta['department']})\n{doc}"
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas), start=1)
    ]
    retrieved_context = "\n\n".join(context_blocks)

    # The system prompt is rebuilt every turn with this turn's retrieved context,
    # instructing the model to answer only from it and cite sources by number.
    system_message = (
        "You are a helpful policy chatbot. Answer the user's question using only the "
        "context below. Cite sources inline using their bracketed number, e.g. [1]. "
        "If the context doesn't contain the answer, say so explicitly.\n\n"
        f"Context:\n{retrieved_context}"
    )

    conversation_history.append({"role": "user", "content": user_query})

    # Sends the fresh system prompt (with this turn's context) plus the full
    # conversation history so far, so the model can use earlier turns for context
    # (e.g. understanding a follow-up question like "what about sick leave?").
    response = ollama.chat(model='llama2', messages=[
        {"role": "system", "content": system_message},
        *conversation_history,
    ])

    answer = response['message']['content']
    print(f"Bot: {answer}\n")

    # Prints which documents backed this answer, so the citation numbers in the
    # answer above (e.g. "[1]") can be matched back to an actual source file.
    print("Sources:")
    for i, meta in enumerate(retrieved_metadatas, start=1):
        print(f"  [{i}] {meta['source']} (department: {meta['department']})")
    print()

    # Save the assistant's reply into history too, so it's included in the next
    # turn's messages and the conversation stays coherent.
    conversation_history.append({"role": "assistant", "content": answer})
