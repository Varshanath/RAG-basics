import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Policy Chatbot", page_icon="\U0001F4C4")
st.title("Policy Chatbot")


@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="policy_documents")
    return model, collection


model, collection = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about company policy...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    query_embedding = model.encode([user_query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents"],
    )
    retrieved_context = "\n\n".join(results["documents"][0])

    system_message = (
        "You are a helpful policy chatbot. Answer the user's question using only the "
        "context below. If the context doesn't contain the answer, say so explicitly.\n\n"
        f"Context:\n{retrieved_context}"
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ollama.chat(model='llama2', messages=[
                {"role": "system", "content": system_message},
                *st.session_state.messages,
            ])
            answer = response['message']['content']
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
