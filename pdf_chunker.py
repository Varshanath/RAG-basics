from pathlib import Path

import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("Policy Raw docs/HR_Policy/HR_Policy_Manual.pdf")

reader = PdfReader(PDF_PATH)
text = "\n".join(page.extract_text() or "" for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

print(f"Extracted {len(text)} characters from {len(reader.pages)} pages in '{PDF_PATH.name}'")
print(f"Split into {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
    print()

#embedding the chunks and storing them in ChromaDB, tagged with department metadata
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks).tolist()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="policy_documents")

collection.add(
    ids=[f"{PDF_PATH.stem}_chunk_{i}" for i in range(len(chunks))],
    embeddings=chunk_embeddings,
    documents=chunks,
    metadatas=[{"department": "HR", "source": PDF_PATH.name} for _ in chunks],
)

print(f"Added {len(chunks)} chunks to ChromaDB collection '{collection.name}' with metadata department=HR")
