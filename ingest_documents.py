from pathlib import Path

import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

POLICY_DOCS_DIR = Path("Policy Raw docs")


def infer_department(file_path: Path) -> str:
    relative = file_path.relative_to(POLICY_DOCS_DIR)
    if len(relative.parts) > 1:
        # nested under a subfolder, e.g. "Finance_Policy/..."
        folder = relative.parts[0]
        return folder.replace("_Policy", "").replace("_", " ")
    # no subfolder - infer from filename prefix
    name = file_path.stem.lower()
    if name.startswith("it_"):
        return "IT"
    if name.startswith("procurement_"):
        return "Procurement"
    if name.startswith("hr_"):
        return "HR"
    if name.startswith("finance_"):
        return "Finance"
    return "General"


def read_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8")


file_paths = sorted(POLICY_DOCS_DIR.rglob("*.txt")) + sorted(POLICY_DOCS_DIR.rglob("*.pdf"))

model = SentenceTransformer('all-MiniLM-L6-v2')
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection(name="policy_documents")
collection = client.get_or_create_collection(name="policy_documents")

department_counts = {}

for file_path in file_paths:
    text = read_file(file_path)
    if not text.strip():
        print(f"Skipping empty file: {file_path}")
        continue

    department = infer_department(file_path)
    chunks = splitter.split_text(text)
    chunk_embeddings = model.encode(chunks).tolist()

    collection.add(
        ids=[f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))],
        embeddings=chunk_embeddings,
        documents=chunks,
        metadatas=[{"department": department, "source": file_path.name} for _ in chunks],
    )

    department_counts[department] = department_counts.get(department, 0) + len(chunks)
    print(f"Ingested {len(chunks)} chunks from '{file_path}' (department={department})")

print(f"\nTotal chunks in collection: {collection.count()}")
print("Chunks per department:")
for department, count in sorted(department_counts.items()):
    print(f"  - {department}: {count}")
