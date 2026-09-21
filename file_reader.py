from pathlib import Path

POLICY_DOCS_DIR = Path("Policy Raw docs")

file_paths = sorted(POLICY_DOCS_DIR.rglob("*.txt"))
documents = []
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        documents.append(file.read())

print(f"Loaded {len(documents)} documents from '{POLICY_DOCS_DIR}':")
for file_path, document in zip(file_paths, documents):
    print(f"\n--- {file_path} ---")
    print(document)
