from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
