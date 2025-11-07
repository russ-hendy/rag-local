import os
import hashlib
import json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import DOCS_DIR, CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

METADATA_FILE = os.path.join(CHROMA_DIR, "doc_hashes.json")

def file_hash(path):
    """Compute a SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_changed_files():
    """Compare current hashes to saved ones and return new/changed files."""
    old_hashes = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            old_hashes = json.load(f)

    current_hashes = {
        f: file_hash(os.path.join(DOCS_DIR, f))
        for f in os.listdir(DOCS_DIR)
        if f.endswith((".pdf", ".txt", ".md"))
    }

    changed = [
        f for f, h in current_hashes.items()
        if f not in old_hashes or old_hashes[f] != h
    ]

    return changed, current_hashes


def load_and_chunk_with_metadata(filename):
    """Load a file, split into chunks, and add metadata."""
    filepath = os.path.join(DOCS_DIR, filename)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = filename
            doc.metadata["page"] = i + 1
        chunks = splitter.split_documents(docs)

    elif filename.endswith((".txt", ".md")):
        loader = TextLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        chunks = splitter.split_documents(docs)
    else:
        chunks = []

    return chunks


def main():
    print("🔍 Checking for new or updated documents...")
    changed_files, new_hashes = get_changed_files()

    if not changed_files:
        print("✅ No new or updated files. Vector store is up to date.")
        return

    print(f"📚 Found {len(changed_files)} new/updated files: {changed_files}")

    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding) if os.path.exists(CHROMA_DIR) else None

    all_chunks = []
    for filename in changed_files:
        chunks = load_and_chunk_with_metadata(filename)
        all_chunks.extend(chunks)
        print(f"🧩 {filename}: {len(chunks)} chunks added")

    if not all_chunks:
        print("⚠️ No embeddable chunks found.")
        return

    if vectordb:
        vectordb.add_documents(all_chunks)
    else:
        vectordb = Chroma.from_documents(all_chunks, embedding, persist_directory=CHROMA_DIR)

    vectordb.persist()
    print("💾 Chroma database updated with new metadata.")

    with open(METADATA_FILE, "w") as f:
        json.dump(new_hashes, f, indent=2)

    print("✅ Done! Vector store is up to date.")


if __name__ == "__main__":
    main()