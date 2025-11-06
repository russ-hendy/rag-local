import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# 1. Load all documents in folder
docs_dir = "docs"
all_docs = []

for filename in os.listdir(docs_dir):
    filepath = os.path.join(docs_dir, filename)
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filename.endswith(".txt") or filename.endswith(".md"):
        loader = TextLoader(filepath)
    else:
        print(f"Skipping unsupported file type: {filename}")
        continue

    docs = loader.load()
    all_docs.extend(docs)

print(f"✅ Loaded {len(all_docs)} documents from {docs_dir}")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)
print(f"🧩 Split into {len(chunks)} chunks")

# 3. Create embeddings + vector store
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")

# 4. Retrieve relevant docs for a query
query = input("\n💬 Ask a question: ")
results = vectordb.similarity_search(query, k=3)

# 5. Generate an answer using the retrieved context
context = "\n\n".join([r.page_content for r in results])
llm = Ollama(model="llama3")
response = llm(f"Answer the question using this context:\n\n{context}\n\nQuestion: {query}")

print("\n🧠 Response:")
print(response)