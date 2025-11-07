import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from config import CHROMA_DIR, EMBED_MODEL, LLM_MODEL

def main():
    if not os.path.exists(CHROMA_DIR):
        print("❌ No Chroma database found.")
        print("💡 Please run `python embed_docs.py` first to create embeddings.")
        return

    print("🔁 Loading existing Chroma database...")
    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        results = vectordb.similarity_search(query, k=3)
        if not results:
            print("⚠️ No relevant documents found.")
            continue

        print("\n🔎 Top matching sources:")
        for r in results:
            src = r.metadata.get("source", "unknown")
            page = r.metadata.get("page", None)
            location = f"{src}, page {page}" if page else src
            print(f"  - {location}")

        context = "\n\n".join([r.page_content for r in results])
        llm = Ollama(model=LLM_MODEL)
        response = llm(
            f"Use the following context to answer the question. Include file references when helpful.\n\n{context}\n\nQuestion: {query}"
        )

        print("\n🧠 Response:")
        print(response)


if __name__ == "__main__":
    main()