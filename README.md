# 🧠 RAG Local — A Minimal Local Retrieval-Augmented Generation Demo

This project demonstrates a **fully local Retrieval-Augmented Generation (RAG)** pipeline using Python, [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), and [Ollama](https://ollama.com/).

The goal is to **learn RAG end-to-end** — from document loading and embedding to retrieval and LLM-based question answering — with everything running **locally** on a Mac.

---

## 🚀 Features

- 📄 Loads all documents (`.pdf`, `.txt`, `.md`) from a local `docs/` folder  
- 🧩 Splits documents into text chunks for efficient search  
- 🔍 Stores embeddings in a local [ChromaDB](https://www.trychroma.com/) vector database  
- 🤖 Uses [Ollama](https://ollama.com/) for both **LLM** and **embedding models** (no API key required)  
- 💬 Lets you ask natural-language questions and get answers based on your own documents  
- 🗂️ Everything runs locally — ideal for learning or offline experimentation  

---

## 🧰 Requirements

- macOS or Linux  
- Python 3.10+  
- [Ollama](https://ollama.com/download) installed and running locally  

---

## 📦 Installation

```bash
# 1. Clone this repository
git clone https://github.com/russ-hendy/rag-local.git
cd rag-local

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # (on macOS/Linux)

# 3. Install dependencies
pip install -r requirements.txt
````

---

## 🦙 Set Up Ollama

Install Ollama if you haven’t already:

👉 [https://ollama.com/download](https://ollama.com/download)

Then pull the models used in this project:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

You can verify it works:

```bash
ollama run llama3
```

---

## 📂 Add Some Documents

Put your `.pdf`, `.txt`, or `.md` files in the `docs/` folder, e.g.:

```
rag-local/
  ├── docs/
  │     ├── article1.pdf
  │     ├── notes.txt
  ├── rag.py
  ├── requirements.txt
  └── README.md
```

You can use any text-rich sources (books, reports, articles, etc.).

---

## ▶️ Run the RAG Script

```bash
python rag.py
```

You’ll see:

```
✅ Loaded 4 documents from docs
🧩 Split into 172 chunks

💬 Ask a question: 
```

Type your question, for example:

```
💬 Ask a question: What are the main themes discussed in these documents?
```

Then watch your local LLM respond based on your retrieved document context.

---

## 🧱 Project Structure

```
rag-local/
│
├── docs/                 # Local documents you’ll embed & search
├── chroma_db/            # Local persisted vector database (auto-created)
├── rag.py                # Main RAG pipeline script
├── requirements.txt      # Python dependencies (pinned)
└── README.md             # This file
```

---

## 🧩 How It Works

1. **Load Documents** → All files in `docs/` are loaded via `PyPDFLoader` or `TextLoader`.
2. **Chunking** → Text is split into small overlapping chunks for better retrieval.
3. **Embedding** → Each chunk is embedded into a high-dimensional vector using `nomic-embed-text`.
4. **Vector Store** → Chunks + embeddings are saved locally in `ChromaDB`.
5. **Query** → When you ask a question, the most similar chunks are retrieved.
6. **LLM Generation** → `llama3` uses the retrieved context to generate a grounded answer.

---

## ⚡ Optional: Use OpenAI Instead of Ollama

If you’d rather test using an OpenAI model for inference:

```bash
pip install langchain-openai openai
export OPENAI_API_KEY="sk-..."
```

Then replace in `rag.py`:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")
```

---

## 🧠 Next Steps

* Add metadata (source filenames) to retrieved chunks
* Build a Streamlit UI for interactive RAG
* Try hybrid search (keyword + vector)
* Experiment with different chunk sizes or embeddings
* Compare Ollama vs OpenAI performance

---

## 🪪 License

MIT License © 2025 Russ Hendy
This project is for educational and experimental use.

---
