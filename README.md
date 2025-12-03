# School RAG - Starter Guide

A FastAPI backend for a **local Retrieval-Augmented Generation (RAG)** system using **Ollama + Phi-4** and **local embeddings**.  
The backend provides ingestion, retrieval, and streaming LLM responses.

---

## 1) Project Layout

```
.
├─ main.py                 # bootstrap only
└─ app/
   ├─ __init__.py
   ├─ api.py               # FastAPI endpoints: /health, /ingest, /stream
   ├─ core.py              # ApplicationCore (runtime orchestration)
   ├─ config.py            # Pydantic Settings (.env)
   ├─ db.py                # KnowledgeBase (Chroma + file reading)
   ├─ ingest.py            # IngestPipeline (text/file -> chunks -> persist)
   ├─ llm.py               # LLMAdapter
   ├─ prompting.py         # PromptBuilder (system/user messages)
   └─ retrieval.py         # RetrievalService (chunking, query, ranking)
```

---

## 2) Prerequisites

- Python 3.13 
- **Ollama** installed and running  
  Download: https://ollama.com/download
- **Phi-4 model** for Ollama (downloade once on first run) (see "4) Install & Run")
- A local **SentenceTransformers embedding snapshot** including `config.json`  
  multilingual E5 model stored in `./models/...` (see "4) Install & Run")

> ChromaDB data is stored in `data/chroma`.  
> Uploaded files are stored in `data/source` with UUID-based filenames.

---

## 3) Configuration (.env)

Adjust settings in the `.env` file if needed:

```ini
# LLM
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=local
LLM_MODEL=phi4:latest

# Embeddings – local snapshot folder path (must contain config.json)
EMBEDDING_MODEL=./models/intfloat--multilingual-e5-base

# Chroma (lokal, persistenter Pfad)
CHROMA_PATH=./data/chroma

# RAG
CHUNK_SIZE=900
CHUNK_OVERLAP=120
TOP_K=4

```

**Important:** The path in `EMBEDDING_MODEL` must exist and include a `config.json` of your local SentenceTransformers snapshot.

---

## 4) Install & Run

### 4.1 Install Ollama + Phi-4 (first use)
1) Install Ollama (see https://ollama.com/download)
2) Download and start phi4 model : 
```bash
ollama run phi4
```

This automatically downloads the model and starts the local server at:
http://localhost:11434.

### 4.2 Install Virtual Environment
```bash
py -3.13 -m venv .venv
```

### 4.3 Install backend dependencies
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```
### 4.4 Install Embedding Model in models/intfloat--multilingual-e5-base
this will take some time, since the model is over 2 GB.
```bash
git clone https://huggingface.co/intfloat/multilingual-e5-base models/intfloat--multilingual-e5-base

```
---

## 5) Start API
```bash
# switch to virtual envirenment, if not already activated
.venv\Scripts\activate 
# run backend
uvicorn main:app --host 0.0.0.0 --port 8080 --http h11 --workers 1 --access-log
```

Open the docs in your browser:  
- Swagger UI: `http://localhost:8080/docs`

**Note:**  
> The recommended way to interact with the chatbot is through the **frontend application**.  
> Health checks and document ingestion are **not yet integrated** in the frontend and must currently be tested manually using the curl commands shown below.

---

## 6) Frontend
The Frontend can be found here: https://github.com/Yamam-al/chatbot-frontend-final

## 7) API Overview

### 7.1 `GET /health`

Simple health check.

```bash
curl.exe -s http://localhost:8080/health
# => {"status":"ok"}
```

---

## 7.2 Ingesting Content

The backend supports ingestion of **plain text** and **files (PDF/TXT/MD)** into the local Chroma knowledge base.  
Each document is chunked, embedded, and persisted.

### **Recommended: Python CLI (`ingest_cli.py`)**

The CLI works on Windows, macOS, and Linux without quoting or shell issues.

---

### a) Ingest **plain text**

```bash
python ingest_cli.py   --text "Der Bundesrat ist ein Verfassungsorgan in Deutschland. Er wirkt bei Gesetzen mit."   --title "Bundesrat Basics"   --topics politik deutschland   --keywords bundesrat gesetzgebung bundesländer
```

---

### b) Ingest a **file** (PDF, TXT, MD)

```bash
python ingest_cli.py   --file "C:/Users/alsho/Downloads/IzPB_347_Weimarer-Republik_barrierefrei-1.pdf"   --title "Weimarer Republik (barrier-free)"   --topics geschichte   --keywords "weimarer republik" verfassung demokratie
```

**Example Response:**

```json
{
  "ingested_chunks": 7,
  "saved_files": ["IzPB_347_Weimarer-Republik_barrierefrei-1.pdf"],
  "failed_files": null
}
```

**Notes**
- Supported file types: `.pdf`, `.txt`, `.md`
- Rich-text formats such as `.docx`, `.odt`, `.html` are **not supported**
- PDF extraction uses **PyMuPDF**, with fallback to **pypdf**
- If `meta.title` is omitted, the backend uses the filename
- Empty or unreadable files result in `ingested_chunks = 0`

---

### 7.2.a. Optional: Using curl manually (less recommended)

Due to differences in quoting across shells, curl requests can break—especially on Windows.  
The Python CLI is the preferred method.

---

### a) Plain text ingest
```bash
curl -X POST "http://localhost:8080/ingest" \
  -H "Content-Type: application/json" \
  -d '{"text":"Der Bundesrat ist ein Verfassungsorgan in Deutschland. Er wirkt bei Gesetzen mit.","meta":{"title":"bpb: Bundesrat Basics","topics":["politik","deutschland"],"keywords":["bundesrat","gesetzgebung","bundesländer"]}}'
# => {"ingested_chunks": 1, "saved_files": null, "failed_files": null}
```

#### b) Multipart (PDF, TXT, MD)

```powershell
curl -s -X POST "http://localhost:8080/ingest" `
  -H "Accept: application/json" `
  -F 'files=@C:\Users\alsho\Downloads\Weimarer_Republik_klexion.pdf' `
  -F 'meta_json={"title":"Weimarer Republik Klexikon","topics":["geschichte"],"keywords":["weimarer republik","verfassung","demokratie"]}'
# => {"ingested_chunks": 6, "saved_files": ["Weimarer_Republik_klexion.pdf"], "failed_files": null}
```

Response shape:
```json
{
  "ingested_chunks": 7,
  "saved_files": ["Weimarer_Republik_klexion.pdf"],
  "failed_files": null
}
```

**Notes**
- Supported file types: `.pdf`, `.txt`, `.md`
- Rich-text formats such as `.docx`, `.odt`, `.html` are **not supported**
- PDF extraction uses **PyMuPDF**, with fallback to **pypdf**
- If `meta.title` is omitted, the backend uses the filename
- Empty or unreadable files result in `ingested_chunks = 0`

---

### 8.3 `POST /stream` (Server‑Sent Events)

Due to differences in quoting across shells, curl requests can break—especially on Windows.  
The Frontend is the preferred method.

### 8.3.a. Optional: Using curl manually (less recommended)

Sends a chat **message list** and receives an SSE stream with incremental tokens.
- `event: message` with `{"delta":"..."}`
- `event: ping` heartbeat every ~10 seconds
- `event: done` when generation ends

**Minimal example (curl)**

```bash
curl -N -X POST "http://localhost:8080/stream" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Was ist der Bundesrat?"}]}'
```

---

## 8) Security & CORS

- CORS in `api.py` allows `http://localhost:3000` by default.  
  Adjust `allow_origins` for your environment.

