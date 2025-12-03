# app/__init__.py
"""
App-Paket für den RAG-Prototyp.
Struktur:
- config.py      : Konfiguration via Pydantic Settings
- db.py          : KnowledgeBase (Chroma + Datei/Text Reader)
- retrieval.py   : RetrievalService (Chunking, Query, Re-Ranking, Heuristiken)
- ingest.py      : IngestPipeline (Text/Datei -> Chunks -> Persist)
- prompting.py   : PromptBuilder (System-/User-Prompt, Regeln)
- llm.py         : LLMAdapter (Streaming)
- core.py        : ApplicationCore (Orchestrierung Query→Prompt→Stream)
- api.py         : FastAPI Endpoints (/health, /ingest, /stream)
"""
