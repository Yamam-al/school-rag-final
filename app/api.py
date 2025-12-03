# app/api.py
from __future__ import annotations
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import json
import uuid
import asyncio, os, sys, signal
import httpx
import shutil

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .db import KnowledgeBase
from .retrieval import RetrievalService
from .prompting import PromptBuilder
from .llm import LLMAdapter
from .ingest import IngestPipeline
from .core import ApplicationCore


import asyncio, os, shutil
import httpx

_ollama_proc = None

def _native_base_url(url: str) -> str:
    # "http://127.0.0.1:11434/v1" -> "http://127.0.0.1:11434"
    u = (url or "").rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u

async def _ollama_ready(base_url_native: str, retries: int = 30) -> bool:
    # ping native endpoint
    async with httpx.AsyncClient(base_url=base_url_native, timeout=3.0) as c:
        for _ in range(retries):
            try:
                r = await c.get("/api/version")
                if r.status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(1.0)
        return False

async def _ensure_ollama_running():
    global _ollama_proc
    base_url_native = _native_base_url(settings.LLM_BASE_URL)

    # Bereits bereit?
    if await _ollama_ready(base_url_native):
        return

    # (DEV) ollama serve starten – in Prod besser systemd/Docker
    if not shutil.which("ollama"):
        raise RuntimeError("ollama binary not found in PATH")

    env = os.environ.copy()
    env.setdefault("OLLAMA_KEEP_ALIVE", "30m")
    env.setdefault("OLLAMA_NUM_PARALLEL", "2")
    env.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")

    _ollama_proc = await asyncio.create_subprocess_exec(
        "ollama", "serve", "-a", "127.0.0.1:11434",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    ok = await _ollama_ready(base_url_native)
    if not ok:
        try:
            _ollama_proc.terminate()
        except ProcessLookupError:
            pass
        raise RuntimeError("ollama did not become ready at /api/version")

async def _shutdown_ollama():
    global _ollama_proc
    if _ollama_proc is not None:
        try:
            _ollama_proc.terminate()
            await asyncio.wait_for(_ollama_proc.wait(), timeout=5.0)
        except Exception:
            try:
                _ollama_proc.kill()
            except Exception:
                pass
        _ollama_proc = None

# ---------- App & DI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        await llm.startup()
        await llm.warmup()     # nutzt nativ /api/chat + keep_alive=30m
        try:
            _ = retrieval.search("warmup", top_k=1)
        except Exception:
            pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Warmup failed: {e}")
    yield

    # Shutdown
    await llm.shutdown()
    # Nur beenden, wenn wir den Prozess selbst gestartet haben
    await _shutdown_ollama()

app = FastAPI(title="School RAG", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

kb = KnowledgeBase(settings)
retrieval = RetrievalService(kb, settings)
prompting = PromptBuilder()
llm = LLMAdapter(settings)
core = ApplicationCore(retrieval, prompting, llm)
ingest = IngestPipeline(kb, retrieval, settings)


# ---------- Models ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class StreamPayload(BaseModel):
    messages: List[ChatMessage]


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_endpoint(request: Request):
    """
    Unterstützte Content-Types:
      - application/json: { "text": "...", "meta": {...} }
      - multipart/form-data:
            files[]=@doc.pdf
            meta_json={"title":"...", "topics":["..."], "keywords":["..."]}
    Antwort: {"ingested_chunks": N, "saved_files": [...], "failed_files": [...]}
    """
    ct = (request.headers.get("content-type") or "").lower()

    # JSON
    if ct.startswith("application/json"):
        body = await request.json()
        text = (body or {}).get("text")
        meta = (body or {}).get("meta") or {}
        if not text:
            raise HTTPException(400, "text required")
        n = ingest.ingest_text(text, meta)
        return {"ingested_chunks": n, "saved_files": None, "failed_files": None}

    # Multipart
    if ct.startswith("multipart/form-data"):
        form = await request.form()

        # meta_json kann entweder ein STRING (inline JSON) oder ein UploadFile sein.
        raw_meta = form.get("meta_json")
        meta = {}
        if raw_meta:
            try:
                if hasattr(raw_meta, "read"):  # UploadFile
                    # bytes -> str
                    meta_bytes = await raw_meta.read()
                    meta_str = meta_bytes.decode("utf-8", errors="ignore")
                    meta = json.loads(meta_str) if meta_str.strip() else {}
                else:
                    # bereits ein str
                    meta = json.loads(raw_meta) if str(raw_meta).strip() else {}
            except Exception as e:
                raise HTTPException(400, f"Invalid meta_json: {e}")

        files = form.getlist("files")
        saved, failed, total = [], [], 0
        for f in files:
            name = Path(f.filename).name
            if not name:
                continue
            target = Path(settings.DATA_SOURCE_DIR) / f"{uuid.uuid4().hex}_{name}"
            target.write_bytes(await f.read())
            try:
                total += ingest.ingest_file(target, meta)
                saved.append(name)
            except ValueError as e:
                failed.append({"file": name, "error": str(e)})
        return {"ingested_chunks": total, "saved_files": saved or None, "failed_files": failed or None}


    raise HTTPException(415, f"Unsupported Content-Type: {ct}")


@app.post("/stream")
async def stream(payload: StreamPayload, request: Request):
    """
    POST /stream
    Body:
      {
        "messages": [{"role":"user","content":"Was ist der Bundesrat?"}]
      }
    Server-Sent Events:
      - event: message, data: {"delta":"..."}
      - event: ping, data: "ping"
      - event: done, data: {"ok": true}
    """
    async def gen():
        try:
            async for evt in core.stream([m.model_dump() for m in payload.messages]):
                if await request.is_disconnected():
                    break
                yield evt
        except Exception as e:
            from sse_starlette.sse import ServerSentEvent
            yield ServerSentEvent(event="error", data=json.dumps({"type": type(e).__name__, "message": str(e)}))

    return EventSourceResponse(gen(), headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

class EvalRequest(BaseModel):
    question: str
    
@app.post("/eval")
async def eval_once(payload: EvalRequest):
    """
    POST /eval
    Body:
      { "question": "Was bedeutet Asyl in Deutschland?" }

    Antwort:
      {
        "ok": true,
        "error_type": null,
        "error_message": null,
        "answer": "...",
        "metrics": { ... },
        "sources": [ ... ]
      }
    """
    result = await core.answer_once(payload.question)
    return result


