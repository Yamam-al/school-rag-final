# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
import json, uuid, os, asyncio

from .config import settings
from .rag import ingest_text, ingest_file, search
from .llm import chat_stream
from .pipeline import build_messages, postprocess_answer


# Offline-Flags sicherheitshalber auch hier
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DATA_DIR = Path("data/source")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _safe_name(name: str) -> str:
    return Path(name).name


# -------------------------
# Pydantic Models
# -------------------------
class AskPayload(BaseModel):
    query: str

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class StreamPayload(BaseModel):
    messages: List[ChatMessage]


# -------------------------
# App Setup
# -------------------------
app = FastAPI(title="School RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt-Dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# -------------------------
# Ingest Endpoint (mit JSON-freundlichen Beispiel-Blöcken)
# -------------------------
def _example_curl_multipart(filename: str = "example.pdf", meta: dict | None = None) -> dict:
    meta = meta or {"title": "beispiel-dokument"}
    lines = [
        "curl -X POST http://localhost:8080/ingest",
        f"  -F 'files=@/path/to/{filename};type=application/pdf'",
        f"  -F 'meta_json={json.dumps(meta, ensure_ascii=False)}'",
    ]
    one_liner = " ".join([
        "curl -X POST http://localhost:8080/ingest",
        f"-F 'files=@/path/to/{filename};type=application/pdf'",
        f"-F 'meta_json={json.dumps(meta, ensure_ascii=False)}'",
    ])
    return {"lines": lines, "one_liner": one_liner}

def _example_curl_json(meta: dict | None = None) -> dict:
    meta = meta or {"title": "beispiel-text"}
    payload = {"text": "Hier steht der einzufuegende Beispieltext ...", "meta": meta}
    payload_str = json.dumps(payload, ensure_ascii=False)
    lines = [
        "curl -X POST http://localhost:8080/ingest",
        "  -H 'Content-Type: application/json'",
        f"  -d '{payload_str}'",
    ]
    one_liner = " ".join([
        "curl -X POST http://localhost:8080/ingest",
        "-H 'Content-Type: application/json'",
        f"-d '{payload_str}'",
    ])
    return {"lines": lines, "one_liner": one_liner}

def _example_curl_form(meta: dict | None = None) -> dict:
    meta = meta or {"title": "beispiel-text"}
    meta_str = json.dumps(meta, ensure_ascii=False)
    lines = [
        "curl -X POST http://localhost:8080/ingest",
        "  -H 'Content-Type: application/x-www-form-urlencoded'",
        "  --data-urlencode 'text=Hier steht der einzufuegende Beispieltext ...'",
        f"  --data-urlencode 'meta_json={meta_str}'",
    ]
    one_liner = " ".join([
        "curl -X POST http://localhost:8080/ingest",
        "-H 'Content-Type: application/x-www-form-urlencoded'",
        "--data-urlencode 'text=Hier steht der einzufuegende Beispieltext ...'",
        f"--data-urlencode 'meta_json={meta_str}'",
    ])
    return {"lines": lines, "one_liner": one_liner}

def _example_pwsh_json() -> dict:
    lines = [
        "$meta = @{ title='beispiel-text' }",
        "$body = @{ text='Hier steht der einzufuegende Beispieltext ...'; meta=$meta } | ConvertTo-Json -Depth 5",
        "Invoke-WebRequest -Uri 'http://localhost:8080/ingest' -Method Post -ContentType 'application/json' -Body $body",
    ]
    return {"lines": lines}

def _example_pwsh_multipart(filename: str = "example.pdf") -> dict:
    lines = [
        "$file = Get-Item 'C:\\\\path\\\\to\\\\%s'" % filename,
        "$meta = @{ title='beispiel-dokument' } | ConvertTo-Json -Depth 5",
        "$form = @{ files = $file; meta_json = $meta }",
        "Invoke-WebRequest -Uri 'http://localhost:8080/ingest' -Method Post -Form $form",
    ]
    return {"lines": lines}

def _raise_400(message: str, *, mode: str, filename: str | None = None, meta: dict | None = None) -> None:
    # Detail als Objekt, damit Frontend sauber rendern kann
    examples = {}
    if mode == "multipart":
        examples["curl_multipart"] = _example_curl_multipart(filename or "example.pdf", meta)
        examples["powershell_multipart"] = _example_pwsh_multipart(filename or "example.pdf")
    elif mode == "json":
        examples["curl_json"] = _example_curl_json(meta)
        examples["powershell_json"] = _example_pwsh_json()
    elif mode == "form":
        examples["curl_form"] = _example_curl_form(meta)

    raise HTTPException(
        status_code=400,
        detail={"message": message, "examples": examples},
    )

def _require_meta(meta_in: Dict[str, Any], *, require_title: bool, mode: str) -> None:
    """
    Prüft Pflichtfelder im Meta:
    - title: Pflicht bei Text (bei Datei kann rag.ingest_file Dateiname als Fallback nutzen)
    - topics: Pflicht (Liste[str])
    - keywords: Pflicht (Liste[str])
    """
    if require_title and not meta_in.get("title"):
        _raise_400("Metadatum 'title' ist für Text-Ingest erforderlich.", mode=mode)
    if not isinstance(meta_in.get("topics"), list) or not meta_in["topics"]:
        _raise_400("Metadatum 'topics' (Liste von Strings) ist erforderlich.", mode=mode)
    if not isinstance(meta_in.get("keywords"), list) or not meta_in["keywords"]:
        _raise_400("Metadatum 'keywords' (Liste von Strings) ist erforderlich.", mode=mode)

@app.post("/ingest")
async def ingest_flexible(request: Request):
    """
    Frontend muss nur noch 'title' (bei Text) liefern.
    - multipart/form-data: files[] (Titel optional; Fallback = Dateiname), optional text, optional meta_json {"title": "..."}
    - application/json: {"text": "...", "meta": {"title": "..."}}  (Titel Pflicht)
    - x-www-form-urlencoded: text + optional meta_json {"title": "..."}  (Titel Pflicht)
    """
    print("INGEST hit", request.headers.get("content-type"))
    ct = (request.headers.get("content-type") or "").lower()

    ingested_chunks = 0
    files_saved: List[str] = []
    failed_files: List[Dict[str, str]] = []
    text: Optional[str] = None
    meta: Dict[str, Any] = {}

    if ct.startswith("multipart/form-data"):
        mode = "multipart"
        form = await request.form()
        text = form.get("text")
        meta_json = form.get("meta_json")
        meta = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                _raise_400("meta_json ist kein gültiges JSON.", mode=mode)

        # Text: title Pflicht (kein Dateiname-Fallback)
        if text:
            _require_meta(meta, require_title=True, mode=mode)

        uploads = form.getlist("files") if "files" in form else []
        # Dateien: title optional – Fallback passiert in rag.ingest_file

        for up in uploads:
            name = _safe_name(getattr(up, "filename", "unnamed"))
            target = DATA_DIR / name
            if target.exists():
                target = DATA_DIR / f"{target.stem}-{uuid.uuid4().hex[:8]}{target.suffix}"
            content = await up.read()
            target.write_bytes(content)
            files_saved.append(target.name)
            try:
                ingested_chunks += await run_in_threadpool(ingest_file, target, meta)
            except ValueError as e:
                failed_files.append({
                    "file": target.name,
                    "error": str(e),
                    "examples": {
                        "curl_multipart": _example_curl_multipart(filename=target.name, meta=meta),
                        "powershell_multipart": _example_pwsh_multipart(filename=target.name),
                    },
                })

    elif ct.startswith("application/json"):
        mode = "json"
        payload = await request.json()
        text = (payload or {}).get("text")
        meta = (payload or {}).get("meta") or {}
        if not text:
            _raise_400("Für application/json wird ein 'text'-Feld erwartet.", mode=mode)
        _require_meta(meta, require_title=True, mode=mode)

    elif ct.startswith("application/x-www-form-urlencoded"):
        mode = "form"
        form = await request.form()
        text = form.get("text")
        meta_json = form.get("meta_json")
        meta = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except json.JSONDecodeError:
                _raise_400("meta_json ist kein gültiges JSON.", mode=mode)
        if not text:
            _raise_400("Bitte 'text' mitsenden oder multipart/form-data für Dateien nutzen.", mode=mode)
        _require_meta(meta, require_title=True, mode=mode)
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported Content-Type: {ct}")

    if not text and not files_saved:
        _raise_400("Bitte mindestens 'text' oder 'files' mitsenden.", mode="json")

    if text:
        # Falls kein Titel mitgegeben wurde, generieren wir einen
        if "title" not in meta or not str(meta["title"]).strip():
            meta = dict(meta)
            meta["title"] = f"text-{uuid.uuid4().hex[:8]}"
        try:
            ingested_chunks += await run_in_threadpool(ingest_text, text, meta)
        except ValueError as e:
            _raise_400(f"Ingest (Text) abgelehnt: {e}", mode="json")

    return {
        "ingested_chunks": ingested_chunks,
        "saved_files": files_saved or None,
        "failed_files": failed_files or None,
        "debug": {
            "text_preview": (text[:80] + "...") if text else None,
            "meta": meta or None,
            "content_type": ct,
        },
    }


# -------------------------
# /ask Endpoint – einfacher Testpfad (SSE-Ausgabe)
# -------------------------
@app.post("/ask")
async def ask(payload: AskPayload, request: Request):
    hits = search(payload.query, settings.TOP_K)

    if not hits:
        return {"error": "Kein Kontext gefunden. Bitte Lehrer*in darum bitten Dokumente hochzuladen", "status": 412}

    # Neu: LLM-Messages mit LL-Regeln + Kontext aus der Pipeline
    llm_messages = build_messages(payload.query, hits)
    async def gen():
        try:
            async for chunk in chat_stream(llm_messages):
                if await request.is_disconnected():
                    break
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [STREAM_ERROR] {type(e).__name__}: {str(e)}\n\n"

    return EventSourceResponse(gen())


# -------------------------
# /stream Endpoint (für Frontend)
# -------------------------
@app.post("/stream")
async def stream_chat(payload: StreamPayload, request: Request):
    # letzte User-Message holen
    last_user = next((m.content for m in reversed(payload.messages) if m.role == "user"), "").strip()
    if not last_user:
        raise HTTPException(status_code=400, detail="No user message found.")

    # RAG-Kontext
    hits = search(last_user, settings.TOP_K)

    if not hits:
        async def noctx():
            yield ServerSentEvent(event="message", data=json.dumps({"delta": "Kein Kontext gefunden. Bitte Dokumente hochladen."}, ensure_ascii=False))
            yield ServerSentEvent(event="done",    data=json.dumps({"ok": False}))
        return EventSourceResponse(noctx(), headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Neu: LLM-Messages mit LL-Regeln + Kontext aus der Pipeline
    llm_messages = build_messages(last_user, hits)

    async def gen():
        try:
            loop = asyncio.get_event_loop()
            last_ping = loop.time()

            async for chunk in chat_stream(llm_messages):
                if await request.is_disconnected():
                    break

                # Heartbeat alle 10s
                now = loop.time()
                if now - last_ping > 10:
                    yield ServerSentEvent(event="ping", data="ping")
                    last_ping = now

                # LLM-Chunk → delta extrahieren
                delta = None
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content")
                except Exception:
                    pass

                if delta:
                    yield ServerSentEvent(
                        event="message",
                        data=json.dumps({"delta": delta}, ensure_ascii=False)
                    )

            # sauberer Abschluss
            yield ServerSentEvent(event="done", data=json.dumps({"ok": True}))
        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"type": type(e).__name__, "message": str(e)}, ensure_ascii=False)
            )

    return EventSourceResponse(
        gen(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
