# app/core.py
from __future__ import annotations
from typing import AsyncGenerator, List, Dict, Optional
import asyncio
import json
import logging
import time
from sse_starlette.sse import ServerSentEvent
import unicodedata


from .config import settings
from .retrieval import RetrievalService
from .prompting import PromptBuilder
from .llm import LLMAdapter

log = logging.getLogger("metrics")

class ApplicationCore:
    """
    Orchestrierung: User-Message -> Retrieval -> Prompt -> LLM-Stream (SSE).
    Hinweis: LLM-Warmup/Keep-Alive passiert im LLMAdapter und beim App-Startup (lifespan).
    """

    def __init__(self, retrieval: RetrievalService, prompting: PromptBuilder, llm: LLMAdapter) -> None:
        self.retrieval = retrieval
        self.prompting = prompting
        self.llm = llm
        self._ping_interval = getattr(settings, "PING_INTERVAL_SECONDS", 10)

    async def stream(self, messages: List[Dict]) -> AsyncGenerator[ServerSentEvent, None]:
        """
        Sendet SSE-Events:
          - event=message: {"delta": "..."}
          - event=ping: "ping" (alle ~PING_INTERVAL_SECONDS)
          - event=metrics: {...}   # NEU
          - event=done: {"ok": true/false}
          - event=error: {"type": "...", "message": "..."}
        """
        # -------- Messung: Start --------
        t0 = time.perf_counter()
        sizes_emitted = 0
        chunks_emitted = 0
        t_retrieval_start = None
        t_retrieval_end = None
        t_prompt_start = None
        t_prompt_end = None
        t_llm_req = None
        t_first_token = None
        t_last_token = None
        # --------------------------------

        # 1) letzte User-Nachricht finden
        last_user: Optional[str] = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            ""
        )
        last_user = (last_user or "").strip()
        if not last_user:
            yield ServerSentEvent(event="error", data=json.dumps({"message": "No user message."}))
            # metrics bei Fehler
            metrics = self._build_metrics(
                t0, t_retrieval_start, t_retrieval_end, t_prompt_start, t_prompt_end,
                t_llm_req, t_first_token, t_last_token, sizes_emitted, chunks_emitted, ok=False
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="metrics", data=json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="done", data=json.dumps({"ok": False}))
            return

        # 2) Retrieval
        try:
            t_retrieval_start = time.perf_counter()
            hits = self.retrieval.search(last_user, settings.TOP_K)
            t_retrieval_end = time.perf_counter()
        except Exception as e:
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"type": "retrieval_error", "message": str(e)})
            )
            metrics = self._build_metrics(
                t0, t_retrieval_start, t_retrieval_end, t_prompt_start, t_prompt_end,
                t_llm_req, t_first_token, t_last_token, sizes_emitted, chunks_emitted, ok=False
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="metrics", data=json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="done", data=json.dumps({"ok": False}))
            return

        if not hits:
            yield ServerSentEvent(
                event="message",
                data=json.dumps({"delta": "Kein Kontext gefunden. Bitte Dokumente hochladen."}, ensure_ascii=False)
            )
            metrics = self._build_metrics(
                t0, t_retrieval_start, t_retrieval_end, t_prompt_start, t_prompt_end,
                t_llm_req, t_first_token, t_last_token, sizes_emitted, chunks_emitted, ok=False
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="metrics", data=json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="done", data=json.dumps({"ok": False}))
            return

        # 3) Prompt bauen
        t_prompt_start = time.perf_counter()
        llm_messages = self.prompting.build_messages(last_user, hits)
        t_prompt_end = time.perf_counter()

        # 4) Stream vom LLM weiterreichen
        loop = asyncio.get_event_loop()
        last_ping_ts = loop.time()
        t_llm_req = time.perf_counter()

        try:
            async for chunk in self.llm.stream(llm_messages):
                now = loop.time()
                if now - last_ping_ts > self._ping_interval:
                    yield ServerSentEvent(event="ping", data="ping")
                    last_ping_ts = now

                # First-token timestamp
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                # robustes Parsing des Stream-Chunks
                delta_text = self._extract_delta_text(chunk)
                if delta_text:
                    sizes_emitted += len(delta_text)
                    chunks_emitted += 1
                    yield ServerSentEvent(
                        event="message",
                        data=json.dumps({"delta": delta_text}, ensure_ascii=False)
                    )
                t_last_token = time.perf_counter()
        except Exception as e:
            # Fehler im LLM-Stream
            yield ServerSentEvent(
                event="error",
                data=json.dumps({"type": "llm_stream_error", "message": str(e)})
            )
            metrics = self._build_metrics(
                t0, t_retrieval_start, t_retrieval_end, t_prompt_start, t_prompt_end,
                t_llm_req, t_first_token, t_last_token, sizes_emitted, chunks_emitted, ok=False
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="metrics", data=json.dumps(metrics, ensure_ascii=False))
            yield ServerSentEvent(event="done", data=json.dumps({"ok": False}))
            return

        # 5) fertig
        sources = []
        seen = set()
        for hit in hits:
            meta = hit.get("meta") or {}
            raw_title = meta.get("title") or ""
            title = unicodedata.normalize("NFKC", raw_title).strip().lower()
            title_clean = " ".join(title.split())  # entfernt doppelte/geschützte Leerzeichen
            if title_clean not in seen:
                sources.append(title)
                seen.add(title.lower())

        if sources:
            md_sources = "\n\n### Quellen:\n" + "\n".join(f"- **{t}**" for t in sources)
            yield ServerSentEvent(
                event="message",
                data=json.dumps({"delta": md_sources}, ensure_ascii=False)
            )
        metrics = self._build_metrics(
            t0, t_retrieval_start, t_retrieval_end, t_prompt_start, t_prompt_end,
            t_llm_req, t_first_token, t_last_token, sizes_emitted, chunks_emitted, ok=True
        )
        log.info(json.dumps(metrics, ensure_ascii=False))
        yield ServerSentEvent(event="metrics", data=json.dumps(metrics, ensure_ascii=False))
        yield ServerSentEvent(event="done", data=json.dumps({"ok": True}))

    async def answer_once(self, user_input: str) -> Dict:
        """
        Nicht-streamende Variante von stream():
        - nimmt eine Frage entgegen
        - führt den gleichen RAG Prozess aus
        - gibt komplette Antwort, Metriken und Quellen als Dict zurück
        """
        # -------- Messung: Start --------
        t0 = time.perf_counter()
        sizes_emitted = 0
        chunks_emitted = 0
        t_retrieval_start = None
        t_retrieval_end = None
        t_prompt_start = None
        t_prompt_end = None
        t_llm_req = None
        t_first_token = None
        t_last_token = None
        # --------------------------------

        last_user = (user_input or "").strip()
        if not last_user:
            metrics = self._build_metrics(
                t0,
                t_retrieval_start,
                t_retrieval_end,
                t_prompt_start,
                t_prompt_end,
                t_llm_req,
                t_first_token,
                t_last_token,
                sizes_emitted,
                chunks_emitted,
                ok=False,
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            return {
                "ok": False,
                "error_type": "invalid_input",
                "error_message": "Empty user input.",
                "answer": "",
                "metrics": metrics,
                "sources": [],
            }

        # 2) Retrieval (wie in stream)
        try:
            t_retrieval_start = time.perf_counter()
            hits = self.retrieval.search(last_user, settings.TOP_K)
            t_retrieval_end = time.perf_counter()
        except Exception as e:
            metrics = self._build_metrics(
                t0,
                t_retrieval_start,
                t_retrieval_end,
                t_prompt_start,
                t_prompt_end,
                t_llm_req,
                t_first_token,
                t_last_token,
                sizes_emitted,
                chunks_emitted,
                ok=False,
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            return {
                "ok": False,
                "error_type": "retrieval_error",
                "error_message": str(e),
                "answer": "",
                "metrics": metrics,
                "sources": [],
            }

        if not hits:
            metrics = self._build_metrics(
                t0,
                t_retrieval_start,
                t_retrieval_end,
                t_prompt_start,
                t_prompt_end,
                t_llm_req,
                t_first_token,
                t_last_token,
                sizes_emitted,
                chunks_emitted,
                ok=False,
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            return {
                "ok": False,
                "error_type": "no_context",
                "error_message": "Kein Kontext gefunden. Bitte Dokumente hochladen.",
                "answer": "Kein Kontext gefunden. Bitte Dokumente hochladen.",
                "metrics": metrics,
                "sources": [],
            }

        # 3) Prompt bauen
        try:
            t_prompt_start = time.perf_counter()
            llm_messages = self.prompting.build_messages(last_user, hits)
            t_prompt_end = time.perf_counter()
        except Exception as e:
            metrics = self._build_metrics(
                t0,
                t_retrieval_start,
                t_retrieval_end,
                t_prompt_start,
                t_prompt_end,
                t_llm_req,
                t_first_token,
                t_last_token,
                sizes_emitted,
                chunks_emitted,
                ok=False,
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            return {
                "ok": False,
                "error_type": "prompt_error",
                "error_message": str(e),
                "answer": "",
                "metrics": metrics,
                "sources": [],
            }

        # 4) LLM streamen, intern sammeln (wie in stream)
        answer_parts: List[str] = []
        t_llm_req = time.perf_counter()
        try:
            async for chunk in self.llm.stream(llm_messages):
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                delta_text = self._extract_delta_text(chunk)
                if delta_text:
                    sizes_emitted += len(delta_text)
                    chunks_emitted += 1
                    answer_parts.append(delta_text)
                t_last_token = time.perf_counter()
        except Exception as e:
            metrics = self._build_metrics(
                t0,
                t_retrieval_start,
                t_retrieval_end,
                t_prompt_start,
                t_prompt_end,
                t_llm_req,
                t_first_token,
                t_last_token,
                sizes_emitted,
                chunks_emitted,
                ok=False,
            )
            log.info(json.dumps(metrics, ensure_ascii=False))
            return {
                "ok": False,
                "error_type": "llm_stream_error",
                "error_message": str(e),
                "answer": "",
                "metrics": metrics,
                "sources": [],
            }

        answer = "".join(answer_parts)

        # 5) Quellen genauso wie in stream() bauen
        sources: List[str] = []
        seen = set()
        for hit in hits:
            # in stream() benutzt du hit.get("meta")
            meta = hit.get("meta") or {}
            raw_title = meta.get("title") or meta.get("source") or meta.get("file_name") or ""
            if not raw_title:
                continue
            title_norm = unicodedata.normalize("NFKC", raw_title).strip()
            key = " ".join(title_norm.lower().split())
            if key in seen:
                continue
            seen.add(key)
            sources.append(title_norm)

        # 6) Metriken wie gehabt
        metrics = self._build_metrics(
            t0,
            t_retrieval_start,
            t_retrieval_end,
            t_prompt_start,
            t_prompt_end,
            t_llm_req,
            t_first_token,
            t_last_token,
            sizes_emitted,
            chunks_emitted,
            ok=True,
        )
        log.info(json.dumps(metrics, ensure_ascii=False))

        return {
            "ok": True,
            "error_type": None,
            "error_message": None,
            "answer": answer,
            "metrics": metrics,
            "sources": sources,
        }
    
    
    @staticmethod
    def _extract_delta_text(chunk: str) -> Optional[str]:
        """
        Erwartet 'chunk' als JSON-Zeile aus dem OpenAI-Stream (data: {...}).
        Extrahiert choices[0].delta.content, wenn vorhanden.
        """
        try:
            obj = json.loads(chunk)
        except Exception:
            return None

        try:
            choices = obj.get("choices") or []
            if not choices:
                return None
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            # optional: falls Anbieter statt delta.content direkt "content" streamt
            if text is None:
                text = obj.get("content")
            return text
        except Exception:
            return None

    @staticmethod
    def _ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return round((b - a) * 1000.0, 2)

    def _build_metrics(
        self,
        t0: float,
        t_r0: Optional[float],
        t_r1: Optional[float],
        t_p0: Optional[float],
        t_p1: Optional[float],
        t_llm_req: Optional[float],
        t_first: Optional[float],
        t_last: Optional[float],
        emitted_chars: int,
        chunks: int,
        ok: bool
    ) -> Dict:
        return {
            "durations_ms": {
                "retrieval": self._ms(t_r0, t_r1),
                "prompt_build": self._ms(t_p0, t_p1),
                "llm_time_to_first_token": self._ms(t_llm_req, t_first),
                "llm_stream_duration": self._ms(t_first, t_last) if t_first and t_last else None,
                "total": self._ms(t0, t_last or t_p1 or t_r1),
            },
            "sizes": {"emitted_chars": emitted_chars, "chunks": chunks},
            "ok": ok,
        }
