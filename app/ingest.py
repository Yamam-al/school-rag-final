# app/ingest.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import uuid

from .config import settings
from .db import KnowledgeBase
from .retrieval import RetrievalService


@dataclass
class IngestPipeline:
    """
    Einheitliche Ingest-Pipeline für Text/Dateien:
    - Meta-Validierung & -Normalisierung (Titel/Topics/Keywords)
    - Lesen (über KnowledgeBase)
    - Chunking (RetrievalService.chunk)
    - Persist (KnowledgeBase.add_chunks)
    """
    kb: KnowledgeBase
    retrieval: RetrievalService
    _settings: settings.__class__

    # --------- Public API ----------
    def ingest_text(self, text: str, meta: Dict[str, Any]) -> int:
        norm_meta = self._normalize_and_validate_meta(meta, require_title=True)
        chunks = self.retrieval.chunk(text)
        if not chunks:
            return 0

        ids = [uuid.uuid4().hex for _ in chunks]
        metas = [norm_meta for _ in chunks]
        self.kb.add_chunks(chunks, metas, ids)
        return len(chunks)

    def ingest_file(self, path: Path, meta: Dict[str, Any] | None) -> int:
        text = self.kb.read_text(path)
        norm_meta = self._normalize_and_validate_meta(meta or {}, require_title=False, fallback_title=path.name)
        chunks = self.retrieval.chunk(text)
        if not chunks:
            return 0
        ids = [uuid.uuid4().hex for _ in chunks]
        metas = [norm_meta for _ in chunks]
        self.kb.add_chunks(chunks, metas, ids)
        return len(chunks)

    # --------- Helpers ----------
    def _normalize_and_validate_meta(
        self,
        meta: Dict[str, Any],
        require_title: bool,
        fallback_title: str | None = None,
    ) -> Dict[str, Any]:
        """
        Erwartete Felder:
        - title: str (bei Text Pflicht, bei Datei optional)
        - topics: List[str] (mind. 1 empfohlen)
        - keywords: List[str] (mind. 1 empfohlen)
        """
        m = dict(meta or {})
        title = (m.get("title") or "").strip()
        if not title:
            if require_title and not fallback_title:
                raise ValueError("meta.title is required for text ingest")
            title = fallback_title or "unbenannt"

        def _norm_list(x: Any) -> List[str]:
            if not x:
                return []
            if isinstance(x, str):
                return [x.strip()] if x.strip() else []
            if isinstance(x, list):
                return [str(e).strip() for e in x if str(e).strip()]
            return []

        topics = _norm_list(m.get("topics"))
        keywords = _norm_list(m.get("keywords"))

        return {
            "title": title,
            "topics": topics,
            "keywords": keywords,
        }
