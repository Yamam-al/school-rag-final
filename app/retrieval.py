# app/retrieval.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple
import math
import re
import numpy as np

from .config import settings
from .db import KnowledgeBase


_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9\-]+")


def _word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Einfache Wort-basiertes Sliding-Window-Chunking.
    Konsistenter Chunker für Ingest & ggf. spätere Re-Chunks.
    """
    words = _word_tokens(text)
    if not words:
        return []
    out: List[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        win = words[i:i + size]
        if not win:
            break
        out.append(" ".join(win))
        if i + size >= len(words):
            break
    return out


# app/retrieval.py
def _listify(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, str):
        # trenne an Kommas oder Whitespace
        return [t.strip() for t in re.split(r"[,\s]+", x) if t.strip()]
    if isinstance(x, (list, tuple, set)):
        return [str(t).strip() for t in x if str(t).strip()]
    return [str(x).strip()]

def _meta_boost(meta: Dict[str, Any], q_tokens: set[str]) -> float:
    """
    Leichter Boost, wenn Query-Tokens in Topics/Keywords vorkommen.
    Funktioniert für String (kommagetrennt) und Listen.
    """
    topics = _listify((meta or {}).get("topics"))
    keywords = _listify((meta or {}).get("keywords"))
    bag = " ".join([*topics, *keywords]).lower()
    if not bag:
        return 0.0

    b = 0.0
    if any(t in bag for t in q_tokens):
        b += 0.22
        if len(q_tokens) >= 2:
            b += 0.10
        if len(q_tokens) >= 3:
            b += 0.12
    return min(b, 0.44)


@dataclass
class RetrievalService:
    """
    Retrieval-Layer:
    - String-Chunking
    - Query gegen Chroma
    - Einfaches Re-Ranking (Cosine via normalisierte Embeddings + Meta-Boost)
    - Duplikatreduktion nach Titel
    """
    kb: KnowledgeBase
    _settings: settings.__class__

    # ---------- Public API ----------
    def chunk(self, text: str) -> List[str]:
        return chunk_text(text, self._settings.CHUNK_SIZE, self._settings.CHUNK_OVERLAP)

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        top_k = top_k or self._settings.TOP_K
        raw = self.kb.query(query, n=max(12, top_k * 3))

        rows: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(raw.get("documents", [[]])[0],
                                   raw.get("metadatas", [[]])[0],
                                   raw.get("distances", [[]])[0]):
            rows.append({"text": doc, "meta": meta or {}, "score": 1.0 - float(dist)})

        if not rows:
            return []

        # Query-Vector über denselben Embedder wie Chroma; benutze eingebettete Funktion via Dummy Query
        # (Chroma liefert keine Vektoren direkt → approximieren wir Ranking mit Scores + Meta-Boost)
        q_tokens = set(_word_tokens(query))
        ranked: List[Dict[str, Any]] = []
        for r in rows:
            combo = 0.70 * r["score"] + 0.30 * _meta_boost(r["meta"], q_tokens)
            r["combo_score"] = combo
            ranked.append(r)
        ranked.sort(key=lambda x: x["combo_score"], reverse=True)

        # Fokus & Dedupe nach Titel
        out: List[Dict[str, Any]] = []
        seen_titles = set()
        for r in ranked:
            title = (r.get("meta") or {}).get("title", "").strip().lower()
            if title and title in seen_titles:
                continue
            out.append(r)
            if title:
                seen_titles.add(title)
            if len(out) >= top_k:
                break
        return out
