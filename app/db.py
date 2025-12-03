# app/db.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import time
import fitz  # PyMuPDF
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from .config import settings


class LocalEmbeddingFn(EmbeddingFunction):
    def __init__(self, model_path: str) -> None:
        import math
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_path)

        # ---- robust max_seq_length ----
        # Viele ST-Modelle (z.B. XLM-RoBERTa) haben effektiv ~512 Token.
        # Wir lesen mögliche Quellen aus und klemmen konservativ auf 512.
        maxs = []
        try:
            if hasattr(self.model, "max_seq_length") and isinstance(self.model.max_seq_length, int):
                maxs.append(self.model.max_seq_length)
        except Exception:
            pass
        try:
            tok = getattr(self.model, "tokenizer", None)
            mlen = getattr(tok, "model_max_length", None)
            if isinstance(mlen, int) and mlen > 0 and mlen < 10_000:
                maxs.append(mlen)
        except Exception:
            pass

        # Fallback: 512
        resolved = min([v for v in maxs if isinstance(v, int) and v > 0], default=512)
        resolved = min(resolved, 512)  # konservativ – 512 ist für XLM-R sicher
        self.model.max_seq_length = resolved
        # --------------------------------

        self.model.eval()
        self._loaded_sec = time.perf_counter() - t0

    def __call__(self, texts: Documents) -> List[List[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


@dataclass
class KnowledgeBase:
    """
    Kapselt Chroma-Verbindung und Ressourcen-Lesen.
    Speichert keine Retrieval- oder Rankinglogik—nur Persist/Query.
    """
    _settings: settings.__class__

    def __post_init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=self._settings.CHROMA_PATH,
            settings=ChromaSettings(allow_reset=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="school_rag",
            embedding_function=LocalEmbeddingFn(self._settings.EMBEDDING_MODEL),
            metadata={"hnsw:space": "cosine"},
        )

    # -------- I/O (Dateien/Strings) ----------
    def read_text(self, path: Path) -> str:
        """
        Liest Text aus verschiedenen Formaten. Unterstützt aktuell: .txt, .md, .pdf
        """
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            # robust: erst PyMuPDF, Fallback pypdf
            try:
                doc = fitz.open(path)
                text = "\n".join(page.get_text("text") for page in doc)
                doc.close()
                if text.strip():
                    return text
            except Exception:
                pass
            # Fallback
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        raise ValueError(f"Unsupported file type: {suffix}")

    # -------- Persist/Query ----------
    # app/db.py
    def add_chunks(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Chroma erlaubt in metadata nur Skalarwerte (str, int, float, bool, None).
        Wir wandeln Listen/Tuples/Sets in kommagetrennte Strings um.
        Andere komplexe Typen -> str(value).
        """
        def _sanitize(m: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in (m or {}).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    out[k] = v
                elif isinstance(v, (list, tuple, set)):
                    out[k] = ", ".join(str(x) for x in v if str(x).strip())
                else:
                    out[k] = str(v)
            return out

        safe_metas = [_sanitize(m) for m in metadatas]
        self.collection.add(documents=documents, metadatas=safe_metas, ids=ids)


    def query(self, query: str, n: int = 8) -> Dict[str, Any]:
        return self.collection.query(query_texts=[query], n_results=n)
