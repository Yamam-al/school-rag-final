# app/llm.py
from __future__ import annotations
from typing import AsyncGenerator, List, Dict, Optional
import httpx
from .config import settings

class LLMAdapter:
    def __init__(self, _settings: settings.__class__) -> None:
        self._settings = _settings
        self.client: Optional[httpx.AsyncClient] = None
    
    
    def _native_url(self, path: str) -> str:
        """
        Baut eine native Ollama-URL (ohne /v1) aus settings.LLM_BASE_URL.
        Beispiel:
          - LLM_BASE_URL = http://127.0.0.1:11434/v1  -> http://127.0.0.1:11434/api/chat
          - LLM_BASE_URL = http://127.0.0.1:11434     -> http://127.0.0.1:11434/api/chat
        """
        base = self._settings.LLM_BASE_URL.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]  # "/v1" abschneiden
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    async def startup(self) -> None:
         if self.client is None:
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
            # Wichtig: base_url kann /v1 enthalten
            self.client = httpx.AsyncClient(
                base_url=self._settings.LLM_BASE_URL.rstrip("/"),
                timeout=httpx.Timeout(120.0),
                http2=True,
                limits=limits,
                headers={
                    "Authorization": f"Bearer {self._settings.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def warmup(self) -> None:
        """
        Lädt das Modell über die native Ollama-API vor und setzt keep_alive=30m.
        Unabhängig davon, ob LLM_BASE_URL mit /v1 konfiguriert ist.
        """
        await self.startup()
        assert self.client is not None

        payload = {
            "model": self._settings.LLM_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "keep_alive": "30m",
        }
        # absoluter, nativer Pfad – NICHT /v1
        url = self._native_url("/api/chat")
        r = await self.client.post(url, json=payload)
        r.raise_for_status()

    async def stream(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 600
    ) -> AsyncGenerator[str, None]:
        await self.startup()
        assert self.client
        payload = {
            "model": self._settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "keep_alive": "30m",  # auch bei echten Calls
        }
        async with self.client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line.removeprefix("data: ").strip()
                if chunk == "[DONE]":
                    break
                yield chunk
