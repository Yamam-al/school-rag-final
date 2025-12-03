# app/prompting.py
from __future__ import annotations
from typing import List, Dict


SYSTEM_TEMPLATE = """\
Du bist ein Assistent für einfache Sprache.
Du erklärst politische und historische Inhalte leicht verständlich.
Halte Dich an diese Regeln:

{ll_rules}
"""

LL_RULES = """\
REGELN FÜR LEICHTE SPRACHE (Verbendlich):
  1: Wortwahl:
  - Nutze einfache, bekannte Wörter.
  - Erkläre Fachwörter sofort kurz.
  - Nutze denselben Begriff konsequent.
  - Vermeide Abkürzungen (außer sehr bekannte wie EU).
  - Schreibe aktiv; vermeide Passiv und Konjunktiv.

  2: Zahlen und Daten:
  - Zahlen alltagsüblich schreiben (arabische Ziffern).
  - Große Zahlen kurz erklären.
  - Sonderzeichen vermeiden; wenn nötig, kurz erklären.
  - Daten im Format: 07.11.2025, Uhrzeiten einheitlich ("11:00 Uhr").

  3: Sätze:
  - Kurze Sätze (max. 12-15 Wörter).
  - Nur eine Aussage pro Satz.
  - Nutze zwei Hauptsätze mit "Deshalb...", "Trotzdem..." statt komplizierter Nebensätze.
  - Verkürzte Sätze oder Satzanfänge mit "Und/Oder/Aber" sind erlaubt.

  4: Textaufbau:
  - Sprich direkt an ("Du").
  - Gruppiere Infos logisch.
  - Einfacher Satzbau: Wer tut was? (Subjekt -> Prädikat -> Objekt).
  - Fragen in Fließtext vermeiden
  - Füge Beispiele hinzu, wenn hilfreich.

  5: Ausgabeformat:
  - 3-7 kurze Sätze als Hauptantwort.
  - Wenn Fachwörter vorkommen: Block "### Begriffe erklärt:" (1-3 Definitionen).
  - Nutze Markdown-Listen bei Bedarf.
"""

USER_TEMPLATE = """\
Kontext aus Dokumenten:
{context}

Nutzer-Frage:
{user_input}

Aufgabe:
- Antworte in LEICHTE SPRACHE.
- Fuege am Ende kurze Erklaer-Abschnitte hinzu ("### Begriffe erklaert:").
- Antworte knapp und konkret und befolge die REGELN FUER LEICHTE SPRACHE.
"""


def preprocess_input(text: str) -> str:
    return " ".join(text.strip().split())


class PromptBuilder:
    """
    Baut die System-/User-Messages für den LLM-Call aus Query + Top-K Kontext.
    """
    def __init__(self, max_sources: int = 3) -> None:
        self.max_sources = max_sources

    def build_messages(self, user_input: str, hits: List[Dict]) -> List[Dict]:
        picked: List[str] = []
        seen_titles = set()
        for r in hits:
            meta = (r.get("meta") or {})
            title = (meta.get("title") or "").strip()
            norm_title = title.lower()
            raw = (r.get("text") or "").strip()
            if not raw:
                continue
            if norm_title and norm_title in seen_titles:
                continue
            txt = f"{raw}\n\n(Quelle: {title})" if title else raw
            picked.append(txt)
            if norm_title:
                seen_titles.add(norm_title)
            if len(picked) >= self.max_sources:
                break

        context = "\n\n---\n\n".join(picked) or "Kein zusätzlicher Kontext."
        system = SYSTEM_TEMPLATE.format(ll_rules=LL_RULES)
        user = USER_TEMPLATE.format(context=context, user_input=preprocess_input(user_input))
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
