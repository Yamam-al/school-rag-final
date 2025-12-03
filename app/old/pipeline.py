# app/pipeline.py
from typing import List, Dict

LL_RULES = """\
REGELN FÜR LEICHTE SPRACHE:
1) Wortwahl:
- Nutze einfache, bekannte Wörter.
- Erkläre Fachwörter sofort kurz.
- Nutze denselben Begriff konsequent.
- Vermeide Abkürzungen (außer sehr bekannte wie WC, EU).
- Schreibe mit Verben und aktiv; vermeide Passiv.
- Vermeide „würde/könnte“; schreibe klar.
- Keine Redewendungen.
- Schreibe positiv. Bei Verneinungen markiere das Negationswort fett (**nicht**, **nie**).

2) Zahlen und Daten:
- Nutze Ziffern (1, 2, 3).
- Große Zahlen verständlich: grobe Angabe + (exakt) optional.
- Sehr alte Jahreszahlen darfst du umschreiben.
- Datum/Uhrzeit klar: „3. März 2024“, „11:00 Uhr“.
- Vermeide Sonderzeichen oder erkläre sie kurz.

3) Satzbau:
- Kurze Sätze (ca. 5–15 Wörter), eine Aussage pro Satz.
- Einfacher Aufbau: Subjekt → Verb → Objekt.
- Vermeide Nebensätze; mache daraus zwei Hauptsätze.
- Vermeide „weil/obwohl/damit“; nutze „Deshalb…/Trotzdem…“.
- „Und/Oder/Aber“ am Satzanfang ist erlaubt.

4) Textaufbau:
- Sprich die Lesenden direkt an („Sie“ oder „Du“).
- Keine Fragen im Fließtext; Fragen nur als Überschrift.
- Bündle zusammengehörige Infos; vermeide „siehe oben“.
- Füge Beispiele/Erklärungen hinzu, wenn es hilft.
- Inhalt/Sinn darf sich **nicht** ändern.

5) Ausgabeformat:
- 3–7 kurze Sätze als Hauptantwort.
- Wenn Fachwörter vorkommen: Block „Begriffe erklärt:“ (1–3 Definitionen).
- Bei großen Zahlen/Prozenten: Block „Zahl erklärt:“ (einfache Einordnung).
- Keine unnötigen Sonderzeichen oder Emojis; nur Fettdruck für Wichtiges.
- Am Ende: Block „Quellen:“ mit den verwendeten Quellen (Namen aus dem Kontext).
"""

SYSTEM_TEMPLATE = """\
Du bist ein Tutor für Politik & Geschichte. Erkläre Inhalte in Leichter Sprache für Sekundarstufe I.

{ll_rules}

AUSGABEFORMAT:
1) Kurze Antwort (3–7 Sätze).
2) Optional „Begriffe erklärt:“ (1–3 kurze Definitionen) – nur wenn Fachwörter vorkommen.
3) Optional „Zahl erklärt:“ – nur wenn große Zahlen/Prozente vorkommen.
4) Block „Quellen:“ – liste die verwendeten Quellen (Namen aus dem Kontext).
"""

USER_TEMPLATE = """\
KONTEXT:
{context}

FRAGE:
{user_input}

ANTWORT:
"""

def preprocess_input(user_input: str) -> str:
    return " ".join(user_input.strip().split())

def build_messages(user_input: str, search_results: List[Dict]) -> List[Dict]:
    # Dedupe nach Quelle (meta.title) und nimm dann Top-3; Quelle sichtbar anhängen
    picked = []
    seen_titles = set()
    for r in (search_results or []):
        meta = (r.get("meta", {}) or {})
        title = meta.get("title", "")
        norm_title = title.strip().lower()
        if norm_title and norm_title in seen_titles:
            continue
        raw = r.get("text", "").strip()
        if not raw:
            continue
        # Quelle sichtbar machen
        txt = f"{raw}\n\n(Quelle: {title})" if title else raw
        picked.append((norm_title, txt))
        if norm_title:
            seen_titles.add(norm_title)
        if len(picked) >= 3:
            break

    context = "\n\n---\n\n".join(t for _, t in picked) or "Kein zusätzlicher Kontext."
    print("\n**context**: {0}\n".format(context))

    system = SYSTEM_TEMPLATE.format(ll_rules=LL_RULES)
    user = USER_TEMPLATE.format(context=context, user_input=preprocess_input(user_input))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def postprocess_answer(text: str) -> str:
    return " ".join(text.strip().split())
