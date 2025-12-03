# app/rag.py
from typing import List, Dict, Any
from pathlib import Path
import os, uuid, time
import re
import numpy as np  # für Cosine/Dot-Sim

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, Embeddings
from chromadb.utils.embedding_functions import EmbeddingFunction

from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from pypdf import PdfReader

from .config import settings

# ---------------------------
# Globale Umgebungsvariablen
# ---------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------
# Embedding-Modell laden (E5)
# ---------------------------
print("[rag] loading embedder from:", settings.EMBEDDING_MODEL)
t0 = time.perf_counter()
# SentenceTransformer kann E5 direkt laden (offline, wenn Cache vorhanden)
_EMB = SentenceTransformer(
    str(settings.EMBEDDING_MODEL),
    cache_folder=settings.MODEL_CACHE,
    device="cpu",
)
print(f"[rag] embedder loaded in {time.perf_counter()-t0:.2f}s")


class LocalEmbeddingFn(EmbeddingFunction):
    """
    E5 benötigt Prefixes: 'passage:' für Dokumente.
    Da Chroma nur eine Embedding-Funktion kennt, fügen wir diesen Prefix
    für alle Texte hinzu, damit die Semantik stabil bleibt.
    """
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        texts = [f"passage: {t}" for t in list(input)]
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def name(self) -> str:
        return "local-e5"

    def serialize(self) -> Dict[str, Any]:
        return {"name": self.name(), "normalize": True}

# ---------------------------
# Chroma Collection
# ---------------------------
print("[rag] init chroma at:", settings.CHROMA_PATH)
t1 = time.perf_counter()
client = chromadb.PersistentClient(path=settings.CHROMA_PATH, settings=Settings(allow_reset=True))
print(f"[rag] chroma client in {time.perf_counter()-t1:.2f}s")

t2 = time.perf_counter()
collection = client.get_or_create_collection(
    name="school_rag",
    embedding_function=LocalEmbeddingFn(_EMB),
    metadata={"hnsw:space": "cosine"},
)
print(f"[rag] collection ready in {time.perf_counter()-t2:.2f}s")

# ---------------------------
# PDF/Text Einlesen
# ---------------------------
def read_pdf_pymupdf(path: Path) -> str:
    parts: List[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            parts.append(page.get_text("text") or "")
    return "\n".join(parts)

def read_pdf_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def read_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suf == ".pdf":
        try:
            return read_pdf_pymupdf(path)
        except Exception:
            return read_pdf_pypdf(path)
    return ""

# ---------------------------
# Chunking
# ---------------------------
# Satz-/Abschnitt-Splitter: Punkt/Frage/! ODER Bullet '•' ODER Leerzeilen
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|•\s+|\n{2,}")

def chunk(text: str, size: int, overlap: int) -> List[str]:
    # Whitespace normalisieren
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    # 1) In Sätze/Abschnitte teilen
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return [text]  # Fallback

    # 2) Sätze zu Chunks packen
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sents:
        s_len = len(s) + 1  # +1 für Leerzeichen
        if cur and cur_len + s_len > size:
            chunks.append(" ".join(cur))
            # Overlap satzweise: nimm so viele Sätze vom Ende mit, bis overlap erreicht
            keep: List[str] = []
            klen = 0
            for ss in reversed(cur):
                if klen + len(ss) + 1 > overlap:
                    break
                keep.insert(0, ss)
                klen += len(ss) + 1
            cur = keep[:] if keep else []
            cur_len = sum(len(ss) + 1 for ss in cur)

        cur.append(s)
        cur_len += s_len

    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ---------------------------
# Metadaten: Reduziertes Schema & Validierung
# ---------------------------
_ALLOWED_META_KEYS = {
    "title": str,     # optional – Fallback: Dateiname (ohne Suffix)
    "topics": list,   # Pflicht – Liste[str]
    "keywords": list, # Pflicht – Liste[str]
}

def _normalize_and_validate_meta(meta: Dict[str, Any], *, title_fallback: str | None = None) -> Dict[str, Any]:
    """
    - Erlaubt nur title, topics, keywords
    - title: str (wenn None -> fallback auf Dateiname)
    - topics, keywords: Pflicht (Liste[str])
    - Wandelt alles in lowercase um
    - Speichert Listen als durch '|' getrennte Strings (wegen Chroma)
    """
    if meta is None:
        meta = {}

    out: Dict[str, Any] = {}

    # ---------- title ----------
    title_val = meta.get("title")
    if title_val is None:
        if title_fallback:
            out["title"] = str(title_fallback).strip().lower()
        else:
            raise ValueError("Metadatum 'title' fehlt und kein Fallback verfügbar.")
    else:
        out["title"] = str(title_val).strip().lower()

    # ---------- topics ----------
    topics_val = meta.get("topics")
    if topics_val is None:
        raise ValueError("Metadatum 'topics' ist erforderlich (Liste von Strings).")

    if isinstance(topics_val, str):
        topics = [topics_val]
    elif isinstance(topics_val, list):
        topics = [str(x).strip().lower() for x in topics_val if str(x).strip()]
    else:
        raise ValueError("Metadatum 'topics' muss eine Liste von Strings sein.")

    if not topics:
        raise ValueError("Metadatum 'topics' darf nicht leer sein.")

    topics = list(dict.fromkeys(topics))[:50]
    out["topics"] = "|".join(topics)

    # ---------- keywords ----------
    kw_val = meta.get("keywords")
    if kw_val is None:
        raise ValueError("Metadatum 'keywords' ist erforderlich (Liste von Strings).")

    if isinstance(kw_val, str):
        keywords = [kw_val]
    elif isinstance(kw_val, list):
        keywords = [str(x).strip().lower() for x in kw_val if str(x).strip()]
    else:
        raise ValueError("Metadatum 'keywords' muss eine Liste von Strings sein.")

    if not keywords:
        raise ValueError("Metadatum 'keywords' darf nicht leer sein.")

    keywords = list(dict.fromkeys([k for k in keywords if 2 <= len(k) <= 40]))[:100]
    out["keywords"] = "|".join(keywords)

    return out

# ---------------------------
# Ingest
# ---------------------------
def ingest_text(text: str, meta: Dict[str, Any]) -> int:
    """
    Nimmt reinen Text + vom User gelieferte Metadaten entgegen.
    Validiert Metadaten streng (topics/keywords Pflicht).
    """
    parts = chunk(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    print(f"[rag] chunks={len(parts)} first='{parts[0][:60] if parts else ''}'")
    if not parts:
        return 0

    # Hier KEIN Dateiname-Fallback – API-Schicht sollte ingest_file verwenden, wenn Fallback nötig.
    norm_meta = _normalize_and_validate_meta(meta, title_fallback=None)

    base = f"{norm_meta.get('title', 'text')}#{uuid.uuid4().hex}"
    ids = [f"{base}-{k}" for k in range(len(parts))]
    metas = [norm_meta] * len(parts)
    collection.add(documents=parts, metadatas=metas, ids=ids)
    return len(parts)

def ingest_file(path: Path, meta: Dict[str, Any] | None = None) -> int:
    """
    Liest Datei ein und ingested sie mit vom User gelieferten Metadaten.
    - title: falls nicht im Meta vorhanden -> Fallback aus Dateiname (ohne Suffix, bereinigt)
    - topics, keywords: PFLICHT – fehlen sie, wird ValueError geworfen, Datei wird NICHT ingestet.
    """
    raw = read_text(path)
    if not raw:
        raise ValueError(f"Datei '{path}' konnte nicht gelesen werden oder ist leer.")

    # Titel-Fallback aus Dateiname
    fallback_title = path.stem.replace("_", " ").replace("-", " ").strip()
    try:
        norm_meta = _normalize_and_validate_meta(meta or {}, title_fallback=fallback_title)
    except ValueError as e:
        # Explizite Fehlermeldung für API-Layer
        raise ValueError(f"Ingest abgelehnt: {e}")

    base = f"{norm_meta.get('title', fallback_title)}#{uuid.uuid4().hex}"
    parts = chunk(raw, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not parts:
        raise ValueError(f"Datei '{path}' enthielt nach Bereinigung keinen verwertbaren Text.")

    ids = [f"{base}-{k}" for k in range(len(parts))]
    metas = [norm_meta] * len(parts)
    collection.add(documents=parts, metadatas=metas, ids=ids)
    print(f"[rag] ingested file='{path.name}' chunks={len(parts)} title='{norm_meta['title']}' topics={norm_meta['topics']} keywords={len(norm_meta['keywords'])}")
    return len(parts)

# ---------------------------
# Suche (einheitlich, mit Auto-Keywords + 5-Topic-Mapping)
# + Boilerplate-Filter + E5-ReRank (query vs. passage)
# ---------------------------
_DE_STOP = set("""
aber alle allein also am an ander andere anderen anderer anderes anderm andern andres auch auf aus außer ausser bald bei beide beiden beider beides bzw bin bist da dadurch dafür dagegen daher darum dann das dass davon dazu dein deine deinem deinen deiner deines dem demgemäß demzufolge den dennoch der derer des deshalb dessen desto dich die dies diese diesem diesen dieser dieses dir doch dort du durch eher ein eine einem einen einer eines einig einige einigem einigen einiger einiges einmal ende endlich etwa etwas ever falls fast ferner für gegen genug gewesen hab habe haben hat hatte hatten hattest hattet her herein heraus hier hierbei hierin hierzu hinweg hin hinter höchst ich ihr ihre ihrem ihren ihrer ihres ihm ihnen ihr im immer in innen innerhalb ins ist je jeder jede jedem jeden jeder jederem jeden jederer jedes jemanden jemand jenem jenen jener jenes jetzt kann keinerlei kein keine keinem keinen keiner keines keinerlei längst man manche manchem manchen mancher manches mehr mein meine meinem meinen meiner meines mich mir mit mittel mithilfe möglicher möglichst müsste müssen musst müsst nach nachdem nah nahe nein nicht nichts nie niemand noch nun nur ob oder ohne per pro rund sehr sein seine seinem seinen seiner seines seit seitdem seither selbst sollte sollten sonst so sofern sogar solch solche solchem solchen solcher solches sowie später statt trotz trotzdem über um und uns unser unserem unseren unserer unseres unter usw vielmehr viele vielem vielen vieler vieles vielleicht vom von vor wann war waren warum was wegen weil weiter weitere weitem weiteren weiterer weiteres welche welchem welchen welcher welches wenn wer werde werden werdet wird wir wirds wirklich wirst wo wohl wollen wollte wollen wollte während währenddessen währenddem wahr wahrlich zu zum zur zwar zwei zwischen
""".split())
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-]+")

# Boilerplate-Heuristiken
_URL_RE = re.compile(r"https?://|www\.", re.I)
def _looks_like_boilerplate(t: str) -> bool:
    s = t.strip()
    if len(s) < 40:                      # zu kurz, z.B. reiner Titel
        return True
    if len(s) > 1800:                    # zu lang / Sammelabsatz
        return True
    if "Inhalt einfach POLITIK: Lexikon" in s or ("Seite " in s and "Inhalt" in s):
        return True                      # Inhaltsverzeichnisse
    if "Fotonachweis" in s or "picture alliance" in s or "©" in s:
        return True                      # Bild-/Quellennachweise
    if _URL_RE.search(s):
        return True                      # nackte Links-Listen
    if s.count(":") > 10 and s.count("\n") > 5:
        return True                      # Listen/Verzeichnisse
    return False

def _extract_query_keywords(q: str, top_n: int = 8) -> list[str]:
    toks = [t.lower() for t in _WORD_RE.findall(q.lower())]
    toks = [t for t in toks if t not in _DE_STOP and len(t) > 2 and not t.isdigit()]
    if not toks:
        return []
    from collections import Counter
    return [w for w, _ in Counter(toks).most_common(top_n)]

# ---- Kanonische Topics (nur diese 5) ----
_TOPIC_CANON = {"recht", "politik", "gesellschaft", "geschichte", "demokratie"}

# ---- Trigger-Wortschatz -> kanonisches Topic ----
_TOPIC_TRIGGERS = {
    "recht": {
        "grundgesetz","gg","grundrechte","grundrecht","verfassung","verfassungsrecht","staatsrecht",
        "föderalismus","verfassungsorgane","rechtsstaat","rechtsordnung","rechtsprinzip","rechtsstaatlichkeit",
        "gesetz","gesetze","gesetzgebung","gesetzesentwurf","gesetzesvorhaben","verordnung","verwaltungsvorschrift",
        "paragraph","§","artikel","normenkontrolle","verfassungsbeschwerde","verfassungsklage",
        "gericht","gerichte","gerichtshof","bundesverfassungsgericht","bverfg","bundesgerichtshof","bgh",
        "oberlandesgericht","olg","landgericht","amtsgericht","verwaltungsgericht","bundesverwaltungsgericht",
        "sozialgericht","arbeitsgericht","finanzgericht","europäischer gerichtshof","eugh","egmr","menschenrechtsgerichtshof",
        "zuständigkeit","kompetenzordnung","bundesstaat","länderkompetenz","bundeskompetenz",
        "menschenwürde","gleichheitsgrundsatz","gleichbehandlung","allgemeines persönlichkeitsrecht",
        "informationsfreiheit","pressefreiheit","religionsfreiheit","versammlungsfreiheit","vereinigungsfreiheit",
        "unverletzlichkeit der wohnung","briefgeheimnis","datenschutz","informationelle selbstbestimmung",
        "klage","kläger","beklagter","rechtsmittel","revision","berufung","einspruch","anhörung","urteil","beschluss",
        "wahlrecht","wahlprüfungsbeschwerde","mandatsprüfung",
    },
    "politik": {
        "bundestag","bundesrat","bundesregierung","bundeskanzler","kanzler","kanzlerin","bundespräsident",
        "minister","ministerium","staatssekretär","fraktion","opposition","koalition","koalitionsvertrag",
        "partei","parteienfinanzierung","parlament","parlamentarismus","ausschuss","gesetzgebungsverfahren",
        "landtag","landesregierung","ministerpräsident","gemeinderat","stadtrat","bürgermeister","kommunalpolitik",
        "außenpolitik","innenpolitik","wirtschaftspolitik","sozialpolitik","bildungspolitik","familienpolitik",
        "umweltpolitik","klimapolitik","gesundheitspolitik","migrationspolitik","sicherheitspolitik","haushaltspolitik",
        "wahl","wahlen","bundestagswahl","landtagswahl","kommunalwahl","europawahl","erststimme","zweitstimme",
        "stimmzettel","wahlkreis","überhangmandat","ausgleichsmandat","fünf-prozent-hürde","5-prozent-hürde",
        "briefwahl","wahlbenachrichtigung","mandat","koalitionsverhandlungen",
        "eu","europa","europäische union","eu-parlament","europäisches parlament","eu-kommission",
        "europäische kommission","eu-rat","europäischer rat","binnenmarkt","schengen","euro","subsidiarität",
        "grundrechtecharta","gasp","handelspolitik",
        "uno","vereinte nationen","un-sicherheitsrat","nato","osze","rat der europa","europarat",
        "wahlkampf","programm","regierungserklärung","misstrauensvotum","vertrauensfrage","ampel","groko",
        "lobbyismus","transparenzregister","petitionsausschuss",
    },
    "gesellschaft": {
        "gesellschaft","zusammenleben","sozialstaat","soziales","solidarität","vielfalt","diversität","werte",
        "generationenvertrag","demografie","altersarmut","armut","wohlstand","miete","wohnungsmarkt","wohnungslosigkeit",
        "gleichbehandlung","gleichstellung","geschlechtergerechtigkeit","gender","inklusion","barrierefreiheit","teilhabe",
        "behindert","behinderung","behindertenrechte","minderheit","minderheitenrechte","migrationshintergrund",
        "diskriminierung","rassismus","antisemitismus","antiziganismus","xenophobie","feindlichkeit","hasskriminalität",
        "migration","migrant","migranten","flucht","flüchtling","geflüchtete","asyl","asylbewerber","integration","heimat",
        "religion","christentum","islam","judentum","kultur","leitkultur","wertevermittlung","medien","soziale medien",
        "desinformation","falschmeldung","fake news","medienkompetenz",
        "arbeit","arbeitslosigkeit","bildung","schule","ausbildung","hochschule","gesundheit","pflege","rente",
        "zivilgesellschaft","bürgerschaftliches engagement","ehrenamt","verein","stiftung","ngo","initiative",
        "queer","lgbt","lsbtqi","trans","non-binary","geschlechtsidentität",
    },
    "geschichte": {
        "geschichte","weimar","weimarer republik","republik","versailler vertrag","inflation","krise",
        "kapp-putsch","bierhalle","hitler-putsch","bierkellerputsch","reichstag","reichsregierung","notverordnung",
        "reichspräsident","reichskanzler","präsidialkabinett","goldene zwanziger",
        "nsdap","nationalsozialismus","ns-zeit","shoah","holocaust","antisemitismus","gleichschaltung",
        "gestapo","ss","sa","konzentrationslager","kz","deportation","vernichtungslager","auschwitz",
        "zweiter weltkrieg","weltkrieg","hitler","goebbels","himmler","nürnberger gesetze",
        "nachkrieg","trümmerfrauen","besatzungszonen","luftbrücke","berliner mauer","mauerbau",
        "ddr","brd","deutsche teilung","sowjetunion","warschauer pakt","nato","ostpolitik","wiedervereinigung",
        "1989","1990","runder tisch","volkskammer","stasi","prager frühling","1968","studentenbewegung",
        "kaiserreich","revolution 1918","revolution 1918/19","weimarer verfassung","kriegsfolgen",
    },
    "demokratie": {
        "demokratie","herrschaft des volkes","pluralismus","rechtsstaatlichkeit","checks and balances","gewaltenteilung",
        "repräsentative demokratie","direkte demokratie","partizipation","transparenz","minderheitenschutz",
        "mitbestimmung","beteiligung","bürgerbeteiligung","bürgerdialog","bürgerforum","bürgerentscheid",
        "volksbegehren","volksentscheid","petition","petitionsrecht","demonstration","versammlungsfreiheit",
        "wahlrecht","wahlbeteiligung","frei","gleich","geheim","unmittelbar","allgemein","freie wahl",
        "freiheit","gleichheit","menschenrechte","pressefreiheit","meinungsfreiheit","opposition","zivilgesellschaft",
        "demokratiebildung","politische bildung","debattieren","deliberation","konsens","kompromiss",
    }
}

# Aliase/Synonyme -> 5 Klassen
_TOPIC_ALIASES = {
    # recht
    "gg": "recht", "grundgesetz": "recht", "grundrechte": "recht", "eugh": "recht", "egmr": "recht",
    "bverfg": "recht", "bgh": "recht", "verfassungsgericht": "recht", "staatsrecht": "recht",
    "wahlrecht": "recht", "datenschutz": "recht", "rechtsstaat": "recht",

    # politik
    "bundestag": "politik", "bundesrat": "politik", "bundesregierung": "politik", "bundeskanzler": "politik",
    "bundespräsident": "politik", "koalition": "politik", "opposition": "politik", "eu": "politik",
    "europäische union": "politik", "eu-kommission": "politik", "eu-parlament": "politik", "nato": "politik",
    "uno": "politik", "bundestagswahl": "politik", "erststimme": "politik", "zweitstimme": "politik",

    # gesellschaft
    "gesleeshcaft": "gesellschaft",  # absichtlicher tippfehler
    "gesellschaft": "gesellschaft", "zusammenleben": "gesellschaft", "barrierefreiheit": "gesellschaft",
    "inklusion": "gesellschaft", "diskriminierung": "gesellschaft", "rassismus": "gesellschaft",
    "antisemitismus": "gesellschaft", "migration": "gesellschaft", "flucht": "gesellschaft", "asyl": "gesellschaft",
    "queer": "gesellschaft", "lgbt": "gesellschaft", "soziales": "gesellschaft",

    # geschichte
    "ns-zeit": "geschichte", "nsdap": "geschichte", "nationalsozialismus": "geschichte", "weimar": "geschichte",
    "weimarer republik": "geschichte", "versailler vertrag": "geschichte", "ddr": "geschichte", "brd": "geschichte",
    "wiedervereinigung": "geschichte", "kalter krieg": "geschichte", "holocaust": "geschichte", "shoah": "geschichte",
    "stasi": "geschichte",

    # demokratie
    "demokratie": "demokratie", "mitbestimmung": "demokratie", "bürgerbeteiligung": "demokratie",
    "petition": "demokratie", "volksbegehren": "demokratie", "volksentscheid": "demokratie",
    "demonstration": "demokratie", "opposition": "demokratie", "gewaltenteilung": "demokratie",
    "pluralismus": "demokratie", "zivilgesellschaft": "demokratie",

    # mehrdeutig – bewusst fix gemappt
    "staat": "recht", "grundgesetzbuch": "recht", "grundrechtecharta": "politik",
}

def _canon_topic(token: str) -> str | None:
    t = token.strip().lower()
    if not t:
        return None
    if t in _TOPIC_CANON:
        return t
    if t in _TOPIC_ALIASES:
        return _TOPIC_ALIASES[t]
    for canon, words in _TOPIC_TRIGGERS.items():
        if t in words:
            return canon
    return None

def _infer_topics_from_query(q: str) -> list[str]:
    """Lege Topics ausschließlich auf die 5 Labels fest."""
    ql = q.lower()
    found: list[str] = []
    # 1) Triggerlisten
    for canon, words in _TOPIC_TRIGGERS.items():
        if any(w in ql for w in words):
            found.append(canon)
    # 2) Keywords der Query durch Alias-Mapper
    for kw in _extract_query_keywords(q, top_n=12):
        ct = _canon_topic(kw)
        if ct:
            found.append(ct)
    canon_only = [t for t in dict.fromkeys(found) if t in _TOPIC_CANON]
    return canon_only

def _raw_semantic_search(query: str, n: int) -> list[dict]:
    res = collection.query(query_texts=[query], n_results=n, where=None)
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or [0.0] * len(docs)
    return [{"score": float(1.0 - d), "text": t, "meta": m} for t, m, d in zip(docs, metas, dists)]

def _canon_topics_from_meta(meta: dict | None) -> set[str]:
    """Meta.topics ist bei uns ein 'a|b|c'-String. Mappe *alle* Tokens auf die 5 Labels."""
    if not meta:
        return set()
    raw = str(meta.get("topics", "")).lower().split("|")
    out: set[str] = set()
    for r in raw:
        ct = _canon_topic(r)
        if ct:
            out.add(ct)
    return out

def search(query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Einzige öffentliche Suche:
    - extrahiert Query-Keywords
    - leitet Topics ab (nur: recht, politik, gesellschaft, geschichte, demokratie)
    - Vorfilter: Boilerplate entfernen
    - Re-Rank mit E5 (query: vs. passage:) + Meta-Boost
    - Fokussieren nach Topic; Titel-Dedupe; Top-k zurückgeben
    """
    fetch_k = max(40, top_k * 10)
    base_res = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    docs  = base_res.get("documents", [[]])[0]
    metas = base_res.get("metadatas", [[]])[0]
    dists = base_res.get("distances", [[]])[0] or [0.0] * len(docs)

    # Vorfilter: Boilerplate raus
    rows: list[dict] = []
    for t, m, d in zip(docs, metas, dists):
        if not t or not isinstance(m, dict):
            continue
        if _looks_like_boilerplate(t):
            continue
        rows.append({"text": t, "meta": m, "score": 1.0 - float(d)})
    if not rows:
        return []

    print("\n **base hits:** {0}\n\n".format(rows[:top_k*2]))

    q_tokens = set(_extract_query_keywords(query, top_n=12))
    q_topics = set(_infer_topics_from_query(query))
    print("\n **q_tokens:** {0}\n\n".format(q_tokens))
    print("\n **q_topics:** {0}\n\n".format(q_topics))

    # Query-/Passage-Embeddings für Re-Rank (E5)
    q_vec = _EMB.encode([f"query: {query}"], normalize_embeddings=True)[0]
    cand_vecs = _EMB.encode([f"passage: {r['text']}" for r in rows], normalize_embeddings=True)

    def meta_boost(meta: dict) -> float:
        m_topics = _canon_topics_from_meta(meta)
        raw_kw = str(meta.get("keywords", "")).lower().split("|")
        kw_tokens = set()
        for phrase in raw_kw:
            for tok in _WORD_RE.findall(phrase):
                if len(tok) > 2:
                    kw_tokens.add(tok)
        m_keywords = set(filter(None, raw_kw)) | kw_tokens

        b = 0.0
        if q_topics and (q_topics & m_topics):
            b += 0.20
        if q_tokens and (q_tokens & m_keywords):
            b += 0.18
        title = str(meta.get("title", "")).lower() if meta else ""
        if title and any(t in title for t in q_tokens):
            b += 0.06
        return min(b, 0.44)

    # Re-Ranking: Cosine (Dot, weil normiert) + ursprünglicher Score + Meta-Boost
    ranked: list[dict] = []
    for r, v in zip(rows, cand_vecs):
        sim = float(np.dot(q_vec, v))
        combo = 0.70 * sim + 0.30 * r["score"] + meta_boost(r["meta"])
        r["combo_score"] = combo
        ranked.append(r)
    ranked.sort(key=lambda x: x["combo_score"], reverse=True)

    # Fokussieren (Topic), aber immer ein paar Top-Kandidaten zulassen
    focused: list[dict] = []
    for r in ranked:
        m_topics = _canon_topics_from_meta(r.get("meta", {}))
        if not q_topics or (q_topics & m_topics):
            focused.append(r)
    if len(focused) < top_k:
        focused = (focused + ranked[: max(top_k, 6)])[: max(top_k * 2, 8)]

    # Titel-Dedupe → Top-k
    out, seen = [], set()
    for r in focused:
        title = (r.get("meta") or {}).get("title", "").strip().lower()
        if title and title in seen:
            continue
        out.append(r)
        if title:
            seen.add(title)
        if len(out) >= top_k:
            break

    print("\n **ranked:** {0}\n\n".format(out[:top_k]))
    return out
