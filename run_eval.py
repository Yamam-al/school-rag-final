#!/usr/bin/env python
import argparse
import csv
import datetime as dt
from typing import List, Tuple

import requests

# Fester Evaluationskatalog
EVAL_QUESTIONS: List[Tuple[int, str]] = [
    (1, "Was bedeutet Asyl in Deutschland?"),
    (2, "Was ist der Unterschied zwischen Grundrechten und Bürgerrechten?"),
    (3, "Warum gibt es in Deutschland eine Gewaltenteilung?"),
]
#    (4, "Was bedeutet Demokratie?"),
#    (5, "Wie funktioniert eine Bundestagswahl?"),
#    (6, "Wie können Bürgerinnen und Bürger in Deutschland politisch mitbestimmen?"),
#    (7, "Welche Aufgaben hat das Europäische Parlament?"),
#    (8, "Was waren die wichtigsten Merkmale der nationalsozialistischen Diktatur?"),
#    (9, "Warum scheiterte die Weimarer Republik?"),
#    (10, "Was bedeutet Diskriminierung und warum ist sie ein Problem?"),


def call_eval(base_url: str, question: str, timeout: float = 120.0) -> dict:
    url = base_url.rstrip("/") + "/eval"
    resp = requests.post(url, json={"question": question}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ensure_header(path: str, fieldnames: list) -> bool:
    """
    Prüft, ob die Datei existiert.
    Falls nicht, wird später ein Header geschrieben.
    """
    try:
        with open(path, "r", encoding="utf-8"):
            return False
    except FileNotFoundError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Führt eine Evaluation über /eval gegen den RAG Chatbot aus."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Basis URL des Backends, z.B. http://localhost:8080",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name des Durchlaufs, z.B. 'einfach' oder 'komplex'.",
    )
    parser.add_argument(
        "--out",
        default="eval_results.csv",
        help="Pfad zur Ausgabedatei (CSV).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Wie oft jede Frage gestellt werden soll (Default: 1).",
    )

    args = parser.parse_args()

    fieldnames = [
        "timestamp",
        "run_name",
        "question_id",
        "question_text",
        "repetition",
        "ok",
        "error_type",
        "error_message",
        "answer_len_chars",
        "answer",
        "sources",
        "durations_ms_retrieval",
        "durations_ms_prompt_build",
        "durations_ms_llm_time_to_first_token",
        "durations_ms_llm_stream_duration",
        "durations_ms_total",
        "sizes_emitted_chars",
        "sizes_chunks",
    ]

    write_header = ensure_header(args.out, fieldnames)

    with open(args.out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for q_id, q_text in EVAL_QUESTIONS:
            for rep in range(1, args.repetitions + 1):
                print(f"[{args.run_name}] Frage {q_id} (Run {rep}): {q_text}")
                timestamp = dt.datetime.now().isoformat(timespec="seconds")

                try:
                    data = call_eval(args.base_url, q_text)
                except Exception as e:
                    # Falls Backend nicht antwortet oder Fehler wirft
                    row = {
                        "timestamp": timestamp,
                        "run_name": args.run_name,
                        "question_id": q_id,
                        "question_text": q_text,
                        "repetition": rep,
                        "ok": False,
                        "error_type": "request_error",
                        "error_message": str(e),
                        "answer_len_chars": 0,
                        "answer": "",
                        "sources": "",
                        "durations_ms_retrieval": None,
                        "durations_ms_prompt_build": None,
                        "durations_ms_llm_time_to_first_token": None,
                        "durations_ms_llm_stream_duration": None,
                        "durations_ms_total": None,
                        "sizes_emitted_chars": None,
                        "sizes_chunks": None,
                    }
                    writer.writerow(row)
                    continue

                ok = data.get("ok", False)
                error_type = data.get("error_type")
                error_message = data.get("error_message")
                answer = data.get("answer") or ""
                metrics = data.get("metrics") or {}
                durations = metrics.get("durations_ms", {})
                sizes = metrics.get("sizes", {})

                sources = data.get("sources") or []
                sources_str = "|".join(str(s) for s in sources)

                row = {
                    "timestamp": timestamp,
                    "run_name": args.run_name,
                    "question_id": q_id,
                    "question_text": q_text,
                    "repetition": rep,
                    "ok": ok,
                    "error_type": error_type,
                    "error_message": error_message,
                    "answer_len_chars": len(answer),
                    "answer": answer,
                    "sources": sources_str,
                    "durations_ms_retrieval": durations.get("retrieval"),
                    "durations_ms_prompt_build": durations.get("prompt_build"),
                    "durations_ms_llm_time_to_first_token": durations.get(
                        "llm_time_to_first_token"
                    ),
                    "durations_ms_llm_stream_duration": durations.get(
                        "llm_stream_duration"
                    ),
                    "durations_ms_total": durations.get("total"),
                    "sizes_emitted_chars": sizes.get("emitted_chars"),
                    "sizes_chunks": sizes.get("chunks"),
                }

                writer.writerow(row)

    print(f"Fertig. Ergebnisse in {args.out}.")


if __name__ == "__main__":
    main()
