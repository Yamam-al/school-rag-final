#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from urllib import request, error

def post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req) as resp:
            print("Status:", resp.status)
            print(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        print("HTTP Error:", e.code)
        print(e.read().decode("utf-8"))
        sys.exit(1)
    except error.URLError as e:
        print("Connection error:", e)
        sys.exit(1)

def post_file(url: str, file_path: str, meta: dict):
    path = Path(file_path)
    if not path.is_file():
        print("File not found:", file_path)
        sys.exit(1)

    # Multipart boundary
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    boundary_bytes = boundary.encode()

    # Meta JSON part
    meta_json = json.dumps(meta, ensure_ascii=False)
    meta_part = (
        b"--" + boundary_bytes + b"\r\n"
        b'Content-Disposition: form-data; name="meta_json"\r\n\r\n'
        + meta_json.encode("utf-8") + b"\r\n"
    )

    # File part
    file_bytes = path.read_bytes()
    file_part = (
        b"--" + boundary_bytes + b"\r\n"
        b'Content-Disposition: form-data; name="files"; filename="' + path.name.encode() + b'"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" +
        file_bytes + b"\r\n"
    )

    # Final boundary
    end_part = b"--" + boundary_bytes + b"--\r\n"

    body = meta_part + file_part + end_part

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    req = request.Request(url, data=body, headers=headers)

    try:
        with request.urlopen(req) as resp:
            print("Status:", resp.status)
            print(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        print("HTTP Error:", e.code)
        print(e.read().decode("utf-8"))
        sys.exit(1)
    except error.URLError as e:
        print("Connection error:", e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file")
    parser.add_argument("--title", required=True)
    parser.add_argument("--topics", nargs="*", default=[])
    parser.add_argument("--keywords", nargs="*", default=[])
    args = parser.parse_args()

    meta = {
        "title": args.title,
        "topics": args.topics,
        "keywords": args.keywords
    }

    ingest_url = args.base_url.rstrip("/") + "/ingest"

    if args.text:
        payload = {"text": args.text, "meta": meta}
        post_json(ingest_url, payload)
    else:
        post_file(ingest_url, args.file, meta)

if __name__ == "__main__":
    main()
