#!/usr/bin/env python3
# coding: utf-8

"""
Query the local ESV verses stored in SUPER_BIBLE/super_bible.db.

Two interfaces:
  1) CLI:  ./esv_query.py --ref "Genesis 1:1-3"
  2) HTTP: ./esv_query.py --serve --port 8000
     GET /v1/esv?ref=Genesis%201:1-3
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class VerseRef:
    book: str
    chapter: int
    start_verse: int
    end_verse: int


_REF_RE = re.compile(
    r"^\s*(?P<book>.+?)\s+"
    r"(?P<chapter>\d+):"
    r"(?P<start>\d+)"
    r"(?:\s*-\s*(?P<end>\d+))?\s*$"
)


def normalize_verse_text(text: str) -> str:
    # SQLite text may contain trailing carriage returns.
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_reference(ref: str) -> VerseRef:
    """
    Parse refs like:
      - "Genesis 1:1"
      - "Genesis 1:1-3"
    """
    m = _REF_RE.match(ref)
    if not m:
        raise ValueError(
            "Invalid reference format. Expected like 'Genesis 1:1' or 'Genesis 1:1-3'."
        )

    book = m.group("book").strip()
    chapter = int(m.group("chapter"))
    start_verse = int(m.group("start"))
    end_verse_raw = m.group("end")
    end_verse = int(end_verse_raw) if end_verse_raw is not None else start_verse

    if chapter <= 0 or start_verse <= 0 or end_verse <= 0:
        raise ValueError("chapter/verse numbers must be positive integers.")
    if end_verse < start_verse:
        raise ValueError("end verse must be >= start verse.")

    return VerseRef(
        book=book,
        chapter=chapter,
        start_verse=start_verse,
        end_verse=end_verse,
    )


def get_db_path() -> Path:
    root = Path(__file__).resolve().parent
    return root / "SUPER_BIBLE" / "super_bible.db"


def query_esv(
    db_path: Path,
    book: str,
    chapter: int,
    start_verse: int,
    end_verse: int,
) -> List[Tuple[int, str]]:
    """
    Returns list of (verse_number, verse_text) sorted by verse asc.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT verse, text
            FROM ESV
            WHERE title = ?
              AND chapter = ?
              AND verse BETWEEN ? AND ?
            ORDER BY verse ASC
            """,
            (book, chapter, start_verse, end_verse),
        )
        rows = cur.fetchall()
        return [(int(v), normalize_verse_text(t)) for (v, t) in rows]
    finally:
        conn.close()


def format_cli(ref: VerseRef, verses: List[Tuple[int, str]]) -> str:
    if not verses:
        return ""

    # Example:
    # Genesis 1:1
    # In the beginning...
    parts: List[str] = []
    if ref.start_verse == ref.end_verse:
        parts.append(f"{ref.book} {ref.chapter}:{ref.start_verse}")
        parts.append(verses[0][1])
    else:
        parts.append(f"{ref.book} {ref.chapter}:{ref.start_verse}-{ref.end_verse}")
        for v, text in verses:
            parts.append(f"{ref.book} {ref.chapter}:{v} {text}")
    return "\n".join(parts)


class ESVHandler(BaseHTTPRequestHandler):
    server_version = "esv_query/1.0"

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/v1/esv":
            self._send_json(404, {"error": "Not found"})
            return

        qs = parse_qs(parsed.query or "")
        ref_list = qs.get("ref", [])
        if not ref_list:
            self._send_json(400, {"error": "Missing query param 'ref'"})
            return

        ref_str = ref_list[0]
        try:
            ref = parse_reference(ref_str)
        except ValueError as e:
            self._send_json(422, {"error": str(e)})
            return

        verses = query_esv(
            db_path=get_db_path(),
            book=ref.book,
            chapter=ref.chapter,
            start_verse=ref.start_verse,
            end_verse=ref.end_verse,
        )
        if not verses:
            self._send_json(404, {"error": "Verse not found"})
            return

        self._send_json(
            200,
            {
                "version": "ESV",
                "book": ref.book,
                "chapter": ref.chapter,
                "start_verse": ref.start_verse,
                "end_verse": ref.end_verse,
                "verses": [{"verse": v, "text": t} for (v, t) in verses],
            },
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Keep logs clean (avoid noisy per-request printing).
        return


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query local ESV verses.")
    parser.add_argument(
        "--ref",
        type=str,
        help='Bible reference like "Genesis 1:1-3".',
    )
    parser.add_argument("--serve", action="store_true", help="Run HTTP server.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000).")

    args = parser.parse_args(argv)

    if args.serve:
        port = args.port
        db_path = get_db_path()
        if not db_path.exists():
            print(f"DB not found at: {db_path}", file=sys.stderr)
            return 1
        httpd = ThreadingHTTPServer(("", port), ESVHandler)
        print(f"Serving ESV on http://localhost:{port}/v1/esv", file=sys.stderr)
        httpd.serve_forever()
        return 0

    if not args.ref:
        parser.error("--ref is required when not using --serve")
        return 2

    ref = parse_reference(args.ref)
    verses = query_esv(
        db_path=get_db_path(),
        book=ref.book,
        chapter=ref.chapter,
        start_verse=ref.start_verse,
        end_verse=ref.end_verse,
    )
    if not verses:
        return 1

    print(format_cli(ref, verses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

