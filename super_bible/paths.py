from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """
    Compute repo root from this file location.

    This file lives at: <repo_root>/super_bible/paths.py
    so the parent of the `super_bible/` directory is the repo root.
    """

    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """
    Root directory that contains repo data blobs.

    Override with SUPER_BIBLE_DATA_ROOT if you want to point at an external
    data checkout.
    """

    override = os.environ.get("SUPER_BIBLE_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data"


# Authoritative paths for this repo's dataset inputs.
SUPER_BIBLE_DIR = data_root() / "SUPER_BIBLE"
ZRAW_DATA_DIR = data_root() / ".zraw_data"
ZRAW_METADATA_DIR = data_root() / ".zraw_metadata"
CROSS_REFERENCES_CSV = data_root() / "cross_references.csv"

# Common runtime/generated outputs.
OUT_DIR = repo_root() / "out"

SUPER_BIBLE_DB_PATH = SUPER_BIBLE_DIR / "super_bible.db"
CROSSREF_GRAPH_PICKLE_PATH = SUPER_BIBLE_DIR / "crossref_graph.gpickle"

