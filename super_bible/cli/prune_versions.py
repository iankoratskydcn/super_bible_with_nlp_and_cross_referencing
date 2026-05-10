from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import pandas as pd

from super_bible import paths


@dataclass(frozen=True)
class PruneResult:
    removed_versions: tuple[str, ...]
    kept_versions: tuple[str, ...]
    wrote_db: Path
    wrote_csv: Path
    wrote_pkl: Path


def _backup_if_exists(path: Path, *, suffix: str) -> None:
    if not path.exists():
        return
    backup = path.with_name(path.name + suffix)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)


def _create_views(conn: sqlite3.Connection, *, versions: Iterable[str]) -> None:
    cur = conn.cursor()
    for v in versions:
        v = v.strip().upper()
        if not v or not all(c.isalnum() or c == "_" for c in v):
            raise ValueError(f"Refusing to create sqlite view for unsafe version token: {v!r}")
        # sqlite view name can be directly version token (existing dataset uses bare tokens like ESV).
        cur.execute(f"DROP VIEW IF EXISTS {v}")
        # SQLite does not allow query parameters in CREATE VIEW definitions,
        # so we inline the version token (sanitized above).
        cur.execute(
            f"""
            CREATE VIEW {v} AS
              SELECT * FROM super_bible
              WHERE version = '{v}'
            """
        )


def _export_version_files(df: pd.DataFrame, *, out_version_dir: Path) -> None:
    out_version_dir.mkdir(parents=True, exist_ok=True)
    for v in sorted(df["version"].dropna().unique().tolist()):
        df_v = df[df["version"] == v].copy()
        df_v.to_csv(
            out_version_dir / f"super_bible_{v}.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        df_v.to_pickle(out_version_dir / f"super_bible_{v}.pkl")


def prune_super_bible(
    *,
    original_db_path: Path,
    output_dir: Path,
    remove_versions: Sequence[str],
    delete_version_files: bool,
    delete_raw_inputs: bool,
) -> PruneResult:
    remove_versions_set = {v.strip().upper() for v in remove_versions if v.strip()}
    if not remove_versions_set:
        raise ValueError("remove_versions must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    version_dir = output_dir / "version_files"

    ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    bak_suffix = f".bak.{ts}"

    # Back up top-level exports so pruning is reversible.
    _backup_if_exists(output_dir / "super_bible.db", suffix=bak_suffix)
    _backup_if_exists(output_dir / "super_bible.csv", suffix=bak_suffix)
    _backup_if_exists(output_dir / "super_bible.pkl", suffix=bak_suffix)

    for v in remove_versions_set:
        _backup_if_exists(version_dir / f"super_bible_{v}.csv", suffix=bak_suffix)
        _backup_if_exists(version_dir / f"super_bible_{v}.pkl", suffix=bak_suffix)

    # Load existing table.
    conn_in = sqlite3.connect(str(original_db_path))
    try:
        df = pd.read_sql("SELECT * FROM super_bible", con=conn_in)
    finally:
        conn_in.close()

    if "version" not in df.columns:
        raise RuntimeError("Expected column `version` in sqlite table `super_bible`.")

    df_pruned = df[~df["version"].isin(remove_versions_set)].copy()
    kept_versions = sorted(df_pruned["version"].dropna().unique().tolist())

    # Regenerate sqlite + exports into place (after backups).
    with tempfile.TemporaryDirectory(prefix="super_bible_prune_") as td:
        td_path = Path(td)
        db_tmp = td_path / "super_bible.db"

        conn_out = sqlite3.connect(str(db_tmp))
        try:
            df_pruned.to_sql("super_bible", con=conn_out, index=False, if_exists="replace")
            _create_views(conn_out, versions=kept_versions)
            conn_out.commit()
        finally:
            conn_out.close()

        # Atomically replace top-level exports.
        shutil.copy2(db_tmp, output_dir / "super_bible.db")
        df_pruned.to_csv(
            output_dir / "super_bible.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        df_pruned.to_pickle(output_dir / "super_bible.pkl")

    # Per-version exports.
    _export_version_files(df_pruned, out_version_dir=version_dir)

    if delete_version_files:
        for v in remove_versions_set:
            (version_dir / f"super_bible_{v}.csv").unlink(missing_ok=True)
            (version_dir / f"super_bible_{v}.pkl").unlink(missing_ok=True)

    if delete_raw_inputs:
        # Raw inputs are currently stored as TSVs inside EN/ and ES/.
        # We delete both EN/<VER>.* and ES/<VER>.*
        for lang_dir in [paths.ZRAW_DATA_DIR / "EN", paths.ZRAW_DATA_DIR / "ES"]:
            # Also try exact version matches.
            for v in remove_versions_set:
                for ext in ["tsv", "csv"]:
                    (lang_dir / f"{v}.{ext}").unlink(missing_ok=True)

    return PruneResult(
        removed_versions=tuple(sorted(remove_versions_set)),
        kept_versions=tuple(kept_versions),
        wrote_db=output_dir / "super_bible.db",
        wrote_csv=output_dir / "super_bible.csv",
        wrote_pkl=output_dir / "super_bible.pkl",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune selected bible versions from the local dataset.")
    parser.add_argument(
        "--remove",
        action="append",
        required=True,
        help="Version token to remove (e.g. KSGM, RSEM). Repeatable.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(paths.SUPER_BIBLE_DB_PATH),
        help="Path to super_bible.db.",
    )
    parser.add_argument(
        "--delete-version-files",
        action="store_true",
        help="Delete per-version CSV/PKL files for removed versions.",
    )
    parser.add_argument(
        "--delete-raw-inputs",
        action="store_true",
        help="Delete raw TSV/CSV inputs for removed versions (EN/<VER>.* and ES/<VER>.*).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    remove_versions: list[str] = []
    if args.remove:
        remove_versions = [str(x).upper() for x in args.remove]

    prune_super_bible(
        original_db_path=Path(args.db),
        output_dir=paths.SUPER_BIBLE_DIR,
        remove_versions=remove_versions,
        delete_version_files=args.delete_version_files,
        delete_raw_inputs=args.delete_raw_inputs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

