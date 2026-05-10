from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import unittest

from super_bible.cli.prune_versions import prune_super_bible


class TestPruneVersions(unittest.TestCase):
    def test_prune_removes_versions_from_views_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            original_db = td_path / "super_bible.db"
            output_dir = td_path / "SUPER_BIBLE"

            df = pd.DataFrame(
                [
                    {
                        "testament": "OT",
                        "book": 1,
                        "title": "Genesis",
                        "chapter": 1,
                        "verse": 1,
                        "text": "text1",
                        "version": "ESV",
                        "language": "EN",
                    },
                    {
                        "testament": "NT",
                        "book": 777,
                        "title": "Evangelio de Maria Magdalena",
                        "chapter": 4,
                        "verse": 122,
                        "text": "ksgm1",
                        "version": "KSGM",
                        "language": "EN",
                    },
                    {
                        "testament": "NT",
                        "book": 777,
                        "title": "Evangelio de Maria Magdalena",
                        "chapter": 4,
                        "verse": 122,
                        "text": "rsem1",
                        "version": "RSEM",
                        "language": "ES",
                    },
                ]
            )

            conn = sqlite3.connect(str(original_db))
            try:
                df.to_sql("super_bible", con=conn, index=False, if_exists="replace")
                conn.commit()
            finally:
                conn.close()

            res = prune_super_bible(
                original_db_path=original_db,
                output_dir=output_dir,
                remove_versions=["KSGM", "RSEM"],
                delete_version_files=True,
                delete_raw_inputs=False,
            )

            self.assertEqual(res.removed_versions, ("KSGM", "RSEM"))
            self.assertEqual(res.kept_versions, ("ESV",))

            out_db = output_dir / "super_bible.db"
            conn2 = sqlite3.connect(str(out_db))
            try:
                cur = conn2.cursor()
                cur.execute("SELECT DISTINCT version FROM super_bible ORDER BY version")
                versions_in_table = [r[0] for r in cur.fetchall()]
                self.assertEqual(versions_in_table, ["ESV"])

                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name"
                )
                views = [r[0] for r in cur.fetchall()]
                self.assertIn("ESV", views)
                self.assertNotIn("KSGM", views)
                self.assertNotIn("RSEM", views)
            finally:
                conn2.close()

            version_dir = output_dir / "version_files"
            self.assertTrue((version_dir / "super_bible_ESV.csv").exists())
            self.assertFalse((version_dir / "super_bible_KSGM.csv").exists())
            self.assertFalse((version_dir / "super_bible_RSEM.csv").exists())


if __name__ == "__main__":
    unittest.main()

