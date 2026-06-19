"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + (ident or "").replace('"', '""') + '"'


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    # Load column descriptions from BIRD's database_description CSVs
    desc_dir = DB_DIR / "dev_20240627" / "dev_databases" / db_id / "database_description"
    col_descs: dict[str, dict[str, str]] = {}  # table -> {original_column_name -> description}
    if desc_dir.exists():
        import csv
        for csv_file in desc_dir.glob("*.csv"):
            table_name = csv_file.stem
            col_descs[table_name] = {}
            with open(csv_file, newline="", encoding="utf-8-sig", errors="replace") as f:
                for row in csv.DictReader(f):
                    orig = (row.get("original_column_name") or "").strip()
                    desc = (row.get("column_description") or row.get("column_name") or "").strip()
                    if orig and desc:
                        col_descs[table_name][orig] = desc

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                desc = col_descs.get(t, {}).get(name, "")
                if desc:
                    line += f"  -- {desc}"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                col_lines.append(
                    f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                )
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
