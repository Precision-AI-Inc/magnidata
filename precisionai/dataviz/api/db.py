# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""SQLite notes database."""
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "notes.db"))


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                stem        TEXT    NOT NULL,
                image_path  TEXT    NOT NULL DEFAULT '',
                note        TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_notes_stem ON notes(stem);

            -- User-created datasets (carved from a parent dataset's selection)
            CREATE TABLE IF NOT EXISTS created_datasets (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                description   TEXT    NOT NULL DEFAULT '',
                source        TEXT    NOT NULL UNIQUE,   -- child CSV path (relative to DATA_ROOT)
                parent_source TEXT,
                emb_source    TEXT,                      -- path whose stem holds the .json embeddings (or NULL)
                row_count     INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- Cluster partitions: each row of a dataset (by image filename) gets a cluster_id
            -- (1-based, NULL = unassigned) from a committed clustering.
            CREATE TABLE IF NOT EXISTS partitions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_source TEXT    NOT NULL,
                image_name     TEXT    NOT NULL,         -- basename of the image path (join key)
                cluster_id     INTEGER,                  -- NULL until committed
                updated_at     TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(dataset_source, image_name)
            );
            CREATE INDEX IF NOT EXISTS idx_partitions_source ON partitions(dataset_source);
        """)


def list_created_datasets() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM created_datasets ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_created_dataset(ds_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM created_datasets WHERE id = ?", (ds_id,)
        ).fetchone()
        return dict(row) if row else None


def get_created_dataset_by_source(source: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM created_datasets WHERE source = ?", (source,)
        ).fetchone()
        return dict(row) if row else None


def create_dataset_record(name, description, source, parent_source, emb_source, row_count) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO created_datasets (name, description, source, parent_source, emb_source, row_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, source, parent_source, emb_source, row_count),
        )
        row = conn.execute(
            "SELECT * FROM created_datasets WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)


def delete_created_dataset(ds_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM created_datasets WHERE id = ?", (ds_id,)
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM created_datasets WHERE id = ?", (ds_id,))
        return dict(row)


# ── Cluster partitions ───────────────────────────────────────────────────────
def list_partitions(source: str) -> list[dict]:
    """All partition annotations for a dataset, keyed by image filename."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT image_name, cluster_id FROM partitions WHERE dataset_source = ?",
            (source,),
        ).fetchall()
        return [{"image_name": r["image_name"], "cluster_id": r["cluster_id"]} for r in rows]


def commit_partitions(source: str, entries: list[dict]) -> int:
    """Upsert cluster_id for each {image_name, cluster_id}. Re-running overwrites the cluster id."""
    with get_conn() as conn:
        for e in entries:
            conn.execute(
                """INSERT INTO partitions (dataset_source, image_name, cluster_id, updated_at)
                   VALUES (?, ?, ?, datetime('now'))
                   ON CONFLICT(dataset_source, image_name)
                   DO UPDATE SET cluster_id = excluded.cluster_id, updated_at = datetime('now')""",
                (source, e["image_name"], e["cluster_id"]),
            )
        return len(entries)


def list_notes(stem: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if stem:
            rows = conn.execute(
                "SELECT * FROM notes WHERE stem = ? ORDER BY created_at DESC", (stem,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def create_note(stem: str, image_path: str, note: str) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes (stem, image_path, note) VALUES (?, ?, ?)",
            (stem, image_path, note),
        )
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)


def update_note(note_id: int, note: str) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE notes SET note = ?, updated_at = datetime('now') WHERE id = ?",
            (note, note_id),
        )
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_note(note_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cur.rowcount > 0
