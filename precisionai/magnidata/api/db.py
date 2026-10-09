# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""SQLite notes database."""

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "notes.db"))


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Open a connection to the notes database.

    Yields a row-factory-enabled connection, committing changes on successful
    exit and always closing the connection afterward.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the notes, datasets, lineage, and partitions tables if missing."""
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

            -- A dataset can derive from more than one source (for example a curated
            -- dataset plus samples pulled from the shared cloud pool).
            CREATE TABLE IF NOT EXISTS dataset_parents (
                child_source  TEXT NOT NULL,
                parent_source TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (child_source, parent_source)
            );
            CREATE INDEX IF NOT EXISTS idx_dataset_parents_parent ON dataset_parents(parent_source);

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
        conn.execute(
            """INSERT OR IGNORE INTO dataset_parents (child_source, parent_source)
               SELECT source, parent_source FROM created_datasets WHERE parent_source IS NOT NULL"""
        )


def list_created_datasets() -> list[dict]:
    """Return all created datasets, most recently created first."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM created_datasets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_created_dataset(ds_id: int) -> dict | None:
    """Return the created dataset with the given id, or None if it doesn't exist."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM created_datasets WHERE id = ?", (ds_id,)).fetchone()
        return dict(row) if row else None


def get_created_dataset_by_source(source: str) -> dict | None:
    """Return the created dataset with the given source path, or None if it doesn't exist."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM created_datasets WHERE source = ?", (source,)).fetchone()
        return dict(row) if row else None


def create_dataset_record(
    name: str,
    description: str,
    source: str,
    parent_source: str | None,
    emb_source: str | None,
    row_count: int,
) -> dict:
    """Insert a new created-dataset row and return the inserted record."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO created_datasets (name, description, source, parent_source, emb_source, row_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, source, parent_source, emb_source, row_count),
        )
        row = conn.execute("SELECT * FROM created_datasets WHERE id = ?", (cur.lastrowid,)).fetchone()
        if parent_source:
            conn.execute(
                "INSERT OR IGNORE INTO dataset_parents (child_source, parent_source) VALUES (?, ?)",
                (source, parent_source),
            )
        return dict(row)


def set_dataset_parents(child_source: str, parent_sources: list[str]) -> None:
    """Replace a dataset's parent edges and keep the legacy first parent in sync."""
    parents = list(dict.fromkeys(parent_sources))
    with get_conn() as conn:
        conn.execute("DELETE FROM dataset_parents WHERE child_source = ?", (child_source,))
        conn.executemany(
            "INSERT INTO dataset_parents (child_source, parent_source) VALUES (?, ?)",
            [(child_source, parent) for parent in parents],
        )
        conn.execute(
            "UPDATE created_datasets SET parent_source = ? WHERE source = ?",
            (parents[0] if parents else None, child_source),
        )


def dataset_parent_map() -> dict[str, list[str]]:
    """Return every child source mapped to its ordered parent sources."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT child_source, parent_source FROM dataset_parents ORDER BY created_at, parent_source"
        ).fetchall()
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row["child_source"], []).append(row["parent_source"])
    return result


def delete_created_dataset(ds_id: int) -> dict | None:
    """Delete the created dataset with the given id and return it, or None if it didn't exist."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM created_datasets WHERE id = ?", (ds_id,)).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM created_datasets WHERE id = ?", (ds_id,))
        conn.execute(
            "DELETE FROM dataset_parents WHERE child_source = ? OR parent_source = ?", (row["source"], row["source"])
        )
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
    """Return notes, optionally filtered by stem, most recently created first."""
    with get_conn() as conn:
        if stem:
            rows = conn.execute("SELECT * FROM notes WHERE stem = ? ORDER BY created_at DESC", (stem,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def create_note(stem: str, image_path: str, note: str) -> dict:
    """Insert a new note and return the inserted record."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes (stem, image_path, note) VALUES (?, ?, ?)",
            (stem, image_path, note),
        )
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)


def update_note(note_id: int, note: str) -> dict | None:
    """Update a note's text and return the updated record, or None if it doesn't exist."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE notes SET note = ?, updated_at = datetime('now') WHERE id = ?",
            (note, note_id),
        )
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None


def delete_note(note_id: int) -> bool:
    """Delete a note by id and return whether a row was deleted."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cur.rowcount > 0
