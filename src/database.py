"""Database management"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from .config import app_config
from .models import MemoryRecord

logger = logging.getLogger(__name__)

# Constants
_ERR_DB_NOT_READY = "Database connection is not initialized"
_PROJECT_FILTER_CLAUSE = " AND project = ?"


class DataPersistence:
    """SQLite database manager for memories."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize database connection."""
        self.db_path = db_path or app_config.db_path
        self.conn: sqlite3.Connection | None = None

    def initialize_connection(self) -> None:
        """Create database connection and initialize schema."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Database connection established: {self.db_path}")

        # Initialize schema
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        if self.conn is None:
            raise RuntimeError(_ERR_DB_NOT_READY)

        cursor = self.conn.cursor()

        # Main memories table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                text_hash TEXT,
                embedding BLOB,
                project TEXT,
                tags TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                archived INTEGER DEFAULT 0
            )
        """
        )

        # Indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project ON memories(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON memories(text_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archived ON memories(archived)")

        self.conn.commit()
        logger.info("Database schema has been verified and initialized.")

    def close_connection(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")

    def execute_query(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        if self.conn is None:
            raise RuntimeError(_ERR_DB_NOT_READY)
        return self.conn.execute(query, params)

    def commit_transaction(self) -> None:
        """Commit current transaction."""
        if self.conn is None:
            raise RuntimeError(_ERR_DB_NOT_READY)
        self.conn.commit()

    def persist_memory_record(self, memory: MemoryRecord) -> None:
        """Save a memory to database."""
        embedding_blob = None
        if memory.embedding:
            # Convert embedding list to bytes (simple JSON encoding for SQLite)
            embedding_blob = json.dumps(memory.embedding).encode("utf-8")

        tags_json_str = json.dumps(memory.tags)

        self.execute_query(
            """
            INSERT INTO memories (id, text, text_hash, embedding, project, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.text,
                memory.text_hash,
                embedding_blob,
                memory.project,
                tags_json_str,
                memory.created_at,
                memory.updated_at,
            ),
        )
        self.commit_transaction()

    def fetch_memory_by_uuid(self, memory_id: str) -> MemoryRecord | None:
        """Retrieve a memory by ID."""
        cursor = self.execute_query("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._map_row_to_memory_object(row)

    def fetch_memory_by_content_hash(self, text_hash: str) -> MemoryRecord | None:
        """Retrieve a memory by text hash (for deduplication)."""
        cursor = self.execute_query(
            "SELECT * FROM memories WHERE text_hash = ? AND archived = 0", (text_hash,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._map_row_to_memory_object(row)

    def retrieve_paginated_memories(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List memories with optional filtering."""
        query = "SELECT * FROM memories WHERE archived = 0"
        params: list[Any] = []

        if project:
            query += _PROJECT_FILTER_CLAUSE
            params.append(project)

        if tags:
            # Simple tag filtering (checks if ANY tag matches)
            tag_clauses = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" AND ({tag_clauses})"
            params.extend([f'%"{tag}"%' for tag in tags])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self.execute_query(query, tuple(params))
        rows = cursor.fetchall()

        return [self._map_row_to_memory_object(row) for row in rows]

    def count_total_memories(self, project: str | None = None, tags: list[str] | None = None) -> int:
        """Count total memories with optional filtering."""
        query = "SELECT COUNT(*) FROM memories WHERE archived = 0"
        params: list[Any] = []

        if project:
            query += _PROJECT_FILTER_CLAUSE
            params.append(project)

        if tags:
            tag_clauses = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" AND ({tag_clauses})"
            params.extend([f'%"{tag}"%' for tag in tags])

        cursor = self.execute_query(query, tuple(params))
        result = cursor.fetchone()
        return result[0] if result else 0

    def hard_delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID (hard delete)."""
        cursor = self.execute_query("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.commit_transaction()
        return cursor.rowcount > 0

    def soft_delete_memory(self, memory_id: str) -> bool:
        """Archive a memory by ID (soft delete)."""
        cursor = self.execute_query(
            "UPDATE memories SET archived = 1 WHERE id = ? AND archived = 0", (memory_id,)
        )
        self.commit_transaction()
        return cursor.rowcount > 0

    def hard_bulk_delete(self, project: str | None = None, before_timestamp: int | None = None) -> int:
        """Bulk delete memories (hard delete)."""
        query = "DELETE FROM memories WHERE 1=1"
        params: list[Any] = []

        if project:
            query += _PROJECT_FILTER_CLAUSE
            params.append(project)

        if before_timestamp:
            query += " AND created_at < ?"
            params.append(before_timestamp)

        cursor = self.execute_query(query, tuple(params))
        self.commit_transaction()
        return cursor.rowcount

    def fetch_all_active_memories(self) -> list[MemoryRecord]:
        """Get all memories (for semantic search, excluding archived)."""
        cursor = self.execute_query("SELECT * FROM memories WHERE archived = 0 ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [self._map_row_to_memory_object(row) for row in rows]

    def collect_database_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        total = self.count_total_memories()

        # Count by project (excluding archived)
        cursor = self.execute_query(
            """
            SELECT project, COUNT(*) as count
            FROM memories
            WHERE project IS NOT NULL AND archived = 0
            GROUP BY project
            ORDER BY count DESC
            """
        )
        project_counts = {row["project"]: row["count"] for row in cursor.fetchall()}

        # Get top tags (excluding archived)
        cursor = self.execute_query("SELECT tags FROM memories WHERE tags IS NOT NULL AND archived = 0")
        tag_frequencies: dict[str, int] = {}
        for row in cursor.fetchall():
            tags_list = json.loads(row["tags"])
            for tag in tags_list:
                tag_frequencies[tag] = tag_frequencies.get(tag, 0) + 1

        top_10_tags = sorted(tag_frequencies.items(), key=lambda item: item[1], reverse=True)[:10]

        # Calculate storage size
        db_file_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        storage_size_mb = db_file_size / (1024 * 1024)

        return {
            "total_memories": total,
            "total_projects": len(project_counts),
            "storage_mb": round(storage_size_mb, 2),
            "by_project": project_counts,
            "top_tags": [tag for tag, _ in top_10_tags],
        }

    def _map_row_to_memory_object(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert database row to Memory object."""
        embedding_data = None
        if row["embedding"]:
            embedding_data = json.loads(row["embedding"].decode("utf-8"))

        tag_data = json.loads(row["tags"]) if row["tags"] else []

        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            text_hash=row["text_hash"],
            embedding=embedding_data,
            project=row["project"],
            tags=tag_data,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Global database instance
db_layer = DataPersistence()