"""Database management for Cognio."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .config import settings
from .models import Memory

logger = logging.getLogger(__name__)

# Constants
_DB_NOT_CONNECTED_ERROR = "Database not connected"
_PROJECT_FILTER_SQL = " AND project = ?"


class Database:
    """SQLite database manager for memories."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize database connection."""
        self.db_path = db_path or settings.db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Create database connection and initialize schema."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {self.db_path}")

        # Initialize schema
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        if self.conn is None:
            raise RuntimeError(_DB_NOT_CONNECTED_ERROR)

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
        logger.info("Database schema initialized")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        if self.conn is None:
            raise RuntimeError(_DB_NOT_CONNECTED_ERROR)
        return self.conn.execute(query, params)

    def commit(self) -> None:
        """Commit current transaction."""
        if self.conn is None:
            raise RuntimeError(_DB_NOT_CONNECTED_ERROR)
        self.conn.commit()

    def save_memory(self, memory: Memory) -> None:
        """Save a memory to database."""
        embedding_bytes = None
        if memory.embedding:
            # Convert embedding list to bytes (simple JSON encoding for SQLite)
            embedding_bytes = json.dumps(memory.embedding).encode("utf-8")

        tags_str = json.dumps(memory.tags)

        self.execute(
            """
            INSERT INTO memories (id, text, text_hash, embedding, project, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.text,
                memory.text_hash,
                embedding_bytes,
                memory.project,
                tags_str,
                memory.created_at,
                memory.updated_at,
            ),
        )
        self.commit()

    def get_memory_by_id(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        cursor = self.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    def get_memory_by_hash(self, text_hash: str) -> Memory | None:
        """Retrieve a memory by text hash (for deduplication)."""
        cursor = self.execute(
            "SELECT * FROM memories WHERE text_hash = ? AND archived = 0", (text_hash,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    def list_memories(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with optional filtering."""
        query = "SELECT * FROM memories WHERE archived = 0"
        params: list[Any] = []

        if project:
            query += _PROJECT_FILTER_SQL
            params.append(project)

        if tags:
            # Simple tag filtering (checks if ANY tag matches)
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" AND ({tag_conditions})"
            params.extend([f'%"{tag}"%' for tag in tags])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self.execute(query, tuple(params))
        rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    def count_memories(self, project: str | None = None, tags: list[str] | None = None) -> int:
        """Count total memories with optional filtering."""
        query = "SELECT COUNT(*) FROM memories WHERE archived = 0"
        params: list[Any] = []

        if project:
            query += _PROJECT_FILTER_SQL
            params.append(project)

        if tags:
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            query += f" AND ({tag_conditions})"
            params.extend([f'%"{tag}"%' for tag in tags])

        cursor = self.execute(query, tuple(params))
        result = cursor.fetchone()
        return result[0] if result else 0

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID (hard delete)."""
        cursor = self.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.commit()
        return cursor.rowcount > 0

    def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory by ID (soft delete)."""
        cursor = self.execute(
            "UPDATE memories SET archived = 1 WHERE id = ? AND archived = 0", (memory_id,)
        )
        self.commit()
        return cursor.rowcount > 0

    def bulk_delete(self, project: str | None = None, before_timestamp: int | None = None) -> int:
        """Bulk delete memories (hard delete)."""
        query = "DELETE FROM memories WHERE 1=1"
        params: list[Any] = []

        if project:
            query += _PROJECT_FILTER_SQL
            params.append(project)

        if before_timestamp:
            query += " AND created_at < ?"
            params.append(before_timestamp)

        cursor = self.execute(query, tuple(params))
        self.commit()
        return cursor.rowcount

    def get_all_memories(self) -> list[Memory]:
        """Get all memories (for semantic search, excluding archived)."""
        cursor = self.execute("SELECT * FROM memories WHERE archived = 0 ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        total = self.count_memories()

        # Count by project (excluding archived)
        cursor = self.execute(
            """
            SELECT project, COUNT(*) as count
            FROM memories
            WHERE project IS NOT NULL AND archived = 0
            GROUP BY project
            ORDER BY count DESC
            """
        )
        by_project = {row["project"]: row["count"] for row in cursor.fetchall()}

        # Get top tags (excluding archived)
        cursor = self.execute("SELECT tags FROM memories WHERE tags IS NOT NULL AND archived = 0")
        all_tags: dict[str, int] = {}
        for row in cursor.fetchall():
            tags = json.loads(row["tags"])
            for tag in tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10]

        # Calculate storage size
        db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        storage_mb = db_size / (1024 * 1024)

        return {
            "total_memories": total,
            "total_projects": len(by_project),
            "storage_mb": round(storage_mb, 2),
            "by_project": by_project,
            "top_tags": [tag for tag, _ in top_tags],
        }

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"].decode("utf-8"))

        tags = json.loads(row["tags"]) if row["tags"] else []

        return Memory(
            id=row["id"],
            text=row["text"],
            text_hash=row["text_hash"],
            embedding=embedding,
            project=row["project"],
            tags=tags,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Global database instance
db = Database()
