"""Core memory operations for Cognio."""

import logging
import uuid
from datetime import datetime
from typing import Any

from .config import settings
from .database import db
from .embeddings import embedding_service
from .models import Memory, MemoryResult, SaveMemoryRequest
from .utils import format_timestamp, generate_text_hash, get_timestamp

logger = logging.getLogger(__name__)

# Constants
_TIMEZONE_OFFSET = "+00:00"


class MemoryService:
    """Service for managing memories."""

    def __init__(self) -> None:
        """Initialize memory service."""
        pass

    def save_memory(self, request: SaveMemoryRequest) -> tuple[str, bool, str]:
        """
        Save a new memory.

        Args:
            request: Save memory request

        Returns:
            Tuple of (memory_id, is_duplicate, reason)
        """
        # Validate text length
        if len(request.text) > settings.max_text_length:
            raise ValueError(f"Text exceeds maximum length of {settings.max_text_length}")

        # Generate text hash for deduplication
        text_hash = generate_text_hash(request.text)

        # Check for duplicates
        existing = db.get_memory_by_hash(text_hash)
        if existing:
            logger.info(f"Duplicate memory found: {existing.id}")
            return existing.id, True, "duplicate"

        # Generate embedding
        embedding = embedding_service.encode(request.text)

        # Create memory object
        memory_id = str(uuid.uuid4())
        timestamp = get_timestamp()

        memory = Memory(
            id=memory_id,
            text=request.text,
            text_hash=text_hash,
            embedding=embedding,
            project=request.project,
            tags=request.tags,
            created_at=timestamp,
            updated_at=timestamp,
        )

        # Save to database
        db.save_memory(memory)
        logger.info(f"Memory saved: {memory_id}")

        return memory_id, False, "created"

    def search_memory(
        self,
        query: str,
        project: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
        threshold: float = 0.7,
        after_date: str | None = None,
        before_date: str | None = None,
    ) -> list[MemoryResult]:
        """
        Search memories using semantic similarity.

        Args:
            query: Search query text
            project: Optional project filter
            tags: Optional tags filter
            limit: Maximum number of results
            threshold: Minimum similarity score
            after_date: Filter memories after this date (ISO 8601)
            before_date: Filter memories before this date (ISO 8601)

        Returns:
            List of matching memories with scores
        """
        # Generate query embedding
        query_embedding = embedding_service.encode(query)

        # Get all memories (with optional filters)
        all_memories = db.get_all_memories()

        # Filter by project and tags if specified
        if project:
            all_memories = [m for m in all_memories if m.project == project]

        if tags:
            all_memories = [m for m in all_memories if any(tag in m.tags for tag in tags)]

        # Filter by date range
        if after_date:
            try:
                after_ts = int(
                    datetime.fromisoformat(after_date.replace("Z", _TIMEZONE_OFFSET)).timestamp()
                )
                all_memories = [m for m in all_memories if m.created_at >= after_ts]
            except ValueError:
                pass  # Ignore invalid date format

        if before_date:
            try:
                before_ts = int(
                    datetime.fromisoformat(before_date.replace("Z", _TIMEZONE_OFFSET)).timestamp()
                )
                all_memories = [m for m in all_memories if m.created_at <= before_ts]
            except ValueError:
                pass  # Ignore invalid date format

        # Calculate similarities
        results: list[tuple[Memory, float]] = []
        for memory in all_memories:
            if memory.embedding is None:
                continue

            score = embedding_service.cosine_similarity(query_embedding, memory.embedding)
            if score >= threshold:
                results.append((memory, score))

        # Sort by score (descending) and take top-k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

        # Convert to MemoryResult
        return [
            MemoryResult(
                id=memory.id,
                text=memory.text,
                score=round(score, 4),
                project=memory.project,
                tags=memory.tags,
                created_at=format_timestamp(memory.created_at),
            )
            for memory, score in results
        ]

    def list_memories(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        limit: int = 20,
        sort: str = "date",
        search_query: str | None = None,
    ) -> tuple[list[MemoryResult], int]:
        """
        List memories with pagination.

        Args:
            project: Optional project filter
            tags: Optional tags filter
            page: Page number (1-indexed)
            limit: Items per page
            sort: Sort order (date or relevance)
            search_query: Query for relevance sorting

        Returns:
            Tuple of (memories, total_count)
        """
        offset = (page - 1) * limit

        # Get all memories for relevance sorting, or use database pagination for date
        if sort == "relevance" and search_query:
            all_memories = db.list_memories(project=project, tags=tags, limit=10000, offset=0)
            total_count = len(all_memories)

            # Generate query embedding and calculate scores
            query_embedding = embedding_service.encode(search_query)
            scored_memories = []

            for memory in all_memories:
                if memory.embedding:
                    score = embedding_service.cosine_similarity(query_embedding, memory.embedding)
                    scored_memories.append((memory, score))

            # Sort by relevance score
            scored_memories.sort(key=lambda x: x[1], reverse=True)

            # Paginate
            paginated = scored_memories[offset : offset + limit]

            results = [
                MemoryResult(
                    id=memory.id,
                    text=memory.text,
                    score=round(score, 4),
                    project=memory.project,
                    tags=memory.tags,
                    created_at=format_timestamp(memory.created_at),
                )
                for memory, score in paginated
            ]
        else:
            # Default: sort by date
            memories = db.list_memories(project=project, tags=tags, limit=limit, offset=offset)
            total_count = db.count_memories(project=project, tags=tags)

            results = [
                MemoryResult(
                    id=memory.id,
                    text=memory.text,
                    score=None,
                    project=memory.project,
                    tags=memory.tags,
                    created_at=format_timestamp(memory.created_at),
                )
                for memory in memories
            ]

        return results, total_count

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory UUID

        Returns:
            True if deleted, False if not found
        """
        deleted = db.delete_memory(memory_id)
        if deleted:
            logger.info(f"Memory deleted: {memory_id}")
        else:
            logger.warning(f"Memory not found: {memory_id}")
        return deleted

    def bulk_delete(self, project: str | None = None, before_date: str | None = None) -> int:
        """
        Bulk delete memories.

        Args:
            project: Delete by project
            before_date: Delete before date (ISO 8601)

        Returns:
            Number of deleted memories
        """
        before_timestamp = None
        if before_date:
            try:
                dt = datetime.fromisoformat(before_date.replace("Z", _TIMEZONE_OFFSET))
                before_timestamp = int(dt.timestamp())
            except ValueError as e:
                raise ValueError(f"Invalid date format: {before_date}") from e

        count = db.bulk_delete(project=project, before_timestamp=before_timestamp)
        logger.info(f"Bulk deleted {count} memories")
        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with stats
        """
        return db.get_stats()

    def export_memories(
        self, format: str = "json", project: str | None = None
    ) -> str | dict[str, Any]:
        """
        Export memories to JSON or Markdown.

        Args:
            format: Export format (json or markdown)
            project: Optional project filter

        Returns:
            Exported data as string or dict
        """
        memories = db.list_memories(project=project, limit=10000)

        if format == "json":
            return {
                "memories": [
                    {
                        "id": m.id,
                        "text": m.text,
                        "project": m.project,
                        "tags": m.tags,
                        "created_at": format_timestamp(m.created_at),
                    }
                    for m in memories
                ]
            }
        elif format == "markdown":
            lines = ["# Memory Export\n"]
            for m in memories:
                lines.append(f"## {m.id}")
                lines.append(f"**Project**: {m.project or 'None'}")
                lines.append(f"**Tags**: {', '.join(m.tags) if m.tags else 'None'}")
                lines.append(f"**Created**: {format_timestamp(m.created_at)}")
                lines.append(f"\n{m.text}\n")
                lines.append("---\n")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global memory service instance
memory_service = MemoryService()
