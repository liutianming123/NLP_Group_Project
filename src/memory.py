"""Core memory operations"""

import logging
import uuid
from datetime import datetime
from typing import Any

from .config import app_config
from .database import db_layer
from .embeddings import vectorizer
from .models import MemoryRecord, RetrievedMemory, StoreMemoryInput
from .utils import create_content_hash, current_timestamp_seconds, timestamp_to_iso_str

logger = logging.getLogger(__name__)

# Constants
UTC_OFFSET_STR = "+00:00"


class CognitiveStore:
    """Service for managing memories."""

    def __init__(self) -> None:
        """Initialize memory service."""
        pass

    def add_new_memory(self, request: StoreMemoryInput) -> tuple[str, bool, str]:
        """
        Save a new memory.

        Args:
            request: Save memory request

        Returns:
            Tuple of (memory_id, is_duplicate, reason)
        """
        # Validate text length
        if len(request.text) > app_config.max_text_length:
            raise ValueError(f"Text content exceeds max length of {app_config.max_text_length}")

        # Generate text hash for deduplication
        content_hash = create_content_hash(request.text)

        # Check for duplicates
        existing_record = db_layer.fetch_memory_by_content_hash(content_hash)
        if existing_record:
            logger.info(f"Duplicate memory record detected: {existing_record.id}")
            return existing_record.id, True, "duplicate"

        # Generate embedding
        embedding_vector = vectorizer.generate_embedding(request.text)

        # Create memory object
        new_memory_id = str(uuid.uuid4())
        current_time = current_timestamp_seconds()

        memory_dto = MemoryRecord(
            id=new_memory_id,
            text=request.text,
            text_hash=content_hash,
            embedding=embedding_vector,
            project=request.project,
            tags=request.tags,
            created_at=current_time,
            updated_at=current_time,
        )

        # Save to database
        db_layer.persist_memory_record(memory_dto)
        logger.info(f"New memory record persisted: {new_memory_id}")

        return new_memory_id, False, "created"

    def find_relevant_memories(
        self,
        query: str,
        project: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
        threshold: float = 0.7,
        after_date: str | None = None,
        before_date: str | None = None,
    ) -> list[RetrievedMemory]:
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
        query_embedding_vector = vectorizer.generate_embedding(query)

        # Get all memories (with optional filters)
        all_memory_records = db_layer.fetch_all_active_memories()

        # Filter by project and tags if specified
        if project:
            all_memory_records = [m for m in all_memory_records if m.project == project]

        if tags:
            all_memory_records = [m for m in all_memory_records if any(tag in m.tags for tag in tags)]

        # Filter by date range
        if after_date:
            try:
                after_ts_val = int(
                    datetime.fromisoformat(after_date.replace("Z", UTC_OFFSET_STR)).timestamp()
                )
                all_memory_records = [m for m in all_memory_records if m.created_at >= after_ts_val]
            except ValueError:
                pass  # Ignore invalid date format

        if before_date:
            try:
                before_ts_val = int(
                    datetime.fromisoformat(before_date.replace("Z", UTC_OFFSET_STR)).timestamp()
                )
                all_memory_records = [m for m in all_memory_records if m.created_at <= before_ts_val]
            except ValueError:
                pass  # Ignore invalid date format

        # Calculate similarities
        scored_results: list[tuple[MemoryRecord, float]] = []
        for record in all_memory_records:
            if record.embedding is None:
                continue

            similarity_score = vectorizer.calculate_cosine_similarity(query_embedding_vector, record.embedding)
            if similarity_score >= threshold:
                scored_results.append((record, similarity_score))

        # Sort by score (descending) and take top-k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:limit]

        # Convert to MemoryResult
        return [
            RetrievedMemory(
                id=memory.id,
                text=memory.text,
                score=round(score, 4),
                project=memory.project,
                tags=memory.tags,
                created_at=timestamp_to_iso_str(memory.created_at),
            )
            for memory, score in top_results
        ]

    def get_all_memories_paginated(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        limit: int = 20,
        sort: str = "date",
        search_query: str | None = None,
    ) -> tuple[list[RetrievedMemory], int]:
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
        page_offset = (page - 1) * limit

        # Get all memories for relevance sorting, or use database pagination for date
        if sort == "relevance" and search_query:
            all_records = db_layer.retrieve_paginated_memories(project=project, tags=tags, limit=10000, offset=0)
            total_item_count = len(all_records)

            # Generate query embedding and calculate scores
            query_emb = vectorizer.generate_embedding(search_query)
            memories_with_scores = []

            for record in all_records:
                if record.embedding:
                    relevance_score = vectorizer.calculate_cosine_similarity(query_emb, record.embedding)
                    memories_with_scores.append((record, relevance_score))

            # Sort by relevance score
            memories_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Paginate
            paginated_list = memories_with_scores[page_offset : page_offset + limit]

            final_results = [
                RetrievedMemory(
                    id=memory.id,
                    text=memory.text,
                    score=round(score, 4),
                    project=memory.project,
                    tags=memory.tags,
                    created_at=timestamp_to_iso_str(memory.created_at),
                )
                for memory, score in paginated_list
            ]
        else:
            # Default: sort by date
            memory_records = db_layer.retrieve_paginated_memories(project=project, tags=tags, limit=limit, offset=page_offset)
            total_item_count = db_layer.count_total_memories(project=project, tags=tags)

            final_results = [
                RetrievedMemory(
                    id=memory.id,
                    text=memory.text,
                    score=None,
                    project=memory.project,
                    tags=memory.tags,
                    created_at=timestamp_to_iso_str(memory.created_at),
                )
                for memory in memory_records
            ]

        return final_results, total_item_count

    def remove_memory_by_id(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory UUID

        Returns:
            True if deleted, False if not found
        """
        was_deleted = db_layer.hard_delete_memory(memory_id)
        if was_deleted:
            logger.info(f"Memory record permanently deleted: {memory_id}")
        else:
            logger.warning(f"Could not find memory record to delete: {memory_id}")
        return was_deleted

    def remove_memories_in_bulk(self, project: str | None = None, before_date: str | None = None) -> int:
        """
        Bulk delete memories.

        Args:
            project: Delete by project
            before_date: Delete before date (ISO 8601)

        Returns:
            Number of deleted memories
        """
        cutoff_timestamp = None
        if before_date:
            try:
                dt_obj = datetime.fromisoformat(before_date.replace("Z", UTC_OFFSET_STR))
                cutoff_timestamp = int(dt_obj.timestamp())
            except ValueError as e:
                raise ValueError(f"Date format is invalid: {before_date}") from e

        deleted_count = db_layer.hard_bulk_delete(project=project, before_timestamp=cutoff_timestamp)
        logger.info(f"Bulk deletion complete. {deleted_count} records removed.")
        return deleted_count

    def fetch_service_analytics(self) -> dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with stats
        """
        return db_layer.collect_database_statistics()

    def dump_memories_to_format(
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
        all_memories = db_layer.retrieve_paginated_memories(project=project, limit=10000)

        if format == "json":
            return {
                "memories": [
                    {
                        "id": m.id,
                        "text": m.text,
                        "project": m.project,
                        "tags": m.tags,
                        "created_at": timestamp_to_iso_str(m.created_at),
                    }
                    for m in all_memories
                ]
            }
        elif format == "markdown":
            md_lines = ["# Memory Export\n"]
            for m in all_memories:
                md_lines.append(f"## {m.id}")
                md_lines.append(f"**Project**: {m.project or 'None'}")
                md_lines.append(f"**Tags**: {', '.join(m.tags) if m.tags else 'None'}")
                md_lines.append(f"**Created**: {timestamp_to_iso_str(m.created_at)}")
                md_lines.append(f"\n{m.text}\n")
                md_lines.append("---\n")
            return "\n".join(md_lines)
        else:
            raise ValueError(f"Format not supported: {format}")


# Global memory service instance
cognitive_store_instance = CognitiveStore()