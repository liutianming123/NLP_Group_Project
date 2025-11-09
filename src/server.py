"""FastAPI application"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from .config import app_config
from .database import db_layer
from .embeddings import vectorizer
from .memory import cognitive_store_instance
from .models import (
    BulkRemovalRequest,
    BulkRemovalOutput,
    DeletionResult,
    ListMemoryOutput,
    StoreMemoryInput,
    StoreMemoryOutput,
    QueryMemoryOutput,
    StatisticsResponse,
)

# Configure logging
logging.basicConfig(
    level=app_config.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
GENERIC_SERVER_ERROR = "An unexpected internal server error occurred"
PROJECT_FILTER_DESC = "Filter by project"


@asynccontextmanager
async def app_lifespan_manager(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Initializing Cognio server...")
    app_config.validate_db_directory_exists()
    db_layer.initialize_connection()
    vectorizer.initialize_transformer()
    logger.info("Server initialization complete. Ready to accept requests.")

    yield

    # Shutdown
    logger.info("Server shutting down...")
    db_layer.close_connection()


# Create FastAPI app
app = FastAPI(
    title="Cognio",
    description="Persistent semantic memory server for MCP",
    version="0.1.0",
    lifespan=app_lifespan_manager,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API key security
api_key_security_schema = APIKeyHeader(name="X-API-Key", auto_error=False)


def validate_api_key(api_key: str | None = Security(api_key_security_schema)) -> bool:
    """Verify API key if configured."""
    if app_config.api_key and (not api_key or api_key != app_config.api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


@app.get("/")
def get_root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": "Cognio",
        "version": "0.1.0",
        "description": "Persistent semantic memory server for MCP",
    }


@app.post("/memory/save", response_model=StoreMemoryOutput)
async def handle_save_memory(
    request: StoreMemoryInput, authenticated: bool = Security(validate_api_key)
) -> StoreMemoryOutput:
    """
    Save a new memory.

    Args:
        request: Memory data with text, optional project and tags

    Returns:
        StoreMemoryOutput with memory ID and status
    """
    try:
        mem_id, is_dup, status_reason = cognitive_store_instance.add_new_memory(request)
        return StoreMemoryOutput(id=mem_id, saved=True, reason=status_reason, duplicate=is_dup)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during memory save operation: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.get("/memory/search", response_model=QueryMemoryOutput)
async def handle_search_memory(
    q: str = Query(..., description="Search query"),
    project: str | None = Query(None, description=PROJECT_FILTER_DESC),
    tags: str | None = Query(None, description="Comma-separated tags"),
    limit: int = Query(5, ge=1, le=50, description="Maximum results"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
    after_date: str | None = Query(None, description="Filter memories after date (ISO 8601)"),
    before_date: str | None = Query(None, description="Filter memories before date (ISO 8601)"),
) -> QueryMemoryOutput:
    """
    Search memories using semantic similarity.

    Args:
        q: Search query text
        project: Optional project filter
        tags: Comma-separated tags to filter
        limit: Maximum number of results
        threshold: Minimum similarity score
        after_date: Filter memories after this date
        before_date: Filter memories before this date

    Returns:
        QueryMemoryOutput with matching memories
    """
    try:
        # Parse tags if provided
        tags_list = [t.strip() for t in tags.split(",")] if tags else None

        search_results = cognitive_store_instance.find_relevant_memories(
            query=q,
            project=project,
            tags=tags_list,
            limit=limit,
            threshold=threshold,
            after_date=after_date,
            before_date=before_date,
        )

        return QueryMemoryOutput(query=q, results=search_results, total=len(search_results))
    except Exception as e:
        logger.error(f"Error during memory search operation: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.get("/memory/list", response_model=ListMemoryOutput)
async def handle_list_memories(
    project: str | None = Query(None, description=PROJECT_FILTER_DESC),
    tags: str | None = Query(None, description="Comma-separated tags"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    sort: str = Query("date", pattern="^(date|relevance)$", description="Sort order"),
    q: str | None = Query(None, description="Query for relevance sorting"),
) -> ListMemoryOutput:
    """
    List all memories with pagination.

    Args:
        project: Optional project filter
        tags: Comma-separated tags to filter
        page: Page number (1-indexed)
        limit: Items per page
        sort: Sort order (date or relevance)
        q: Query text for relevance sorting

    Returns:
        ListMemoryOutput with paginated memories
    """
    try:
        # Parse tags if provided
        tags_list = [t.strip() for t in tags.split(",")] if tags else None

        memories_list, total_items_count = cognitive_store_instance.get_all_memories_paginated(
            project=project, tags=tags_list, page=page, limit=limit, sort=sort, search_query=q
        )

        num_total_pages = (total_items_count + limit - 1) // limit

        return ListMemoryOutput(
            memories=memories_list, page=page, total_pages=num_total_pages, total_items=total_items_count
        )
    except Exception as e:
        logger.error(f"Error during memory list operation: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.delete("/memory/{memory_id}", response_model=DeletionResult)
async def handle_delete_memory(
    memory_id: str, authenticated: bool = Security(validate_api_key)
) -> DeletionResult:
    """
    Delete a memory by ID (hard delete).

    Args:
        memory_id: Memory UUID

    Returns:
        DeletionResult with status
    """
    try:
        was_deleted = cognitive_store_instance.remove_memory_by_id(memory_id)
        if not was_deleted:
            raise HTTPException(status_code=404, detail="Memory record not found")
        return DeletionResult(deleted=True, id=memory_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during memory delete operation: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.post("/memory/{memory_id}/archive", response_model=DeletionResult)
async def handle_archive_memory(
    memory_id: str, authenticated: bool = Security(validate_api_key)
) -> DeletionResult:
    """
    Archive a memory by ID (soft delete).

    Archived memories are excluded from search and list operations
    but not permanently deleted from the database.

    Args:
        memory_id: Memory UUID

    Returns:
        DeletionResult with status
    """
    try:
        # Note: Using the db_layer directly here as in original code
        was_archived = db_layer.soft_delete_memory(memory_id)
        if not was_archived:
            raise HTTPException(status_code=404, detail="Memory not found or was already archived")
        return DeletionResult(deleted=True, id=memory_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during memory archive operation: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.post("/memory/bulk-delete", response_model=BulkRemovalOutput)
async def handle_bulk_delete_memories(
    request: BulkRemovalRequest, authenticated: bool = Security(validate_api_key)
) -> BulkRemovalOutput:
    """
    Bulk delete memories.

    Args:
        request: Bulk delete criteria (project, before_date)

    Returns:
        BulkRemovalOutput with count of deleted memories
    """
    try:
        num_deleted = cognitive_store_instance.remove_memories_in_bulk(
            project=request.project, before_date=request.before_date
        )
        return BulkRemovalOutput(deleted_count=num_deleted)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during memory bulk delete operation: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.get("/memory/stats", response_model=StatisticsResponse)
async def handle_get_stats() -> StatisticsResponse:
    """
    Get memory statistics.

    Returns:
        StatisticsResponse with statistics
    """
    try:
        stats_data = cognitive_store_instance.fetch_service_analytics()
        return StatisticsResponse(**stats_data)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.get("/memory/export", response_model=None)
async def handle_export_memories(
    format: str = Query("json", pattern="^(json|markdown)$", description="Export format"),
    project: str | None = Query(None, description=PROJECT_FILTER_DESC),
) -> JSONResponse | PlainTextResponse:
    """
    Export memories to JSON or Markdown.

    Args:
        format: Export format (json or markdown)
        project: Optional project filter

    Returns:
        Exported data
    """
    try:
        exported_data = cognitive_store_instance.dump_memories_to_format(format=format, project=project)

        if format == "json":
            return JSONResponse(content=exported_data)
        else:
            # Type assertion to help type checker
            content_data = exported_data if isinstance(exported_data, str) else ""
            return PlainTextResponse(content=content_data, media_type="text/markdown")
    except Exception as e:
        logger.error(f"Error during memory export: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_SERVER_ERROR)


@app.get("/health")
async def health_check_endpoint() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=app_config.api_host, port=app_config.api_port)