"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field

# Constants
_PROJECT_FILTER_DESCRIPTION = "Filter by a specific project"


# Request models
class StoreMemoryInput(BaseModel):
    """DTO for saving a new memory."""

    text: str = Field(..., max_length=10000, description="The textual content of the memory")
    project: str | None = Field(None, max_length=100, description="Categorization project")
    tags: list[str] = Field(default_factory=list, description="List of associated tags")


class QueryMemoryParams(BaseModel):
    """Parameters for memory search operations."""

    q: str = Field(..., description="The search query text")
    project: str | None = Field(None, description=_PROJECT_FILTER_DESCRIPTION)
    tags: str | None = Field(None, description="A comma-delimited string of tags")
    limit: int = Field(5, ge=1, le=50, description="Max number of results to return")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score cutoff")


class ListMemoryParams(BaseModel):
    """Parameters for listing memories."""

    project: str | None = Field(None, description=_PROJECT_FILTER_DESCRIPTION)
    tags: str | None = Field(None, description="A comma-delimited string of tags")
    page: int = Field(1, ge=1, description="Page number for pagination")
    limit: int = Field(20, ge=1, le=100, description="Number of items per page")
    sort: str = Field("date", pattern="^(date|relevance)$", description="Sorting criteria")


class BulkRemovalRequest(BaseModel):
    """DTO for bulk deleting memories."""

    project: str | None = Field(None, description="Delete all memories in this project")
    before_date: str | None = Field(None, description="Delete memories older than this ISO 8601 date")


class ExportOptions(BaseModel):
    """Parameters for memory export."""

    format: str = Field("json", pattern="^(json|markdown)$", description="Output format")
    project: str | None = Field(None, description=_PROJECT_FILTER_DESCRIPTION)


# Response models
class StoreMemoryOutput(BaseModel):
    """Response after saving a memory."""

    id: str = Field(..., description="The unique ID of the memory")
    saved: bool = Field(..., description="Indicates if the save was successful")
    reason: str = Field(..., description="Status message (e.g., 'created', 'duplicate')")
    duplicate: bool = Field(..., description="Flag indicating if the content was a duplicate")


class RetrievedMemory(BaseModel):
    """Represents a single memory item in results."""

    id: str
    text: str
    score: float | None = None
    project: str | None = None
    tags: list[str]
    created_at: str


class QueryMemoryOutput(BaseModel):
    """Response from a memory search."""

    query: str
    results: list[RetrievedMemory]
    total: int


class ListMemoryOutput(BaseModel):
    """Paginated response from listing memories."""

    memories: list[RetrievedMemory]
    page: int
    total_pages: int
    total_items: int


class DeletionResult(BaseModel):
    """Response from a delete operation."""

    deleted: bool
    id: str


class BulkRemovalOutput(BaseModel):
    """Response from a bulk delete operation."""

    deleted_count: int


class StatisticsResponse(BaseModel):
    """Database statistics response."""

    total_memories: int
    total_projects: int
    storage_mb: float
    by_project: dict[str, int]
    top_tags: list[str]


# Internal models
class MemoryRecord(BaseModel):
    """Internal data structure for a memory."""

    id: str
    text: str
    text_hash: str
    embedding: list[float] | None = None
    project: str | None = None
    tags: list[str]
    created_at: int
    updated_at: int