"""Utility functions"""

import hashlib
from datetime import datetime


def create_content_hash(content: str) -> str:
    """Generate SHA256 hash of text for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def current_timestamp_seconds() -> int:
    """Get current Unix timestamp."""
    return int(datetime.now().timestamp())


def timestamp_to_iso_str(ts: int) -> str:
    """Convert Unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ts).isoformat() + "Z"