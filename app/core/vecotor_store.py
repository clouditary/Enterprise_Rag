"""
Compatibility alias module for 'app.core.vector_store'.
Provides backwards-compatibility with the 'vecotor_store' typo.
"""
from app.core.vector_store import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTOR_SIZE,
    client,
    embedding_model,
    create_collection,
    store_chunks,
)

__all__ = [
    "COLLECTION_NAME",
    "EMBEDDING_MODEL_NAME",
    "VECTOR_SIZE",
    "client",
    "embedding_model",
    "create_collection",
    "store_chunks",
]