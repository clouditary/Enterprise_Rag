import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "enterprise_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Connect to Qdrant (Docker or remote instance running on port 6333)
client = QdrantClient(url="http://localhost:6333")

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_collection():
    """Create the Qdrant collection if it does not already exist."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )


def store_chunks(chunks: list[dict], department: str) -> int:
    """
    Generate dense embeddings for chunks and store them in Qdrant with metadata payload.

    Args:
        chunks: List of text chunk dictionaries with keys:
                'text', 'chunk_id', 'chunk_index', 'page_number'
        department: Department metadata for enterprise access control and filtering

    Returns:
        Total number of points upserted
    """
    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_model.encode(texts, show_progress_bar=True).tolist()

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk["page_number"],
                    "department": department,
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return len(points)
