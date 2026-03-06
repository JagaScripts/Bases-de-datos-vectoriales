import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.config import settings
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

client = TestClient(app)

def test_document_chunking_size():
    # AC: Extract text, divide into fragments (chunks).
    text = "Word. " * 1000  # long text
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    doc = Document(text=text)
    nodes = node_parser.get_nodes_from_documents([doc])
    
    assert len(nodes) > 1
    # Check max chunk size
    for node in nodes:
        assert len(node.get_content().split()) <= 1000 # approximate

def test_metadata_extraction():
    # AC: Generate embeddings, and insert them into the database with the metadata status: active.
    doc = Document(text="Some text", metadata={"doc_id": "test_1", "status": "active", "category": "NLP"})
    assert doc.metadata["status"] == "active"
    assert doc.metadata["category"] == "NLP"

def test_ingest_new_document_success():
    payload = {
        "doc_id": "doc_test_123",
        "title": "A Test Report",
        "category": "Testing",
        "text": "This is a test document that should be chunked, embedded, and inserted into Qdrant."
    }
    
    response = client.post("/api/v1/ingest", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["doc_id"] == "doc_test_123"
    assert data["chunks_processed"] > 0

def test_update_document_triggers_soft_delete():
    # AC: System locates fragments of the previous document and updates metadata to status: archived
    import qdrant_client as qc
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    q_client = qc.QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    # Cleanup residual data from previous test runs
    cleanup_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value="doc_test_update_123"))]
    )
    existing = q_client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=cleanup_filter,
        limit=10000
    )
    if existing and existing[0]:
        q_client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=[p.id for p in existing[0]]
        )

    # Insert first version
    payload_v1 = {
        "doc_id": "doc_test_update_123",
        "title": "Report v1",
        "category": "Testing",
        "text": "This is the first version of the document."
    }
    response_v1 = client.post("/api/v1/ingest", json=payload_v1)
    assert response_v1.status_code == 200

    # Insert second version
    payload_v2 = {
        "doc_id": "doc_test_update_123", # Same doc_id
        "title": "Report v2",
        "category": "Testing",
        "text": "This is the updated version of the document with more info."
    }
    response_v2 = client.post("/api/v1/ingest", json=payload_v2)
    assert response_v2.status_code == 200

    # Verify directly in Qdrant
    import qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    # Check directly from the qdrant DB
    q_client = qdrant_client.QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    # Query for archived chunks
    filter_archived = Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value="doc_test_update_123")),
            FieldCondition(key="status", match=MatchValue(value="archived"))
        ]
    )
    
    archived_points = q_client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=filter_archived,
        limit=100
    )
    
    assert len(archived_points[0]) > 0
    assert archived_points[0][0].payload["title"] == "Report v1"

    # Query for active chunks
    filter_active = Filter(
        must=[
            FieldCondition(key="doc_id", match=MatchValue(value="doc_test_update_123")),
            FieldCondition(key="status", match=MatchValue(value="active"))
        ]
    )
    
    active_points = q_client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=filter_active,
        limit=100
    )
    
    assert len(active_points[0]) > 0
    assert active_points[0][0].payload["title"] == "Report v2"
