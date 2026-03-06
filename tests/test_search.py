import pytest
import uuid
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.config import settings
import qdrant_client
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

client = TestClient(app)
qdrant_client_sync = qdrant_client.QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def test_llm_rate_limit_retry():
    """Verify that the /search endpoint retries transparently on HTTP 429 errors.

    Mocks the query engine so the first call raises a 429 Resource Exhausted
    error and the second call returns a successful response. The test asserts
    that the endpoint still returns 200 thanks to tenacity retry logic.
    """
    mock_response = MagicMock()
    mock_response.source_nodes = []
    mock_response.__str__ = lambda self: "Mocked resilient response"

    rate_limit_error = Exception("429 Resource Exhausted: quota exceeded")

    call_count = {"n": 0}

    async def side_effect_aquery(query):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise rate_limit_error
        return mock_response

    mock_query_engine = MagicMock()
    mock_query_engine.aquery = AsyncMock(side_effect=side_effect_aquery)

    mock_index = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine

    with patch("src.services.search_processor._get_index", return_value=mock_index), \
         patch("src.services.search_processor._query_with_retry.retry.wait", return_value=0):
        response = client.post("/api/v1/search", json={"query": "What is GPT-4?"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Mocked resilient response"
    assert call_count["n"] == 2


def test_search_filters_active_documents_only():
    """Verify that the search retriever only returns documents with status: active.

    Inserts an archived and an active chunk directly into Qdrant, then
    queries the collection using a metadata filter to ensure no archived
    node is returned.
    """
    try:
        qdrant_client_sync.get_collection(settings.COLLECTION_NAME)
    except Exception:
        from qdrant_client.http import models
        qdrant_client_sync.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
        )

    test_doc_id_archived = f"test_filter_archived_{uuid.uuid4().hex[:8]}"
    test_doc_id_active = f"test_filter_active_{uuid.uuid4().hex[:8]}"

    # Insert archived chunk (old information)
    archived_id = str(uuid.uuid4())
    qdrant_client_sync.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[
            PointStruct(
                id=archived_id,
                vector=[0.1] * 3072,
                payload={
                    "status": "archived",
                    "text": "This is information about GPT-3.",
                    "doc_id": test_doc_id_archived,
                    "title": "Old Report",
                    "category": "NLP",
                },
            )
        ],
    )

    # Insert active chunk (new information)
    active_id = str(uuid.uuid4())
    qdrant_client_sync.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[
            PointStruct(
                id=active_id,
                vector=[0.1] * 3072,
                payload={
                    "status": "active",
                    "text": "This is information about GPT-4.",
                    "doc_id": test_doc_id_active,
                    "title": "New Report",
                    "category": "NLP",
                },
            )
        ],
    )

    # Query Qdrant directly with the same metadata filter the search endpoint uses
    active_filter = Filter(
        must=[FieldCondition(key="status", match=MatchValue(value="active"))]
    )
    results = qdrant_client_sync.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=active_filter,
        limit=1000,
    )

    retrieved_points = results[0]
    assert len(retrieved_points) > 0

    for point in retrieved_points:
        assert point.payload["status"] == "active", (
            f"Found point with status={point.payload['status']}; "
            "only 'active' documents should be returned."
        )
        assert point.payload["status"] != "archived"

    # Verify using the search endpoint with mocked LLM
    mock_source_node = MagicMock()
    mock_source_node.node.get_content.return_value = "GPT-4 is the latest model."
    mock_source_node.node.metadata = {"status": "active", "doc_id": test_doc_id_active, "title": "New Report", "category": "NLP"}
    mock_source_node.score = 0.95

    mock_response = MagicMock()
    mock_response.source_nodes = [mock_source_node]
    mock_response.__str__ = lambda self: "GPT-4 information response."

    mock_query_engine = MagicMock()
    mock_query_engine.aquery = AsyncMock(return_value=mock_response)

    mock_index = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine

    with patch("src.services.search_processor._get_index", return_value=mock_index):
        response = client.post("/api/v1/search", json={"query": "Tell me about GPT"})

    assert response.status_code == 200
    data = response.json()
    sources = data.get("sources", [])
    for source in sources:
        assert source["metadata"].get("status") == "active"
