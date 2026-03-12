from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.services.document_processor import process_and_ingest_document

router = APIRouter()

class IngestRequest(BaseModel):
    doc_id: str
    title: str
    category: str
    text: str

class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks_processed: int

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    try:
        chunks = await process_and_ingest_document(
            doc_id=request.doc_id,
            title=request.title,
            category=request.category,
            text=request.text
        )
        return IngestResponse(status="success", doc_id=request.doc_id, chunks_processed=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
