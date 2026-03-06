from fastapi import FastAPI
from src.api.ingest import router as ingest_router

app = FastAPI(title="Jupiter Phishing Detect - RAG Engine", version="1.0.0")

app.include_router(ingest_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
