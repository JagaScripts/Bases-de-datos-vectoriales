from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import shutil
from src.services.document_processor import process_and_ingest_document
from scripts.preprocessing import process_pdf
from scripts.create_llamaindex_index import create_index_from_processed_chunks

router = APIRouter()

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se admiten archivos PDF.")
    
    temp_path = f"data/raw/{file.filename}"
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        # 1. Guardar archivo original
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Pre-procesamiento (División y Resumen con Gemini)
        # Llamamos físicamente a la lógica del script
        await process_pdf(temp_path)
        
        # 3. Indexación (Carga masiva en Qdrant con LlamaIndex)
        chunks_count = await create_index_from_processed_chunks()
        
        return {
            "status": "success",
            "filename": file.filename,
            "sections_extracted": chunks_count,
            "message": "Documento procesado, resumido e indexado correctamente."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

