import os
import httpx
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/api/v1/ingest"
DATA_DIR = "data"

async def ingest_all_pdfs():
    """Busca todos los PDFs en la carpeta data/ y los envía al endpoint /ingest."""
    if not os.path.exists(DATA_DIR):
        logger.error(f"No se encontró la carpeta {DATA_DIR}")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.info("No se encontraron archivos PDF para ingerir.")
        return

    logger.info(f"Encontrados {len(pdf_files)} archivos PDF. Iniciando ingesta...")

    async with httpx.AsyncClient(timeout=300.0) as client:
        for pdf_file in pdf_files:
            file_path = os.path.join(DATA_DIR, pdf_file)
            logger.info(f"Enviando {pdf_file} a la API...")
            
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (pdf_file, f, "application/pdf")}
                    response = await client.post(API_URL, files=files)
                
                if response.status_code == 200:
                    logger.info(f"Éxito: {pdf_file} procesado correctamente.")
                    logger.info(f"Respuesta: {response.json()}")
                else:
                    logger.error(f"Error al procesar {pdf_file}: {response.status_code} - {response.text}")
            
            except Exception as e:
                logger.error(f"Error de conexión al procesar {pdf_file}: {e}")

if __name__ == "__main__":
    asyncio.run(ingest_all_pdfs())
