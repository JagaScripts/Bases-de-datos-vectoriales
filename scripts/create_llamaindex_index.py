import os
import json
import logging
import qdrant_client
from qdrant_client.models import VectorParams, Distance
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from src.config import settings

logger = logging.getLogger(__name__)

async def create_index_from_processed_chunks(data_path: str = "data/optimized_chunks", summaries_path: str = "data/summaries.json"):
    """
    Carga los Mini-PDFs procesados en Qdrant enriqueciéndolos con los metadatos de los resúmenes.
    """
    if not os.path.exists(data_path):
        logger.error(f"No se encontró la carpeta de chunks: {data_path}")
        return 0

    # Configurar modelos en Settings de LlamaIndex
    if settings.GEMINI_API_KEY == "dummy_key_for_testing" or not settings.GEMINI_API_KEY:
        from llama_index.core.llms import MockLLM
        from llama_index.core.embeddings import MockEmbedding
        Settings.llm = MockLLM(max_tokens=64)
        Settings.embed_model = MockEmbedding(embed_dim=3072)
    else:
        Settings.llm = GoogleGenAI(model=settings.MODEL_NAME, api_key=settings.GEMINI_API_KEY)
        Settings.embed_model = GoogleGenAIEmbedding(model_name=settings.EMBEDDING_MODEL, api_key=settings.GEMINI_API_KEY)

    # Cargar resúmenes para metadatos
    summaries = {}
    if os.path.exists(summaries_path):
        with open(summaries_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)

    # Leer documentos de la carpeta
    reader = SimpleDirectoryReader(data_path)
    documents = reader.load_data()

    # Enriquecer documentos con metadatos del resumen original
    for doc in documents:
        file_name = doc.metadata.get("file_name", "")
        if file_name in summaries:
            doc.metadata["section_title"] = summaries[file_name]["title"]
            doc.metadata["section_summary"] = summaries[file_name]["summary"]
        doc.metadata["status"] = "active" # Mantener compatibilidad con filtros existentes

    # Configuración de Qdrant (Unificado en ai_reports como se pidió)
    client = qdrant_client.QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=60)
    aclient = qdrant_client.AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=60)
    
    # Asegurar que la colección existe
    try:
        client.get_collection(settings.COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=settings.COLLECTION_NAME
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Crear el índice y guardar los documentos
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    logger.info(f"Indexación completada en {settings.COLLECTION_NAME}")
    return len(documents)

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(create_index_from_processed_chunks())
