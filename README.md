# RAG Engine - Artificial Intelligence Reports and News

Este proyecto implementa una API de RAG (Retrieval-Augmented Generation) diseñada para gestionar y consultar un flujo dinámico de reportes y noticias sobre Inteligencia Artificial.

## 🚀 Arquitectura
- **Backend:** FastAPI
- **Orquestación RAG:** LlamaIndex
- **Base de Datos Vectorial:** Qdrant (Docker)
- **Modelos:** Google Gemini (Embeddings & LLM)

## 🛠️ Requisitos y Comprobación Preliminar

Antes de comenzar, asegúrate de tener instalados los siguientes componentes:

1.  **Docker:** Necesario para la base de datos vectorial Qdrant.
    ```bash
    docker --version
    docker compose version
    ```
2.  **Python 3.10+:**
    ```bash
    python --version
    ```
3.  **Clave de API de Google Gemini:** Necesaria para embeddings y LLM.

## 🔧 Configuración Inicial

1.  **Entorno Virtual:**
    ```bash
    python -m venv .venv
    # En Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1
    # En Linux/Mac:
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

2.  **Infraestructura (Qdrant):**
    Asegúrate de que Docker está corriendo y ejecuta:
    ```bash
    docker compose up -d
    ```

3.  **Variables de Entorno:**
    Crea un archivo `.env` en la raíz del proyecto basándote en el siguiente formato:
    ```env
    GEMINI_API_KEY=tu_clave_aquí
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    ```

## 🏃 Ejecución

Para iniciar el servidor de desarrollo:
```bash
uvicorn src.main:app --reload
```

El servidor estará disponible en `http://localhost:8000`. Puedes acceder a la documentación interactiva en `http://localhost:8000/docs`.

## 📂 Ingesta de Documentos (PDF)

El sistema ahora soporta el procesamiento avanzado de PDFs (división por capítulos y resúmenes automáticos con Gemini).

1. **Carga Manual Individual:** Puedes usar el endpoint `/api/v1/ingest` desde la documentación Swagger (`/docs`) subiendo cualquier archivo PDF.

2. **Carga Masiva (Bulk Ingest):** Para cargar todos los PDFs que tengas en la carpeta `data/` de una sola vez:
   ```bash
   # Asegúrate de que la API esté corriendo en otra terminal
   python -m scripts.bulk_ingest
   ```


## 🧪 Testing

Para ejecutar la suite de pruebas:
```bash
pytest
```

## 📡 Endpoints Principales

- `POST /api/v1/ingest`: Ingesta de nuevos reportes (PDF/Texto). Maneja automáticamente la obsolescencia (Soft Delete).
- `POST /api/v1/search`: Realiza consultas semánticas sobre los documentos activos.
- `GET /health`: Comprobación del estado del servicio.

## 📂 Estructura del Proyecto
- `src/api/`: Definición de endpoints y rutas.
- `src/services/`: Lógica de negocio, procesamiento de documentos y búsqueda.
- `src/config.py`: Configuración global y gestión de variables de entorno.
- `tests/`: Pruebas unitarias e integración.
