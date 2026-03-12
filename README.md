# RAG Engine - Artificial Intelligence Reports and News

Este proyecto implementa una API de RAG (Retrieval-Augmented Generation) diseñada para gestionar y consultar un flujo dinámico de reportes y noticias sobre Inteligencia Artificial.

## 🚀 Arquitectura
- **Backend:** FastAPI
- **Orquestación RAG:** LlamaIndex
- **Base de Datos Vectorial:** Qdrant (Docker)
- **Modelos:** Google Gemini (Embeddings & LLM)

## 🛠️ Requisitos
- Python 3.10+
- Docker y Docker Compose
- Clave de API de Google Gemini (configurada en `.env`)

## 🔧 Configuración Inicial

1. **Entorno Virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Infraestructura (Qdrant):**
   ```bash
   docker compose up -d
   ```

3. **Variables de Entorno:**
   Crea un archivo `.env` en la raíz del proyecto:
   ```env
   GOOGLE_API_KEY=tu_clave_aqui
   QDRANT_URL=http://localhost:6333
   ```

## 🏃 Ejecución

Para iniciar el servidor de desarrollo:
```bash
uvicorn src.main:app --reload
```

El servidor estará disponible en `http://localhost:8000`. Puedes acceder a la documentación interactiva en `http://localhost:8000/docs`.

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
