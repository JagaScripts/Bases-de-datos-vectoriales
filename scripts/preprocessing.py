import os
import json
import logging
from typing import List, Dict, Any
from pypdf import PdfReader, PdfWriter
from pypdf.generic import Destination
from src.config import settings
from llama_index.llms.google_genai import GoogleGenAI

logger = logging.getLogger(__name__)

async def generate_summary(text: str) -> str:
    """Genera un resumen del texto usando Gemini."""
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "dummy_key_for_testing":
        return "Resumen de prueba (Mock)."
    
    llm = GoogleGenAI(model=settings.MODEL_NAME, api_key=settings.GEMINI_API_KEY)
    prompt = f"Resume el siguiente texto de forma concisa pero manteniendo los puntos clave informativos:\n\n{text[:10000]}"
    try:
        response = await llm.acomplete(prompt)
        return str(response)
    except Exception as e:
        logger.error(f"Error generando resumen: {e}")
        return "No se pudo generar el resumen."

async def process_pdf(pdf_path: str, output_dir: str = "data/optimized_chunks") -> Dict[str, Any]:
    """
    Divide un PDF en secciones basadas en sus marcadores y genera resúmenes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    reader = PdfReader(pdf_path)
    bookmarks = reader.outline
    
    sections = []
    
    def extract_bookmarks(outline, level=0):
        for item in outline:
            if isinstance(item, Destination):
                page_num = reader.get_destination_page_number(item)
                sections.append({"title": item.title, "start_page": page_num, "level": level})
            elif isinstance(item, list):
                extract_bookmarks(item, level + 1)

    extract_bookmarks(bookmarks)
    
    # Si no hay marcadores, tratamos el PDF como una sola sección
    if not sections:
        sections.append({"title": os.path.basename(pdf_path).replace(".pdf", ""), "start_page": 0, "level": 0})

    # Calcular páginas finales
    num_pages = len(reader.pages)
    for i in range(len(sections)):
        if i + 1 < len(sections):
            sections[i]["end_page"] = sections[i+1]["start_page"]
        else:
            sections[i]["end_page"] = num_pages

    summaries = {}
    processed_count = 0

    for section in sections:
        writer = PdfWriter()
        start = section["start_page"]
        end = section["end_page"]
        
        # Extraer texto para el resumen
        section_text = ""
        for p in range(start, end):
            writer.add_page(reader.pages[p])
            section_text += reader.pages[p].extract_text() + "\n"
        
        filename = f"{section['title'].replace(' ', '_').replace('/', '_')}.pdf"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "wb") as f:
            writer.write(f)
            
        summary = await generate_summary(section_text)
        summaries[filename] = {
            "title": section["title"],
            "summary": summary,
            "path": filepath
        }
        processed_count += 1
        logger.info(f"Procesada sección: {section['title']}")

    with open("data/summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False)
        
    return {"count": processed_count, "summaries_path": "data/summaries.json"}

if __name__ == "__main__":
    # Para ejecución manual (Opción B del usuario)
    import asyncio
    logging.basicConfig(level=logging.INFO)
    pdf = "data/Estrategia_IA_2024.pdf"
    if os.path.exists(pdf):
        asyncio.run(process_pdf(pdf))
    else:
        print(f"No se encontró el archivo {pdf}")
