from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import spaces

from ChatService import ChatService
from DocumentParser import DocumentParser, DocumentParserError
from ZeroGPULLMClient import ZeroGPULLMClient, ZeroGPULLMError

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

llm_client = ZeroGPULLMClient(model_name=MODEL_ID, max_new_tokens=500)
chat_service = ChatService(llm_client)
document_parser = DocumentParser()

@spaces.GPU(duration=120)
def smooth_text(text: str) -> str:
    try:
        return chat_service.smooth_text(text)
    except (ValueError, ZeroGPULLMError) as exc:
        raise gr.Error(str(exc)) from exc

def _file_path(file_value: Any) -> Path:
    if isinstance(file_value, str):
        return Path(file_value)
    name = getattr(file_value, "name", None)
    if name:
        return Path(name)
    raise DocumentParserError("Eine hochgeladene Datei konnte nicht verarbeitet werden.")

def _build_document_context(files: list[Any] | None) -> str:
    if not files:
        raise ValueError("Bitte zuerst mindestens ein Dokument hochladen.")
    document_parts: list[str] = []
    for file_value in files:
        path = _file_path(file_value)
        with path.open("rb") as file_handle:
            text = document_parser.parse(uploaded_file=file_handle, filename=path.name)
        document_parts.append(f"===== DATEI: {path.name} =====\n\n{text}")
    return "\n\n".join(document_parts)

@spaces.GPU(duration=120)
def answer_documents(files: list[Any] | None, question: str) -> str:
    try:
        document_context = _build_document_context(files)
        return chat_service.answer_from_documents(question=question, document_context=document_context)
    except (ValueError, DocumentParserError, ZeroGPULLMError) as exc:
        raise gr.Error(str(exc)) from exc

with gr.Blocks(title="ZtQ - Sprachmodell") as demo:
    gr.Markdown("""
# 🧠 ZtQ - Sprachmodell

Deutschsprachiger Assistent für Textüberarbeitung und Dokumentfragen.

**Online-Modell:** `Qwen/Qwen2.5-0.5B-Instruct`
""")

    with gr.Tab("Text überarbeiten"):
        smoothing_input = gr.Textbox(label="Ausgangstext", lines=10, placeholder="Text eingeben, der sprachlich geglättet werden soll …")
        smoothing_button = gr.Button("Text überarbeiten", variant="primary")
        smoothing_output = gr.Textbox(label="Überarbeitete Fassung", lines=10)
        smoothing_button.click(fn=smooth_text, inputs=smoothing_input, outputs=smoothing_output)

    with gr.Tab("Dokumente befragen"):
        document_files = gr.File(label="Dokumente", file_count="multiple", file_types=[".pdf", ".docx", ".txt", ".md"], type="filepath")
        document_question = gr.Textbox(label="Frage", lines=3, placeholder="Welche Informationen stehen in den Dokumenten?")
        document_button = gr.Button("Frage beantworten", variant="primary")
        document_answer = gr.Textbox(label="Antwort", lines=10)
        document_button.click(fn=answer_documents, inputs=[document_files, document_question], outputs=document_answer)

    gr.Markdown("Die Berechnung erfolgt über Hugging Face ZeroGPU. Die lokale Streamlit-Version bleibt im `main`-Branch erhalten.")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=20).launch()
