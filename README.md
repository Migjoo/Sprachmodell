---
title: ZtQ - Sprachmodell
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: 3.10.13
pinned: false
short_description: Deutscher Assistent für Texte und Dokumente
startup_duration_timeout: 1h
models:
  - Qwen/Qwen2.5-0.5B-Instruct
tags:
  - gradio
  - zerogpu
  - transformers
  - german
  - text-generation
  - document-question-answering
---

# ZtQ - Sprachmodell

Kostenlose Hugging-Face-Version des Projekts mit Gradio und ZeroGPU.

## Funktionen

- deutsche Texte sprachlich überarbeiten
- PDF-, DOCX-, TXT- und Markdown-Dateien einlesen
- Fragen zu hochgeladenen Dokumenten beantworten

## Online-Modell

Für die ZeroGPU-Version wird fest das kompakte Modell `Qwen/Qwen2.5-0.5B-Instruct` verwendet.

Die vollständige modellagnostische Streamlit-Version befindet sich weiterhin im `main`-Branch des GitHub-Repositories.
