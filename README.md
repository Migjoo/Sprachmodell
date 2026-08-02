---
title: ZtQ - Sprachmodell
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Deutscher Assistent für Texte und Dokumente
models:
  - Qwen/Qwen2.5-0.5B-Instruct
  - Qwen/Qwen2.5-1.5B-Instruct
  - HuggingFaceTB/SmolLM2-1.7B-Instruct
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - streamlit
  - transformers
  - german
  - text-generation
  - document-question-answering
---

# ZtQ - Sprachmodell

ZtQ ist ein lokaler und modellagnostischer Sprachassistent auf Basis von Streamlit, PyTorch und Hugging Face Transformers.

## Funktionen

- deutsche Texte sprachlich überarbeiten
- Rechtschreibung, Grammatik und Formulierungen verbessern
- PDF-, DOCX-, TXT- und Markdown-Dateien einlesen
- Fragen zu hochgeladenen Dokumenten beantworten
- verschiedene Open-Source-Sprachmodelle auswählen
- eigene kompatible Hugging-Face-Modell-IDs verwenden
- lokale Ausführung ohne externen LLM-Dienst

## Unterstützte Modelle

- Qwen2.5 0.5B Instruct
- Qwen2.5 1.5B Instruct
- SmolLM2 1.7B Instruct
- TinyLlama 1.1B Chat
- weitere kompatible Hugging-Face-Modelle

## Lokaler Start unter Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Die Anwendung ist anschließend normalerweise unter `http://localhost:8501` erreichbar.

## Docker

Docker-Image erstellen:

```powershell
docker build -t ztq-sprachmodell .
```

Container starten:

```powershell
docker run --rm -p 7860:7860 ztq-sprachmodell
```

Die Anwendung ist danach unter `http://localhost:7860` erreichbar.

## Hugging Face Spaces

Das Repository ist für die Bereitstellung als Hugging Face Docker Space vorbereitet.

Beim erstmaligen Laden eines Modells werden die benötigten Modelldateien von Hugging Face heruntergeladen und lokal im Container zwischengespeichert.

## Datenschutz

Hochgeladene Dokumente werden innerhalb der laufenden Anwendung verarbeitet. Für einen produktiven betrieblichen Einsatz sollten zusätzlich Berechtigungen, Löschkonzepte und eine Datenschutzprüfung ergänzt werden.
