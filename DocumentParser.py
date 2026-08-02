from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from docx import Document
from pypdf import PdfReader


class DocumentParserError(RuntimeError):
    """Fehler beim Auslesen eines hochgeladenen Dokuments."""


class DocumentParser:
    """Liest Text aus PDF-, DOCX-, TXT- und Markdown-Dateien."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    def parse(
        self,
        uploaded_file: BinaryIO,
        filename: str,
    ) -> str:
        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise DocumentParserError(
                f"Der Dateityp '{extension}' wird nicht unterstützt."
            )

        try:
            uploaded_file.seek(0)

            if extension == ".pdf":
                text = self._parse_pdf(uploaded_file)

            elif extension == ".docx":
                text = self._parse_docx(uploaded_file)

            else:
                text = self._parse_text(uploaded_file)

        except DocumentParserError:
            raise

        except Exception as exc:
            raise DocumentParserError(
                f"Die Datei '{filename}' konnte nicht gelesen werden: {exc}"
            ) from exc

        cleaned_text = text.strip()

        if not cleaned_text:
            raise DocumentParserError(
                f"In der Datei '{filename}' wurde kein lesbarer Text gefunden."
            )

        return cleaned_text

    @staticmethod
    def _parse_pdf(uploaded_file: BinaryIO) -> str:
        reader = PdfReader(uploaded_file)

        pages: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(
                    f"--- Seite {page_number} ---\n{page_text.strip()}"
                )

        return "\n\n".join(pages)

    @staticmethod
    def _parse_docx(uploaded_file: BinaryIO) -> str:
        document = Document(uploaded_file)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n\n".join(paragraphs)

    @staticmethod
    def _parse_text(uploaded_file: BinaryIO) -> str:
        raw_data = uploaded_file.read()

        if isinstance(raw_data, str):
            return raw_data

        for encoding in ("utf-8", "utf-8-sig", "cp1252"):
            try:
                return raw_data.decode(encoding)
            except UnicodeDecodeError:
                continue

        raise DocumentParserError(
            "Die Zeichenkodierung der Textdatei konnte nicht erkannt werden."
        )