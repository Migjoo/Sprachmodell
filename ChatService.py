from __future__ import annotations

from LocalLLMClient import LocalLLMClient


class ChatService:
    """Fachlogik für Textglättung und Dokumentfragen."""

    SMOOTHING_PROMPT = """
Du bist ein deutschsprachiger Schreibassistent.

Überarbeite den eingegebenen Text nach folgenden Regeln:

- Verbessere Rechtschreibung, Grammatik und Zeichensetzung.
- Formuliere natürlich, verständlich und professionell.
- Erhalte die ursprüngliche Aussage vollständig.
- Erfinde keine zusätzlichen Tatsachen.
- Verändere Namen, Zahlen, Aktenzeichen und Datumsangaben nicht.
- Gib ausschließlich den überarbeiteten Text zurück.
- Schreibe keine Einleitung.
""".strip()

    DOCUMENT_PROMPT = """
Du bist ein deutschsprachiger Dokumentenassistent.

Beantworte die Frage ausschließlich anhand des bereitgestellten
Dokumenteninhalts.

Regeln:

- Erfinde keine Informationen.
- Trenne sichere Aussagen von Unsicherheiten.
- Wenn die Antwort nicht im Dokument steht, sage das ausdrücklich.
- Antworte verständlich und möglichst konkret.
- Nenne nach Möglichkeit den Dateinamen oder die betreffende Seite.
""".strip()

    def __init__(
        self,
        llm_client: LocalLLMClient,
    ) -> None:
        self.llm_client = llm_client

    def smooth_text(
        self,
        text: str,
    ) -> str:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "Der zu überarbeitende Text darf nicht leer sein."
            )

        messages = [
            {
                "role": "system",
                "content": self.SMOOTHING_PROMPT,
            },
            {
                "role": "user",
                "content": cleaned_text,
            },
        ]

        return self.llm_client.chat(messages)

    def answer_from_documents(
        self,
        question: str,
        document_context: str,
    ) -> str:
        cleaned_question = question.strip()
        cleaned_context = document_context.strip()

        if not cleaned_question:
            raise ValueError("Die Frage darf nicht leer sein.")

        if not cleaned_context:
            raise ValueError(
                "Es wurde noch kein lesbares Dokument hochgeladen."
            )

        # Schutz vor einem zu großen Prompt.
        limited_context = cleaned_context[:20_000]

        messages = [
            {
                "role": "system",
                "content": self.DOCUMENT_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "DOKUMENTENINHALT:\n\n"
                    f"{limited_context}\n\n"
                    "FRAGE:\n\n"
                    f"{cleaned_question}"
                ),
            },
        ]

        return self.llm_client.chat(messages)