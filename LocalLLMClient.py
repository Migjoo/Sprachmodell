from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelDefinition:
    """Konfiguration eines auswählbaren Sprachmodells."""

    model_id: str
    description: str
    max_new_tokens: int = 500
    trust_remote_code: bool = False


MODEL_CATALOG: dict[str, ModelDefinition] = {
    "Qwen2.5 0.5B – sehr schnell": ModelDefinition(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        description=(
            "Kleines Modell für schnelle Funktionstests. "
            "Die Textqualität ist geringer als beim 1.5B-Modell."
        ),
    ),
    "Qwen2.5 1.5B – empfohlen": ModelDefinition(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        description=(
            "Guter Kompromiss aus Geschwindigkeit, Speicherbedarf "
            "und deutscher Textqualität."
        ),
    ),
    "SmolLM2 1.7B – Alternative": ModelDefinition(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        description=(
            "Kompaktes Instruct-Modell von Hugging Face. "
            "Gut zum Vergleich unterschiedlicher Modellfamilien."
        ),
    ),
    "TinyLlama 1.1B – leicht": ModelDefinition(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description=(
            "Relativ kleines Chatmodell. Läuft schnell, ist bei "
            "deutschen Formulierungen aber meist schwächer."
        ),
    ),
}


class LocalLLMError(RuntimeError):
    """Fehler beim Laden oder Ausführen eines lokalen Sprachmodells."""


class LocalLLMClient:
    """
    Modellagnostischer Client für Hugging-Face-Textmodelle.

    Das konkrete Modell wird beim Erstellen der Klasse übergeben.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 500,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )

            self.model.to(self.device)
            self.model.eval()

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = (
                    self.tokenizer.eos_token_id
                )

        except Exception as exc:
            raise LocalLLMError(
                f"Das Modell '{self.model_name}' konnte nicht "
                f"geladen werden:\n\n{exc}"
            ) from exc

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Erzeugt eine Antwort auf einen Chatverlauf."""

        if not messages:
            raise LocalLLMError(
                "Es wurden keine Nachrichten übergeben."
            )

        try:
            prompt = self._create_prompt(messages)

            maximum_input_length = self._get_maximum_input_length()

            model_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=maximum_input_length,
            )

            model_inputs = {
                name: tensor.to(self.device)
                for name, tensor in model_inputs.items()
            }

            input_length = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            answer_tokens = generated_ids[0, input_length:]

            answer = self.tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
                raise LocalLLMError(
                    "Das Modell hat keine Antwort erzeugt."
                )

            return answer

        except LocalLLMError:
            raise

        except Exception as exc:
            raise LocalLLMError(
                f"Fehler während der Textgenerierung:\n\n{exc}"
            ) from exc

    def _create_prompt(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Verwendet nach Möglichkeit das zum Modell gehörende
        Chat-Template.

        Für Modelle ohne Chat-Template wird ein einfaches
        allgemeines Format verwendet.
        """

        chat_template = getattr(
            self.tokenizer,
            "chat_template",
            None,
        )

        if chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_parts: list[str] = []

        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()

            prompt_parts.append(
                f"{role}:\n{content}"
            )

        prompt_parts.append("ASSISTANT:")

        return "\n\n".join(prompt_parts)

    def _get_maximum_input_length(self) -> int:
        """Ermittelt eine sinnvolle maximale Eingabelänge."""

        model_limit = getattr(
            self.model.config,
            "max_position_embeddings",
            4096,
        )

        if not isinstance(model_limit, int):
            model_limit = 4096

        available_length = model_limit - self.max_new_tokens

        return max(
            512,
            min(available_length, 16_000),
        )

    def get_device_name(self) -> str:
        """Gibt den verwendeten Prozessor zurück."""

        if self.device.type == "cuda":
            return torch.cuda.get_device_name(0)

        return "CPU"