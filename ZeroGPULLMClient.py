from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ZeroGPULLMError(RuntimeError):
    """Fehler beim Laden oder Ausführen des ZeroGPU-Sprachmodells."""

class ZeroGPULLMClient:
    def __init__(self, model_name: str, max_new_tokens: int = 500) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if os.getenv("SPACE_ID") or torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=self.dtype, low_cpu_mem_usage=True)
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        except Exception as exc:
            raise ZeroGPULLMError(f"Das Modell '{self.model_name}' konnte nicht geladen werden: {exc}") from exc

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            raise ZeroGPULLMError("Es wurden keine Nachrichten übergeben.")
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_limit = getattr(self.model.config, "max_position_embeddings", 4096)
            maximum_input_length = max(512, min(int(model_limit) - self.max_new_tokens, 12000))
            model_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=maximum_input_length)
            model_inputs = {name: tensor.to(self.device) for name, tensor in model_inputs.items()}
            input_length = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.05, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            answer_tokens = generated_ids[0, input_length:]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            if not answer:
                raise ZeroGPULLMError("Das Modell hat keine Antwort erzeugt.")
            return answer
        except ZeroGPULLMError:
            raise
        except Exception as exc:
            raise ZeroGPULLMError(f"Fehler während der Textgenerierung: {exc}") from exc
