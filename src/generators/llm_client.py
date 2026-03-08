"""
LLM Client — Maarif-Gen
Desteklenen sağlayıcılar: Ollama (local), vLLM (Docker), OpenAI, Gemini, Claude
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

from src.generators import prompts


# ── Sağlayıcı varsayılanları ───────────────────────────────────────────────────
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "local": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:72b",
        "api_key": "EMPTY",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "api_key": "EMPTY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "api_key": "",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-opus-4-5",
        "api_key": "",
    },
}


class LLMClient:
    """OpenAI uyumlu /chat/completions endpoint'i kullanan çok sağlayıcılı istemci."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "",
        provider: str = "local",
        timeout: int = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.provider = provider
        self.timeout = timeout

        # Yerel sağlayıcılarda model adı sunucudan alınacak; önce boş bırak
        self._model_override = model
        self.model = model or PROVIDER_DEFAULTS.get(provider, {}).get("model", "")

    # ── Bağlantı & model keşfi ────────────────────────────────────────────────

    def check_connection(self) -> bool:
        """Sunucuya bağlanarak ilk modeli seçer. Bağlantı yoksa False döndürür."""
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    # Kullanıcı belirli bir model istemediyse ilkini al
                    if not self._model_override:
                        self.model = models[0].get("id", self.model)
                    else:
                        # Belirtilen model gerçekten var mı diye kontrol et
                        ids = [m.get("id") for m in models]
                        if self._model_override in ids:
                            self.model = self._model_override
                        else:
                            # Sunucuda yoksa mevcut ilk modeli kullan
                            self.model = models[0].get("id", self.model)
                return True
        except Exception as exc:
            logger.warning("LLM bağlantı hatası: %s", exc)
        return False

    # ── Dahili yardımcılar ────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.provider == "claude":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _filter_reasoning(self, text: str) -> str:
        """Modelin içsel akıl yürütme (thought/reasoning) bloklarını temizler."""
        # 1. <thought> ... </thought> bloklarını temizle
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        # 2. 'Thinking Process:', 'Düşünme Süreci:' gibi başlıkları ve altındaki bloğu temizle
        # Genellikle bu kısımlar metnin başında olur ve boş satırla biter
        text = re.sub(r"^(?:Thinking Process|Düşünme Süreci|Reasoning):.*?(?:\n\s*\n|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
        # 3. Kalan döküntüleri (başıboş düşünce notları vb.) temizle
        return text.strip()

    def _chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Chat completion isteği gönderir; ham metin yanıtını döndürür."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"]
        return self._filter_reasoning(raw_text)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """LLM çıktısından ilk JSON bloğunu ayıklar."""
        # Markdown kod bloğunu temizle
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        # İlk { ... } bloğunu bul
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)

    # ── Public API ────────────────────────────────────────────────────────────

    def _create_prompt(self, outcome: dict, context: str) -> str:
        """Soru üretimi için kullanıcı mesajı oluşturur (Geriye uyumluluk için, doğrudan da çağrılabilir)."""
        return prompts.build_question_user_prompt(outcome, context)

    def generate_context(self, outcome: dict, impl_guide: str = "") -> str:
        """Öğrenme çıktısı için bağlam/senaryo üretir."""
        prompt = prompts.build_context_user_prompt(outcome, impl_guide)

        try:
            messages = [
                {"role": "system", "content": prompts.CONTEXT_SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ]
            return self._chat(messages, temperature=0.8)
        except Exception as exc:
            logger.error("Bağlam üretim hatası: %s", exc)
            return f"Bağlam üretilirken hata oluştu: {exc}"

    def generate_question(self, outcome: dict, context: str) -> dict:
        """Verilen öğrenme çıktısı ve bağlam için soru + rubrik üretir."""
        prompt = prompts.build_question_user_prompt(outcome, context)
        try:
            messages = [
                {"role": "system", "content": prompts.QUESTION_SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ]
            raw = self._chat(messages, temperature=0.7)
            result = self._extract_json(raw)
            # Beklenen alanları garantile
            result.setdefault("question_text", "")
            result.setdefault("cognitive_level", "Belirtilmemiş")
            result.setdefault("rubric", [])
            result.setdefault("correct_answer_summary", "")
            return result
        except Exception as exc:
            logger.error("Soru üretim hatası: %s", exc)
            return {
                "question_text": f"Hata: {exc}",
                "cognitive_level": "Belirtilmemiş",
                "rubric": [],
                "correct_answer_summary": "",
            }
