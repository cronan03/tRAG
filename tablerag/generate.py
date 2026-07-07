"""Gemini generation client for tablerag (mirrors trag's rag.py behavior)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("tablerag")

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def get_default_model() -> str:
  from dotenv import load_dotenv

  load_dotenv()
  return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


class GeminiGenerator:
  def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    self.model = model or get_default_model()
    self._api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not self._api_key:
      raise EnvironmentError("GEMINI_API_KEY is not set.")
    self._client = None

  @property
  def client(self):
    if self._client is None:
      from google import genai

      self._client = genai.Client(api_key=self._api_key)
    return self._client

  def generate(self, prompt: str) -> str:
    logger.info("Calling Gemini model=%s (prompt %d chars)", self.model, len(prompt))
    response = self.client.models.generate_content(
      model=self.model, contents=prompt
    )
    answer = response.text or ""
    if not answer:
      logger.warning("Gemini returned an empty response")
    return answer
