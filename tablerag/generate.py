"""Answer generation, provider-agnostic.

tablerag never hard-depends on a specific LLM vendor. Anything that satisfies
the `Generator` protocol (`generate(prompt: str) -> str`, plus a `model`
label) can drive generation:

- GeminiGenerator      native google-genai (zero extra deps)
- LangChainGenerator   wraps any LangChain chat model (OpenAI, Anthropic,
                       Gemini, Cohere, local, ...)
- CallableGenerator    wraps any `fn(prompt) -> str` (custom SDKs, mocks)

See tablerag.providers for one-line constructors.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger("tablerag")

DEFAULT_MODEL = "gemini-3.1-flash-lite"


@runtime_checkable
class Generator(Protocol):
  """Minimal contract for an answer generator."""

  model: str

  def generate(self, prompt: str) -> str: ...


def get_default_model() -> str:
  """Gemini generation model from env (used by the Gemini provider)."""
  from dotenv import load_dotenv

  load_dotenv()
  return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


class GeminiGenerator:
  """Native google-genai generator."""

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


def _content_to_text(content) -> str:
  """Normalize a LangChain message .content (str or content-block list)."""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for block in content:
      if isinstance(block, str):
        parts.append(block)
      elif isinstance(block, dict):
        parts.append(block.get("text", ""))
    return "".join(parts)
  return str(content)


class LangChainGenerator:
  """Wraps any LangChain chat model (BaseChatModel).

  Works with langchain_openai.ChatOpenAI, langchain_anthropic.ChatAnthropic,
  langchain_google_genai.ChatGoogleGenerativeAI, etc. Duck-typed: only
  `.invoke()` is required, so langchain need not be importable here.
  """

  def __init__(self, chat_model, model: str | None = None) -> None:
    self.chat_model = chat_model
    self.model = model or getattr(chat_model, "model", None) or getattr(
      chat_model, "model_name", chat_model.__class__.__name__
    )

  def generate(self, prompt: str) -> str:
    logger.info(
      "Calling LangChain model=%s (prompt %d chars)", self.model, len(prompt)
    )
    message = self.chat_model.invoke(prompt)
    return _content_to_text(getattr(message, "content", message))


class CallableGenerator:
  """Wraps any `fn(prompt: str) -> str` as a Generator."""

  def __init__(self, fn: Callable[[str], str], model: str = "callable") -> None:
    self._fn = fn
    self.model = model

  def generate(self, prompt: str) -> str:
    logger.info("Calling callable model=%s (prompt %d chars)", self.model, len(prompt))
    return self._fn(prompt)
