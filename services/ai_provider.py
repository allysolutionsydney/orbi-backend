"""
ORBI AI Provider Abstraction Layer
===================================
This is the core of ORBI's model-agnostic architecture.
Users can choose their preferred AI brain — OpenAI, Anthropic Claude,
Google Gemini, or any OpenAI-compatible model (Ollama, Together, Groq, etc.)

Adding a new provider:
1. Create a class that extends AIProvider
2. Implement chat(), stream(), and raw_completion()
3. Register it in PROVIDER_REGISTRY
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional
import os
import logging

logger = logging.getLogger(__name__)


# ── Base Interface ─────────────────────────────────────────────────────────────

class AIProvider(ABC):
    """
    Abstract base class for all AI providers.
    Every provider must implement these three methods.
    """

    provider_name: str = "base"
    default_model: str = "unknown"

    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.8,
    ) -> str:
        """Synchronous chat — returns the full reply string."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.8,
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields token chunks for streaming."""
        pass

    @abstractmethod
    def raw_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Single-turn completion for structured tasks (e.g. memory extraction).
        Takes a plain string prompt, returns a plain string response.
        """
        pass


# ── OpenAI Provider ────────────────────────────────────────────────────────────

class OpenAIProvider(AIProvider):
    """
    Supports: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    Also works with any OpenAI-compatible endpoint (Groq, Together, etc.)
    """

    provider_name = "openai"
    default_model = "gpt-4o"

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,  # for OpenAI-compatible endpoints
    ):
        from openai import OpenAI, AsyncOpenAI

        self.model = model or os.getenv("OPENAI_MODEL", self.default_model)
        key = api_key or os.getenv("OPENAI_API_KEY")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.async_client = AsyncOpenAI(**kwargs)

    def chat(self, messages, max_tokens=1024, temperature=0.8) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    async def stream(self, messages, max_tokens=1024, temperature=0.8):
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def raw_completion(self, prompt, max_tokens=512, temperature=0.2) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )


# ── Anthropic / Claude Provider ────────────────────────────────────────────────

class AnthropicProvider(AIProvider):
    """
    Supports: claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001
    Requires: ANTHROPIC_API_KEY env var
    """

    provider_name = "anthropic"
    default_model = "claude-sonnet-4-6"

    def __init__(self, model: str = None, api_key: str = None):
        try:
            import anthropic
            self.model = model or os.getenv("ANTHROPIC_MODEL", self.default_model)
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=key)
            self.async_client = anthropic.AsyncAnthropic(api_key=key)
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def _split_system(self, messages):
        """Anthropic keeps system prompt separate from message list."""
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                # Anthropic doesn't support consecutive same-role messages
                if filtered and filtered[-1]["role"] == m["role"]:
                    filtered[-1]["content"] += "\n" + m["content"]
                else:
                    filtered.append({"role": m["role"], "content": m["content"]})
        return system, filtered

    def chat(self, messages, max_tokens=1024, temperature=0.8) -> str:
        system, filtered = self._split_system(messages)
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    async def stream(self, messages, max_tokens=1024, temperature=0.8):
        system, filtered = self._split_system(messages)
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system

        async with self.async_client.messages.stream(**kwargs) as s:
            async for text in s.text_stream:
                yield text

    def raw_completion(self, prompt, max_tokens=512, temperature=0.2) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )


# ── Google Gemini Provider ─────────────────────────────────────────────────────

class GeminiProvider(AIProvider):
    """
    Supports: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
    Requires: GOOGLE_API_KEY env var
    """

    provider_name = "gemini"
    default_model = "gemini-1.5-flash"

    def __init__(self, model: str = None, api_key: str = None):
        try:
            import google.generativeai as genai
            self.model_name = model or os.getenv("GOOGLE_MODEL", self.default_model)
            key = api_key or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=key)
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

    def _prepare(self, messages):
        """Convert OpenAI-style messages to Gemini format."""
        system = None
        history = []
        last_user_msg = None

        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                if last_user_msg is not None:
                    history.append({"role": "user", "parts": [last_user_msg]})
                last_user_msg = m["content"] if isinstance(m["content"], str) else str(m["content"])
            elif m["role"] == "assistant":
                if last_user_msg is not None:
                    history.append({"role": "user", "parts": [last_user_msg]})
                    last_user_msg = None
                history.append({"role": "model", "parts": [m["content"]]})

        return system, history, last_user_msg

    def chat(self, messages, max_tokens=1024, temperature=0.8) -> str:
        system, history, last_msg = self._prepare(messages)
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(
            last_msg,
            generation_config=self._genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text

    async def stream(self, messages, max_tokens=1024, temperature=0.8):
        # Gemini streaming is synchronous under the hood; wrap it
        system, history, last_msg = self._prepare(messages)
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(
            last_msg,
            stream=True,
            generation_config=self._genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def raw_completion(self, prompt, max_tokens=512, temperature=0.2) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )


# ── OpenAI-Compatible Provider (Groq, Together, Ollama, etc.) ─────────────────

class OpenAICompatibleProvider(OpenAIProvider):
    """
    Works with any API that mirrors the OpenAI chat completions interface.
    Examples: Groq (groq.com), Together AI, Ollama (local), Mistral, etc.

    Set OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_API_KEY env vars,
    or pass them directly.
    """

    provider_name = "openai_compatible"
    default_model = "llama3-8b-8192"  # Groq default

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        super().__init__(
            model=model or os.getenv("OPENAI_COMPATIBLE_MODEL", self.default_model),
            api_key=api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY", "ollama"),
            base_url=base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1"),
        )
        self.provider_name = "openai_compatible"


# ── Provider Registry & Factory ────────────────────────────────────────────────

PROVIDER_REGISTRY = {
    # OpenAI
    "openai":       (OpenAIProvider, {}),
    "gpt-4o":       (OpenAIProvider, {"model": "gpt-4o"}),
    "gpt-4o-mini":  (OpenAIProvider, {"model": "gpt-4o-mini"}),
    "gpt-4-turbo":  (OpenAIProvider, {"model": "gpt-4-turbo"}),

    # Anthropic
    "anthropic":    (AnthropicProvider, {}),
    "claude":       (AnthropicProvider, {}),
    "claude-opus":  (AnthropicProvider, {"model": "claude-opus-4-6"}),
    "claude-sonnet":(AnthropicProvider, {"model": "claude-sonnet-4-6"}),
    "claude-haiku": (AnthropicProvider, {"model": "claude-haiku-4-5-20251001"}),

    # Google
    "gemini":       (GeminiProvider, {}),
    "gemini-flash": (GeminiProvider, {"model": "gemini-1.5-flash"}),
    "gemini-pro":   (GeminiProvider, {"model": "gemini-1.5-pro"}),

    # OpenAI-compatible (local / third-party)
    "ollama":       (OpenAICompatibleProvider, {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}),
    "groq":         (OpenAICompatibleProvider, {"base_url": "https://api.groq.com/openai/v1"}),
}

# Default if nothing is configured
_DEFAULT_PROVIDER = os.getenv("DEFAULT_AI_PROVIDER", "openai")


def get_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> AIProvider:
    """
    Factory function — returns the correct AIProvider instance.

    Priority:
      1. provider_name argument (from user's profile setting)
      2. DEFAULT_AI_PROVIDER env var
      3. Falls back to OpenAI

    Usage:
      provider = get_provider("anthropic", "claude-sonnet-4-6")
      reply = provider.chat(messages)
    """
    name = (provider_name or _DEFAULT_PROVIDER or "openai").lower().strip()

    if name not in PROVIDER_REGISTRY:
        logger.warning(f"Unknown provider '{name}', falling back to openai")
        name = "openai"

    provider_class, defaults = PROVIDER_REGISTRY[name]
    kwargs = dict(defaults)
    if model:
        kwargs["model"] = model

    try:
        return provider_class(**kwargs)
    except Exception as e:
        logger.error(f"Failed to initialise provider '{name}': {e}. Falling back to openai.")
        return OpenAIProvider()


def list_providers() -> dict:
    """Return available providers and their supported models — used by the settings screen."""
    return {
        "openai": {
            "label": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "requires_key": "OPENAI_API_KEY",
        },
        "anthropic": {
            "label": "Anthropic Claude",
            "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            "requires_key": "ANTHROPIC_API_KEY",
        },
        "gemini": {
            "label": "Google Gemini",
            "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            "requires_key": "GOOGLE_API_KEY",
        },
        "ollama": {
            "label": "Ollama (Local)",
            "models": ["llama3", "mistral", "phi3", "gemma"],
            "requires_key": None,
        },
        "groq": {
            "label": "Groq",
            "models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
            "requires_key": "OPENAI_COMPATIBLE_API_KEY",
        },
    }
