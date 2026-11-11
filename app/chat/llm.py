"""MILESTONE 1 — The LLM.

Return a LangChain chat model for the provider/model set in `.env`.

  LLM_PROVIDER=anthropic  -> `uv add langchain-anthropic`,     use ChatAnthropic
  LLM_PROVIDER=openai     -> `uv add langchain-openai`,        use ChatOpenAI
  LLM_PROVIDER=gemini     -> `uv add langchain-google-genai`,  use ChatGoogleGenerativeAI

The API key is read by the provider package from the environment
(ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY, else GEMINI_API_KEY). Never hard-code keys.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

ANTHROPIC_MAX_TOKENS = 16000  # keeps non-streaming requests within the SDK's 10-minute limit
LLM_TIMEOUT_SECONDS = 600.0


def get_llm() -> BaseChatModel:
    """OUT: a ready-to-use chat model (supports `.invoke(messages)` and `.bind_tools(tools)`).

    The provider package is imported only for the chosen provider, so the other one
    does not need to be installed. Raises ValueError for a missing model or unknown provider.
    """
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    model = settings.llm_model.strip()
    if not model:
        raise ValueError("LLM_MODEL is not set in .env")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature: current Claude models reject non-default sampling parameters (HTTP 400).
        # Explicit max_tokens/timeout: the defaults are the model maximum (128k) and no timeout at all.
        # Arguments use the pydantic aliases (model_name, max_tokens_to_sample, timeout, stop) so mypy can
        # check them; `stop` is passed at its default only because mypy sees aliased fields as required.
        return ChatAnthropic(
            model_name=model,
            max_tokens_to_sample=ANTHROPIC_MAX_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
            stop=None,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # The key comes from GOOGLE_API_KEY, else GEMINI_API_KEY (the package's documented order).
        # No temperature: the package leaves it unset for Gemini 3+, as Google recommends.
        # `request_timeout` is the alias of `timeout` (seconds), used so mypy can check it.
        return ChatGoogleGenerativeAI(model=model, request_timeout=LLM_TIMEOUT_SECONDS)
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected 'anthropic', 'openai' or 'gemini'")
