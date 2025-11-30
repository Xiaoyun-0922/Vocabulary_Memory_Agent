import os
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load environment variables from .env if present
load_dotenv()

# Vocabulary file path (can be overridden via VOCAB_FILE in .env)
VOCAB_FILE = os.getenv("VOCAB_FILE", "CET6_vocabulary.txt")


def detect_available_providers() -> Dict[str, str]:
    """Detect available model providers from environment variables.

    Returns a mapping provider_id -> human readable label.
    """

    providers: Dict[str, str] = {}

    # ChatGPT / OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        providers["openai"] = "ChatGPT (OpenAI)"

    # DeepSeek: prefer dedicated variables; fall back to base URL inference
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    deepseek_base = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    if deepseek_key and (os.getenv("DEEPSEEK_API_KEY") or "deepseek" in deepseek_base.lower()):
        providers["deepseek"] = "DeepSeek (OpenAI 兼容接口)"

    return providers


def build_llm(provider: str) -> ChatOpenAI:
    """Build a ChatOpenAI client for the given provider.

    API key and base URL are taken from environment variables (optionally
    loaded from .env).
    """

    if provider == "deepseek":
        # Prefer DeepSeek-specific variables; fall back to OPENAI_* if needed
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
        if not api_key:
            raise ValueError("未检测到 DeepSeek 的 API Key，请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。")

        # DeepSeek is OpenAI-compatible; write values into OPENAI_* for the client
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    else:  # default: OpenAI / ChatGPT
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise ValueError("未检测到 OpenAI 的 API Key，请设置 OPENAI_API_KEY。")

        os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return ChatOpenAI(model=model_name)
