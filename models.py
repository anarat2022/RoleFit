"""
Chat model factory -- same provider-swap pattern used in the RAG project:
one function picks the right LangChain chat model based on an env var, so
the agent logic in agent.py never has to know or care which LLM is behind it.
"""

import os

OLLAMA_MODEL = "llama3.2"
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"


def build_chat_model():
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL), temperature=0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (LLM_PROVIDER=openai).")
        return ChatOpenAI(model=OPENAI_MODEL, api_key=api_key, temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (LLM_PROVIDER=gemini).")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key, temperature=0)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'ollama', 'openai', or 'gemini')")
