"""
Embedding factory.

The capstone requires comparing open-source embeddings (HuggingFace) against a
commercial one (OpenAI). This module hides the differences behind one function
so the rest of the code just asks for an embedding by its friendly name.
"""

from langchain_core.embeddings import Embeddings

import config


def get_embeddings(name: str = config.DEFAULT_EMBEDDING) -> Embeddings:
    """
    Return a LangChain Embeddings object for the given friendly name.

    Names are defined in config.EMBEDDING_MODELS, e.g. "minilm", "bge", "openai".
    HuggingFace models run locally on the VPS CPU; "openai" calls the API.
    """
    if name not in config.EMBEDDING_MODELS:
        valid = ", ".join(config.EMBEDDING_MODELS)
        raise ValueError(f"Unknown embedding '{name}'. Choose one of: {valid}")

    spec = config.EMBEDDING_MODELS[name]
    provider = spec["provider"]
    model_name = spec["model_name"]

    if provider == "huggingface":
        # Imported lazily so the OpenAI-only path does not require torch.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "gemini":
        # Commercial embedding with a free tier (Google Generative AI).
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=config.require_gemini_key(),
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            api_key=config.require_openai_key(),
        )

    raise ValueError(f"Unsupported provider '{provider}' for embedding '{name}'.")


def embedding_label(name: str) -> str:
    return config.EMBEDDING_MODELS[name]["label"]
