from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_JUDGE_MODEL = "NousResearch/Hermes-4-70B"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    base_url: str = DEFAULT_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    nebius_api_key_env: str = "NEBIUS_API_KEY"
    hf_token_env: str = "HF_TOKEN"

    @property
    def nebius_api_key(self) -> str | None:
        return os.getenv(self.nebius_api_key_env)

    @property
    def hf_token(self) -> str | None:
        return os.getenv(self.hf_token_env)


settings = Settings()


def load_environment() -> None:
    load_dotenv()


def get_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    load_environment()
    return OpenAI(
        api_key=api_key or settings.nebius_api_key,
        base_url=base_url or settings.base_url,
    )


def get_async_client(base_url: str | None = None, api_key: str | None = None) -> AsyncOpenAI:
    load_environment()
    return AsyncOpenAI(
        api_key=api_key or settings.nebius_api_key,
        base_url=base_url or settings.base_url,
    )
