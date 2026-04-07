# config.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=os.getenv("NEBIUS_API_KEY"))

GEN_CONFIG = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "temperature": 0.7,
    "top_p": None,
    "top_k": None,
    "max_tokens": 150
}

JUDGE_CONFIG = {
    "model": "google/gemma-2-9b-it-fast",
    # "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "temperature": 0.1,
    "max_tokens": 500
}