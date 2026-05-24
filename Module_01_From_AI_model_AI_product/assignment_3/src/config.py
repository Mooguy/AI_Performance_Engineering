import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# We use a powerful, fast model suitable for routing and tool calling
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
EXTRACTOR_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"