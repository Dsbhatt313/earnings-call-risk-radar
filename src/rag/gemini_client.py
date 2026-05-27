"""
Gemini API client singleton for the Risk Radar project.

Loads GEMINI_API_KEY from .env once per process, builds a genai.Client,
and caches it via lru_cache. All Gemini-calling code in the project should
use get_client() instead of constructing clients directly.

Pattern matches Day 6's retriever.py (lru_cache singletons for expensive
or stateful resources).
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
import os


# ---- Module constants ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
API_KEY_ENV_VAR = "GEMINI_API_KEY"


# ---- Public API ----------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """
    Return a cached genai.Client instance.

    First call: loads .env, reads GEMINI_API_KEY, constructs the client.
    Subsequent calls: returns the same cached instance.

    Raises:
        FileNotFoundError: if .env is missing at PROJECT_ROOT.
        ValueError: if GEMINI_API_KEY is not set in .env or environment.
    """
    if not ENV_FILE_PATH.exists():
        raise FileNotFoundError(
            f".env file not found at {ENV_FILE_PATH}. "
            f"Create it with: GEMINI_API_KEY=your_key_here"
        )

    load_dotenv(dotenv_path=ENV_FILE_PATH)

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(
            f"{API_KEY_ENV_VAR} not found in environment after loading "
            f"{ENV_FILE_PATH}. Check the .env file contains a line like: "
            f"{API_KEY_ENV_VAR}=AIza..."
        )

    return genai.Client(api_key=api_key)