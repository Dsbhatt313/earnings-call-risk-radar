"""
Gemini API client singleton for the Risk Radar project.

Resolves GEMINI_API_KEY from (in order) the process environment, a local
.env file, or Streamlit secrets, builds a genai.Client, and caches it via
lru_cache. All Gemini-calling code in the project should use get_client()
instead of constructing clients directly.

Pattern matches Day 6's retriever.py (lru_cache singletons for expensive
or stateful resources).
"""

from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai


# ---- Module constants ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
API_KEY_ENV_VAR = "GEMINI_API_KEY"


# ---- Key resolution ------------------------------------------------------

def _resolve_api_key() -> str:
    """
    Find GEMINI_API_KEY from whichever source is available.

    Order of precedence:
      1. Process environment (already exported, or set by a previous load).
      2. Local .env file at PROJECT_ROOT (developer machine).
      3. Streamlit secrets (Streamlit Community Cloud deployment).

    Raises:
        RuntimeError: if the key is not found in any source.
    """
    # 1. Already in the environment?
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if api_key:
        return api_key

    # 2. Local .env file (developer machine). Optional — absent on the cloud.
    if ENV_FILE_PATH.exists():
        load_dotenv(dotenv_path=ENV_FILE_PATH)
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if api_key:
            return api_key

    # 3. Streamlit secrets (Streamlit Community Cloud). Imported lazily so the
    #    module still works in non-Streamlit contexts (scripts, tests).
    try:
        import streamlit as st
        if API_KEY_ENV_VAR in st.secrets:
            return st.secrets[API_KEY_ENV_VAR]
    except Exception:
        pass

    raise RuntimeError(
        f"{API_KEY_ENV_VAR} not found. Set it in one of:\n"
        f"  - the process environment ({API_KEY_ENV_VAR}=...),\n"
        f"  - a local .env file at {ENV_FILE_PATH},\n"
        f"  - Streamlit secrets (Manage app -> Settings -> Secrets) as:\n"
        f"      {API_KEY_ENV_VAR} = \"your_key_here\""
    )


# ---- Public API ----------------------------------------------------------

@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """
    Return a cached genai.Client instance.

    First call: resolves GEMINI_API_KEY (env -> .env -> Streamlit secrets)
    and constructs the client. Subsequent calls return the cached instance.

    Raises:
        RuntimeError: if GEMINI_API_KEY cannot be found in any source.
    """
    api_key = _resolve_api_key()
    return genai.Client(api_key=api_key)