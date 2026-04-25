"""
Configuration loader — works for both local dev and Streamlit Cloud.

Token resolution order:
  1. .env file  (local development)
  2. st.secrets (Streamlit Cloud)
  3. System environment variable
  4. None       (no AI — rule-based fallback activates automatically)

USAGE:
    from src.utils.config import HUGGINGFACE_API_TOKEN
"""

import os

from dotenv import load_dotenv

# Load .env file if present (local dev). Silently ignored on Streamlit Cloud.
load_dotenv()

# 1. Try .env / system env first
HUGGINGFACE_API_TOKEN: str | None = os.getenv("HUGGINGFACE_API_TOKEN")

# 2. Fall back to st.secrets (Streamlit Cloud deployment)
if not HUGGINGFACE_API_TOKEN:
    try:
        import streamlit as st

        HUGGINGFACE_API_TOKEN = st.secrets.get("HUGGINGFACE_API_TOKEN")
    except Exception:
        # streamlit not installed, or secrets not configured — that's fine
        pass

# Token is None → scanner_agent will catch it at runtime and fall back to
# rule-based analysis. We never raise here so the app always starts up.
if not HUGGINGFACE_API_TOKEN:
    import warnings

    warnings.warn(
        "HUGGINGFACE_API_TOKEN not set. AI scanner disabled — "
        "rule-based fallback will be used automatically. "
        "Add the token to .env (local) or Streamlit Cloud secrets to enable AI.",
        stacklevel=2,
    )

