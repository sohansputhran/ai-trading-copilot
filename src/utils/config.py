"""
Simple configuration for Sprint 1.

WHY THIS FILE EXISTS:
- We need to load the HuggingFace API token from .env file
- Better than hardcoding the token in our code
- Keeps secrets out of Git

HOW IT WORKS:
1. python-dotenv reads .env file
2. os.getenv() gets the API token
3. We export it so other files can import it

USAGE:
    from src.utils.config import HUGGINGFACE_API_TOKEN
"""

import os
from dotenv import load_dotenv

# Load .env file (looks for .env in project root)
load_dotenv()

# Get API token from environment
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Check if token is set
if not HUGGINGFACE_API_TOKEN:
    raise ValueError(
        "HUGGINGFACE_API_TOKEN not found! "
        "Please create a .env file with your HuggingFace token. "
        "Get one free at: https://huggingface.co/settings/tokens "
        "See .env.example for template."
    )
