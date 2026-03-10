"""
Simple configuration for Sprint 1.

WHY THIS FILE EXISTS:
- We need to load the Claude API key from .env file
- Better than hardcoding the key in our code
- Keeps secrets out of Git

HOW IT WORKS:
1. python-dotenv reads .env file
2. os.getenv() gets the API key
3. We export it so other files can import it

USAGE:
    from src.utils.config import ANTHROPIC_API_KEY
"""

import os
from dotenv import load_dotenv

# Load .env file (looks for .env in project root)
load_dotenv()

# Get API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Check if key is set
if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found! "
        "Please create a .env file with your API key. "
        "See .env.example for template."
    )
