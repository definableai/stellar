"""Chat Completions, in two files: mapping.py is the table, model.py is the rest."""

from .model import CHAT, OpenAI

__all__ = ["CHAT", "OpenAI"]
