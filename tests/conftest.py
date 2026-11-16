"""Configuration commune des tests : aucun appel reseau au LLM (le RAG ne lit que son cache versionne)."""
import os

os.environ["RAG_CACHE_ONLY"] = "1"
