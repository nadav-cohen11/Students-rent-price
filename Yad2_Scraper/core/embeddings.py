import requests
import json
from typing import List
from logger import logger

OLLAMA_API_URL = "http://localhost:11434/api/embeddings"

def generate_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    try:
        payload = {
            "model": model,
            "prompt": text,
            "options": {"temperature": 0}
        }
        
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=60 
        )
        response.raise_for_status()
        
        return response.json().get("embedding", [])
        
    except Exception as e:
        logger.error(f"[Embedding] Error generating embedding: {e}")
        return []