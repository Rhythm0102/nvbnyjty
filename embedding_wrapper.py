import requests
from typing import List

class EmbeddingWrapper:
    def __init__(self, api_key):
        self.api_key = api_key

    def embed_query(self, text):
        return self.create_embeddings([text])[0]

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        url = "https://api.deepinfra.com/v1/openai/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        embeddings = []
        
        for text in texts:
            data = {
                "input": text,
                "model": "BAAI/bge-large-en-v1.5",
                "encoding_format": "float"
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings.append(embedding)
        
        return embeddings