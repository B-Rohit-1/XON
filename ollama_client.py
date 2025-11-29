"""
Ollama API Client Wrapper
"""
import os
import json
import base64
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the Ollama client"""
        self.base_url = os.getenv("OLLAMA_HOST", base_url)
        self.logger = logging.getLogger("OllamaClient")
        
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama hub"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            # Stream the response to show progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        self.logger.info(f"Downloading {model_name}: {data['status']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def chat(self, 
             model: str, 
             messages: List[Dict[str, Any]], 
             stream: bool = False,
             **kwargs) -> str:
        """
        Chat with a model
        
        Args:
            model: Model name (e.g., 'llama3.2:3b')
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional parameters for the API
            
        Returns:
            Response content as a string
        """
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            response = requests.post(url, json=payload, stream=stream)
            response.raise_for_status()
            
            if stream:
                # Collect all chunks from the stream
                full_response = []
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            full_response.append(chunk["message"]["content"])
                return "".join(full_response)
            else:
                # Handle non-streaming response
                response_data = response.json()
                if isinstance(response_data, dict):
                    if "message" in response_data and "content" in response_data["message"]:
                        return response_data["message"]["content"]
                    elif "response" in response_data:
                        return response_data["response"]
                return str(response_data)
                
                
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise
    
    def generate_embeddings(self, model: str, text: str) -> List[float]:
        """
        Generate embeddings for text
        
        Args:
            model: Model name for embeddings
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """List all available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name}
            )
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False
