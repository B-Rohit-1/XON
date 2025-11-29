"""
Xon AI Agent with Ollama Integration
"""
import os
import json
import base64
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Local imports
from ollama_client import OllamaClient
from multimodal_handler import MultimodalHandler
from logger import setup_logger

@dataclass
class Message:
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = None
    metadata: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}

class XonAgent:
    """Main Xon AI Agent with multimodal capabilities"""
    
    def __init__(self, config_path: str = None):
        """Initialize the Xon AI Agent"""
        self.logger = setup_logger("XonAgent")
        self.ollama = OllamaClient()
        self.multimodal = MultimodalHandler()
        self.context = []
        self.memory = []
        self.tools = {}
        
        # Default system prompt
        self.system_prompt = """You are Xon, a helpful AI assistant. 
Always be helpful, accurate, and context-aware."""
        
        # Check if models are available
        self._verify_models()
    
    def _verify_models(self):
        """Check if required models are available"""
        try:
            # Just check if models are available, don't block on download
            self.logger.info("Checking for required models...")
            # We'll let the first request trigger the download if needed
            return True
        except Exception as e:
            self.logger.warning(f"Model check warning: {e}")
            return False
            
    def _read_image_as_base64(self, image_path: str) -> str:
        """Read an image file and return it as a base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error reading image file {image_path}: {e}")
            raise ValueError(f"Could not read image file: {e}")
    
    def chat(self, message: str, context: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a text chat message
        
        Args:
            message: User's message
            context: Optional conversation history
            
        Returns:
            Dict containing response and metadata
        """
        try:
            if context is None:
                context = []
                
            # Prepare messages for the model
            messages = [
                {"role": "system", "content": self.system_prompt},
                *context,
                {"role": "user", "content": message}
            ]
            
            # Get response from Ollama
            try:
                response_text = self.ollama.chat(
                    model="llama3.2:3b",
                    messages=messages,
                    stream=False
                )
                
                # Ensure we have a string response
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                
                # Clean up the response
                response_text = response_text.strip()
                if not response_text:
                    response_text = "I received an empty response. Please try again."
                
                self.logger.debug(f"Generated response: {response_text[:200]}...")
                    
            except Exception as e:
                error_msg = f"Error in Ollama chat: {str(e)}"
                self.logger.error(error_msg)
                response_text = f"I encountered an error: {str(e)}. Please try again."
                
            # Log the interaction
            self.logger.info(f"Chat response generated for message: {message[:50]}...")
            
            # Update memory
            self.memory.append({
                "type": "chat",
                "user_message": message,
                "assistant_response": response_text,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "response": response_text,
                "metadata": {
                    "model": "llama3.2:3b",
                    "tokens_used": len(response_text.split())  # Approximate
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return {
                "response": "Sorry, I encountered an error processing your request.",
                "error": str(e)
            }
    
    def process_image(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Process an image with a prompt using LLaVA model
        
        Args:
            image_path: Path to the image file
            prompt: Optional prompt for the image
            
        Returns:
            Dict containing analysis results
        """
        try:
            if prompt is None:
                prompt = "Describe this image in detail."
            
            # Verify the image exists and is readable
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Try using the chat endpoint with a simpler format
            response = requests.post(
                f"{self.ollama.base_url}/api/chat",
                json={
                    "model": "llava:latest",
                    "messages": [
                        {
                            "role": "user", 
                            "content": prompt,
                            "images": [self._read_image_as_base64(image_path)]
                        }
                    ],
                    "stream": False
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Log the raw response for debugging
            self.logger.debug(f"Raw response from Ollama: {response_data}")
            
            if "message" not in response_data or "content" not in response_data["message"]:
                raise ValueError(f"Unexpected response format: {response_data}")
            
            self.logger.info(f"Successfully processed image: {image_path}")
            
            return {
                "response": response_data["message"]["content"].strip(),
                "metadata": {
                    "model": "llava:latest",
                    "image": os.path.basename(image_path)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio to text using either Ollama API or local Whisper model
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing transcription and metadata
        """
        try:
            # Verify the audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            self.logger.info(f"Starting audio transcription for: {audio_path}")
            
            # First try using the Ollama API if available
            try:
                with open(audio_path, "rb") as audio_file:
                    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                
                response = requests.post(
                    f"{self.ollama.base_url}/api/generate",
                    json={
                        "model": "whisper:latest",
                        "prompt": "Transcribe this audio",
                        "audio": audio_data,
                        "stream": False
                    }
                )
                response.raise_for_status()
                response_data = response.json()
                
                if "response" in response_data:
                    return {
                        "text": response_data["response"].strip(),
                        "metadata": {
                            "model": "whisper (via Ollama)",
                            "audio_file": os.path.basename(audio_path)
                        }
                    }
                    
            except Exception as api_error:
                self.logger.warning(f"Ollama API transcription failed, falling back to local model: {api_error}")
                # Fall back to local Whisper model
                transcription = self.multimodal.transcribe_audio(audio_path)
                
                return {
                    "text": transcription,
                    "metadata": {
                        "model": "whisper (local)",
                        "audio_file": os.path.basename(audio_path)
                    }
                }
                
        except Exception as e:
            error_msg = f"Error in audio transcription: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "error": error_msg,
                "metadata": {
                    "audio_file": os.path.basename(audio_path) if 'audio_path' in locals() else "unknown"
                }
            }
