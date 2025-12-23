"""
Xon AI Agent - Core AI agent implementation
"""
import os
import logging
from typing import Dict, Any, Optional, List

from ollama_client import OllamaClient, ModelProvider
from logger import setup_logger
from config_manager import get_config

class XonAgent:
    """Main Xon AI Agent class"""
    
    def __init__(self, model: str = None):
        """Initialize the Xon AI Agent"""
        self.logger = setup_logger("XonAgent")
        self.config = get_config()
        
        # Always use OpenAI for now
        provider = ModelProvider.OPENAI
        api_key = self.config.api.openai_api_key
            
        self.client = OllamaClient(provider=provider.value, api_key=api_key)
        self.model = model or self.config.model_settings.text_model
        self.logger.info(f"Initialized XonAgent with model: {self.model} using {provider.value}")
    
    async def chat(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Send a chat message to the agent
        
        Args:
            prompt: The user's message or prompt
            model: Optional model to use for this request
            **kwargs: Additional arguments to pass to the model client
            
        Returns:
            The agent's response as a string
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat(
                model=model or self.model,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            self.logger.error(f"Error in chat: {e}", exc_info=True)
            return f"Error: {str(e)}"
