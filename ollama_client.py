"""
Ollama API Client Wrapper

This module provides a client for interacting with various LLM APIs including Ollama,
OpenAI, and Anthropic with proper error handling and type safety.
"""
import os
import json
import base64
import logging
import requests
import aiohttp
import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, TypeVar, Type, cast
from dataclasses import dataclass
from http import HTTPStatus

# Type variable for generic response type
T = TypeVar('T')

class ModelError(Exception):
    """Base exception for model-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)

class ModelConnectionError(ModelError):
    """Raised when there's a connection error with the model API"""
    pass

class ModelResponseError(ModelError):
    """Raised when the model API returns an error response"""
    pass

class ModelValidationError(ModelError):
    """Raised when there's a validation error in the request"""
    pass

@dataclass
class ModelResponse:
    """Standardized response from model API"""
    content: str
    status_code: int
    headers: Dict[str, str]
    model: str
    request_id: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ModelProvider(Enum):
    """Supported model providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if the value is a valid ModelProvider"""
        return value in {e.value for e in cls}

class OllamaClient:
    """Client for interacting with various LLM APIs including Ollama, OpenAI, and Anthropic.
    
    This client provides a unified interface for different LLM providers with consistent
    error handling, retries, and response formatting.
    
    Args:
        base_url: Base URL for the API (defaults to provider's default)
        api_key: API key for the service (if required)
        provider: Service provider ('ollama', 'openai', 'anthropic', etc.)
        
    Raises:
        ModelValidationError: If provider is invalid or required parameters are missing
    """
    
    DEFAULT_TIMEOUT = 30.0  # Default timeout in seconds
    MAX_RETRIES = 3  # Maximum number of retries for failed requests
    
    def __init__(
        self, 
        base_url: Optional[str] = None, 
        api_key: Optional[str] = None, 
        provider: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES
    ) -> None:
        # Validate provider
        if provider and not ModelProvider.has_value(provider.lower()):
            raise ModelValidationError(f"Invalid provider: {provider}")
            
        self.provider = ModelProvider(provider.lower()) if provider else ModelProvider.OLLAMA
        self.base_url = base_url or self._get_default_base_url()
        self.api_key = api_key or os.getenv(f"{self.provider.value.upper()}_API_KEY")
        self.timeout = max(1.0, float(timeout))  # Ensure timeout is at least 1 second
        self.max_retries = max(1, int(max_retries))  # Ensure at least 1 retry
        
        # Setup logging
        self.logger = logging.getLogger(f"LLMClient-{self.provider.value}")
        self.logger.setLevel(logging.INFO)
        
        # Initialize session with connection pooling
        self._session = None
        self._session_owner = False
        
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with connection pooling
        
        Returns:
            aiohttp.ClientSession: The session instance
            
        Note:
            The session is created on first access and reused for subsequent requests.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self._get_default_headers(),
                raise_for_status=False
            )
            self._session_owner = True
        return self._session
    
    async def close(self) -> None:
        """Close the underlying session if it was created by this client"""
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._session_owner = False
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for the API requests
        
        Returns:
            Dict[str, str]: Dictionary of default headers
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"XonAI/1.0 (Python; {self.__class__.__name__})"
        }
        
        # Add provider-specific headers
        if self.provider == ModelProvider.OPENAI and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == ModelProvider.ANTHROPIC and self.api_key:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            
        return headers
    
    def _get_default_base_url(self) -> str:
        """Get default base URL based on provider
        
        Returns:
            str: The base URL for the provider
            
        Raises:
            ValueError: If the provider is not supported
        """
        defaults = {
            ModelProvider.OLLAMA: "http://localhost:11434",
            ModelProvider.OPENAI: "https://api.openai.com/v1",
            ModelProvider.ANTHROPIC: "https://api.anthropic.com/v1"
        }
        
        if self.provider not in defaults:
            raise ValueError(f"No default URL configured for provider: {self.provider}")
            
        return defaults[self.provider]
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request with retries and error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments to pass to aiohttp request
            
        Returns:
            Dict[str, Any]: Parsed JSON response
            
        Raises:
            ModelConnectionError: If there's a connection error
            ModelResponseError: If the API returns an error response
            ModelValidationError: If the request is invalid
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        last_exception = None
        
        # Set default timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    "Sending %s request to %s (attempt %d/%d)",
                    method.upper(),
                    url,
                    attempt + 1,
                    self.max_retries + 1
                )
                
                async with self.session.request(method, url, **kwargs) as response:
                    # Handle rate limiting (429)
                    if response.status == 429:
                        retry_after = float(response.headers.get('Retry-After', 5))
                        self.logger.warning(
                            "Rate limited. Retrying after %.1f seconds...",
                            retry_after
                        )
                        await asyncio.sleep(retry_after)
                        continue
                        
                    # Handle server errors (5xx)
                    if response.status >= 500:
                        raise ModelConnectionError(
                            f"Server error: {response.status} {response.reason}",
                            status_code=response.status
                        )
                        
                    # Parse response
                    try:
                        response_data = await response.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError) as e:
                        text = await response.text()
                        raise ModelResponseError(
                            f"Failed to parse JSON response: {e}\nResponse: {text[:500]}",
                            status_code=response.status
                        )
                    
                    # Handle API errors (non-2xx)
                    if response.status >= 400:
                        error_msg = response_data.get('error', {}).get('message', str(response_data))
                        if response.status == 400:
                            raise ModelValidationError(
                                f"Validation error: {error_msg}",
                                status_code=response.status
                            )
                        elif response.status == 401:
                            raise ModelValidationError(
                                "Authentication failed. Please check your API key.",
                                status_code=response.status
                            )
                        elif response.status == 403:
                            raise ModelValidationError(
                                "Permission denied. Check your API key and permissions.",
                                status_code=response.status
                            )
                        elif response.status == 404:
                            raise ModelValidationError(
                                "The requested resource was not found.",
                                status_code=response.status
                            )
                        else:
                            raise ModelResponseError(
                                f"API error: {error_msg}",
                                status_code=response.status
                            )
                    
                    return response_data
                    
            except aiohttp.ClientError as e:
                last_exception = ModelConnectionError(
                    f"Connection error: {str(e)}",
                    status_code=500
                )
                self.logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    str(e)
                )
                
                # Exponential backoff for retries
                if attempt < self.max_retries:
                    backoff = min(2 ** attempt, 10)  # Cap at 10 seconds
                    await asyncio.sleep(backoff)
            
        # If we've exhausted all retries
        if last_exception:
            raise last_exception
            
        raise ModelConnectionError("Request failed after multiple retries")

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from the model provider
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ModelError: If there's an error pulling the model
        """
        try:
            if self.provider == ModelProvider.OLLAMA:
                # For Ollama, use the streaming API for pull progress
                url = f"{self.base_url}/api/pull"
                async with self.session.post(url, json={"name": model_name}) as response:
                    async for line in response.content:
                        if line:
                            self.logger.debug("Pull progress: %s", line.decode().strip())
                    
                    if response.status != 200:
                        error = await response.text()
                        raise ModelError(f"Failed to pull model: {error}")
                        
                return True
                
            # For other providers, just check if the model is available
            await self.list_models()
            return True
            
        except Exception as e:
            self.logger.error("Error pulling model %s: %s", model_name, str(e))
            if not isinstance(e, ModelError):
                raise ModelError(f"Failed to pull model: {str(e)}") from e

            return False
    
    async def chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[Dict[str, Any], None]]:
        """Chat with a model using a conversation interface.
        
        This method provides a unified interface for chat-based completions across different
        model providers. It handles the differences between providers internally.
        
        Args:
            model: The name of the model to use (e.g., 'llama2', 'gpt-4')
            messages: A list of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello!"}]
            stream: If True, returns an async generator that yields response chunks
            **kwargs: Additional parameters for the chat completion
            
        Returns:
            Union[str, AsyncGenerator[Dict[str, Any], None]]: 
                - If stream=False: The full response content as a string
                - If stream=True: An async generator yielding response chunks with the format:
                  {
                      'content': str,         # The content chunk
                      'done': bool,           # Whether this is the final chunk
                      'model': str,           # The model used
                      'usage': Dict[str, int] # Token usage (if available)
                  }
                
        Raises:
            ModelValidationError: If the request is invalid
            ModelConnectionError: If there's a connection error
            ModelResponseError: If the API returns an error response
            
        Example:
            # Non-streaming
            response = await client.chat("llama2", [
                {"role": "user", "content": "Hello!"}
            ])
            
            # Streaming
            async for chunk in client.chat("llama2", [
                {"role": "user", "content": "Hello!"}
            ], stream=True):
                print(chunk['content'], end="")
        """
        if not messages:
            raise ModelValidationError("Messages list cannot be empty")
            
        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ModelValidationError(
                    f"Message at index {i} must have 'role' and 'content' keys"
                )
        
        if stream:
            return self._chat_stream(model, messages, **kwargs)
        return await self._chat_complete(model, messages, **kwargs)
        
    async def _chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Handle non-streaming chat completion
        
        Args:
            model: The model to use for completion
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the API
            
        Returns:
            str: The full response content
            
        Raises:
            ModelError: If there's an error during completion
        """
        try:
            if self.provider == ModelProvider.OLLAMA:
                return await self._ollama_chat_complete(model, messages, **kwargs)
            elif self.provider == ModelProvider.OPENAI:
                return await self._openai_chat_complete(model, messages, **kwargs)
            elif self.provider == ModelProvider.ANTHROPIC:
                return await self._anthropic_chat_complete(model, messages, **kwargs)
            else:
                raise ModelError(f"Chat completion not supported for provider: {self.provider}")
        except Exception as e:
            if not isinstance(e, ModelError):
                raise ModelError(f"Chat completion failed: {str(e)}") from e
            raise
            
    async def _chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming chat completion
        
        Args:
            model: The model to use for completion
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the API
            
        Yields:
            Dict[str, Any]: Response chunks with content and metadata
            
        Raises:
            ModelError: If there's an error during streaming
        """
        try:
            if self.provider == ModelProvider.OLLAMA:
                async for chunk in self._ollama_chat_stream(model, messages, **kwargs):
                    yield chunk
            elif self.provider == ModelProvider.OPENAI:
                async for chunk in self._openai_chat_stream(model, messages, **kwargs):
                    yield chunk
            elif self.provider == ModelProvider.ANTHROPIC:
                async for chunk in self._anthropic_chat_stream(model, messages, **kwargs):
                    yield chunk
            else:
                raise ModelError(f"Streaming not supported for provider: {self.provider}")
        except Exception as e:
            if not isinstance(e, ModelError):
                raise ModelError(f"Streaming failed: {str(e)}") from e
            raise
            
    async def _ollama_chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Handle non-streaming Ollama chat completion
        
        Args:
            model: The Ollama model to use
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the Ollama API
            
        Returns:
            str: The generated response content
            
        Raises:
            ModelError: If there's an error with the Ollama API
        """
        try:
            response = await self._request(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    **kwargs
                }
            )
            
            if not isinstance(response, dict):
                raise ModelResponseError("Invalid response format from Ollama API")
                
            message = response.get("message", {})
            if not isinstance(message, dict) or "content" not in message:
                raise ModelResponseError("Invalid message format in Ollama API response")
                
            return str(message["content"])
            
        except json.JSONDecodeError as e:
            raise ModelResponseError(f"Failed to parse Ollama API response: {str(e)}")
        except aiohttp.ClientError as e:
            raise ModelConnectionError(f"Connection error with Ollama API: {str(e)}")
        except Exception as e:
            if not isinstance(e, ModelError):
                raise ModelError(f"Ollama chat completion failed: {str(e)}") from e
            raise
    
    async def _ollama_chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming Ollama chat completion
        
        Args:
            model: The Ollama model to use
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the Ollama API
            
        Yields:
            Dict[str, Any]: Response chunks with the following structure:
                {
                    'content': str,   # The content chunk
                    'done': bool,     # Whether this is the final chunk
                    'model': str,     # The model used
                    'usage': {        # Token usage (if available)
                        'prompt_tokens': int,
                        'completion_tokens': int,
                        'total_tokens': int
                    }
                }
                
        Raises:
            ModelError: If there's an error with the Ollama API
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        buffer = ""
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise ModelResponseError(
                    f"Ollama API error: {error}",
                    status_code=response.status
                )
            
            async for line in response.content:
                if not line:
                    continue
                    
                # Handle potential partial JSON lines
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content is not None:  # Only yield if content exists
                        yield {
                            "content": content,
                            "done": chunk.get("done", False),
                            "model": chunk.get("model", model),
                            "usage": {
                                "prompt_tokens": chunk.get("prompt_eval_count"),
                                "completion_tokens": chunk.get("eval_count"),
                                "total_tokens": (chunk.get("prompt_eval_count", 0) + 
                                               chunk.get("eval_count", 0))
                            }
                        }
                except json.JSONDecodeError:
                    # Try to recover by appending to buffer
                    buffer += line.decode('utf-8', errors='replace')
                    try:
                        chunk = json.loads(buffer)
                        content = chunk.get("message", {}).get("content", "")
                        if content is not None:
                            yield {
                                "content": content,
                                "done": chunk.get("done", False),
                                "model": chunk.get("model", model),
                                "usage": {
                                    "prompt_tokens": chunk.get("prompt_eval_count"),
                                    "completion_tokens": chunk.get("eval_count"),
                                    "total_tokens": (chunk.get("prompt_eval_count", 0) + 
                                                   chunk.get("eval_count", 0))
                                }
                            }
                        buffer = ""
                    except json.JSONDecodeError:
                        # If we can't parse even with buffer, log and continue
                        self.logger.debug("Failed to parse JSON chunk: %s", line)
                        continue
    
    async def _openai_chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Handle non-streaming OpenAI chat completion
        
        Args:
            model: The OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            str: The generated response content
            
        Raises:
            ModelValidationError: If the request is invalid
            ModelConnectionError: If there's a connection error
            ModelResponseError: If the API returns an error response
        """
        try:
            if not self.api_key:
                raise ModelValidationError("OpenAI API key is required")
                
            response = await self._request(
                "POST",
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    **kwargs
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if not isinstance(response, dict):
                raise ModelResponseError("Invalid response format from OpenAI API")
                
            choices = response.get("choices", [])
            if not choices or not isinstance(choices, list):
                raise ModelResponseError("No choices in OpenAI API response")
                
            message = choices[0].get("message", {})
            if not isinstance(message, dict) or "content" not in message:
                raise ModelResponseError("Invalid message format in OpenAI API response")
                
            return str(message["content"])
            
        except json.JSONDecodeError as e:
            raise ModelResponseError(f"Failed to parse OpenAI API response: {str(e)}")
        except aiohttp.ClientError as e:
            raise ModelConnectionError(f"Connection error with OpenAI API: {str(e)}")
        except Exception as e:
            if not isinstance(e, ModelError):
                raise ModelError(f"OpenAI chat completion failed: {str(e)}") from e
            raise
    
    async def _openai_chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming OpenAI chat completion
        
        Args:
            model: The OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the OpenAI API
            
        Yields:
            Dict[str, Any]: Response chunks with the following structure:
                {
                    'content': str,   # The content chunk
                    'done': bool,     # Whether this is the final chunk
                    'model': str,     # The model used
                    'usage': {        # Token usage (if available)
                        'prompt_tokens': int,
                        'completion_tokens': int,
                        'total_tokens': int
                    }
                }
                
        Raises:
            ModelError: If there's an error with the OpenAI API
        """
        if not self.api_key:
            raise ModelValidationError("OpenAI API key is required")
            
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        buffer = ""
        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error = await response.text()
                raise ModelResponseError(
                    f"OpenAI API error: {error}",
                    status_code=response.status
                )
            
            async for line in response.content:
                if not line or line.strip() == b'data: [DONE]':
                    continue
                    
                # Handle potential partial JSON lines
                try:
                    if line.startswith(b'data: '):
                        line = line[6:].strip()
                    
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict):
                        continue
                        
                    choices = chunk.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        continue
                        
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content is not None:
                        yield {
                            "content": content,
                            "done": False,
                            "model": chunk.get("model", model),
                            "usage": chunk.get("usage", {})
                        }
                        
                except json.JSONDecodeError:
                    # Try to recover by appending to buffer
                    buffer += line.decode('utf-8', errors='replace')
                    try:
                        chunk = json.loads(buffer)
                        if isinstance(chunk, dict):
                            choices = chunk.get("choices", [])
                            if choices and isinstance(choices, list):
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content is not None:
                                    yield {
                                        "content": content,
                                        "done": False,
                                        "model": chunk.get("model", model),
                                        "usage": chunk.get("usage", {})
                                    }
                        buffer = ""
                    except json.JSONDecodeError:
                        # If we can't parse even with buffer, log and continue
                        self.logger.debug("Failed to parse OpenAI stream chunk: %s", line)
                        continue
            
            # Signal completion
            yield {
                "content": "",
                "done": True,
                "model": model,
                "usage": {}
            }
                                    
    async def _anthropic_chat_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Handle non-streaming Anthropic chat completion
        
        Args:
            model: The Anthropic model to use (e.g., 'claude-2', 'claude-instant-1')
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the Anthropic API
            
        Returns:
            str: The generated response content
            
        Raises:
            ModelValidationError: If the request is invalid
            ModelConnectionError: If there's a connection error
            ModelResponseError: If the API returns an error response
            
        Note:
            - System messages should be included as a message with role 'system'
            - The last message must be from the 'user' role
        """
        try:
            if not self.api_key:
                raise ModelValidationError("Anthropic API key is required")
                
            # Convert messages to Anthropic's format
            system = ""
            messages_list = []
            
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ModelValidationError(
                        "Each message must be a dict with 'role' and 'content' keys"
                    )
                
                role = msg["role"]
                if role == "system":
                    system = msg["content"]
                else:
                    # Map 'assistant' role to 'assistant' and everything else to 'user'
                    mapped_role = role if role == "assistant" else "user"
                    messages_list.append({
                        "role": mapped_role,
                        "content": msg["content"]
                    })
            
            # Ensure last message is from user
            if messages_list and messages_list[-1]["role"] == "assistant":
                raise ModelValidationError(
                    "Last message must be from the 'user' role"
                )
            
            payload = {
                "model": model,
                "messages": messages_list,
                "max_tokens": kwargs.pop("max_tokens", 4000),
                "stream": False,
                **kwargs
            }
            
            if system:
                payload["system"] = system
            
            response = await self._request(
                "POST",
                "/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
            )
            
            if not isinstance(response, dict):
                raise ModelResponseError("Invalid response format from Anthropic API")
                
            content = response.get("content", [])
            if not content or not isinstance(content, list):
                raise ModelResponseError("No content in Anthropic API response")
                
            # Join all text content from the response
            return "".join(block.get("text", "") for block in content if block.get("type") == "text")
            
        except json.JSONDecodeError as e:
            raise ModelResponseError(f"Failed to parse Anthropic API response: {str(e)}")
        except aiohttp.ClientError as e:
            raise ModelConnectionError(f"Connection error with Anthropic API: {str(e)}")
        except Exception as e:
            if not isinstance(e, ModelError):
                raise ModelError(f"Anthropic chat completion failed: {str(e)}") from e
            raise
    
    async def _anthropic_chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming Anthropic chat completion
        
        Args:
            model: The Anthropic model to use (e.g., 'claude-2', 'claude-instant-1')
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the Anthropic API
            
        Yields:
            Dict[str, Any]: Response chunks with the following structure:
                {
                    'content': str,   # The content chunk
                    'done': bool,     # Whether this is the final chunk
                    'model': str,     # The model used
                    'usage': {        # Token usage (if available)
                        'input_tokens': int,
                        'output_tokens': int
                    }
                }
                
        Raises:
            ModelError: If there's an error with the Anthropic API
            
        Note:
            - System messages should be included as a message with role 'system'
            - The last message must be from the 'user' role
        """
        if not self.api_key:
            raise ModelValidationError("Anthropic API key is required")
            
        # Convert messages to Anthropic's format
        system = ""
        messages_list = []
        
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ModelValidationError(
                    "Each message must be a dict with 'role' and 'content' keys"
                )
            
            role = msg["role"]
            if role == "system":
                system = msg["content"]
            else:
                # Map 'assistant' role to 'assistant' and everything else to 'user'
                mapped_role = role if role == "assistant" else "user"
                messages_list.append({
                    "role": mapped_role,
                    "content": msg["content"]
                })
        
        # Ensure last message is from user
        if messages_list and messages_list[-1]["role"] == "assistant":
            raise ModelValidationError(
                "Last message must be from the 'user' role"
            )
        
        payload = {
            "model": model,
            "messages": messages_list,
            "max_tokens": kwargs.pop("max_tokens", 4000),
            "stream": True,
            **kwargs
        }
        
        if system:
            payload["system"] = system
        
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream"
        }
        
        buffer = ""
        async with self.session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error = await response.text()
                raise ModelResponseError(
                    f"Anthropic API error: {error}",
                    status_code=response.status
                )
            
            async for line in response.content:
                if not line or line.strip() == b'data: [DONE]':
                    continue
                    
                # Handle potential partial JSON lines
                try:
                    if line.startswith(b'data: '):
                        line = line[6:].strip()
                    
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict):
                        continue
                        
                    # Handle different event types
                    event_type = chunk.get("type")
                    
                    if event_type == "content_block_delta":
                        # Content chunk
                        delta = chunk.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield {
                                "content": text,
                                "done": False,
                                "model": chunk.get("model", model),
                                "usage": {}
                            }
                    
                    elif event_type == "message_delta":
                        # Final message with usage
                        usage = chunk.get("usage", {})
                        yield {
                            "content": "",
                            "done": True,
                            "model": chunk.get("model", model),
                            "usage": {
                                "input_tokens": usage.get("input_tokens"),
                                "output_tokens": usage.get("output_tokens")
                            }
                        }
                        
                except json.JSONDecodeError:
                    # Try to recover by appending to buffer
                    buffer += line.decode('utf-8', errors='replace')
                    try:
                        chunk = json.loads(buffer)
                        if isinstance(chunk, dict):
                            event_type = chunk.get("type")
                            
                            if event_type == "content_block_delta":
                                delta = chunk.get("delta", {})
                                text = delta.get("text", "")
                                if text:
                                    yield {
                                        "content": text,
                                        "done": False,
                                        "model": chunk.get("model", model),
                                        "usage": {}
                                    }
                            
                            elif event_type == "message_delta":
                                usage = chunk.get("usage", {})
                                yield {
                                    "content": "",
                                    "done": True,
                                    "model": chunk.get("model", model),
                                    "usage": {
                                        "input_tokens": usage.get("input_tokens"),
                                        "output_tokens": usage.get("output_tokens")
                                    }
                                }
                        buffer = ""
                    except json.JSONDecodeError:
                        # If we can't parse even with buffer, log and continue
                        self.logger.debug("Failed to parse Anthropic stream chunk: %s", line)
                        continue
    
    def list_models(self) -> List[str]:
        """List all available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    async def generate_embeddings(self, model: str, text: str) -> List[float]:
        """
        Generate embeddings for text
        
        Args:
            model: Model name for embeddings
            text: Text to embed
            
        Returns:
            List[float]: List of floats representing the embedding
            
        Raises:
            ModelError: If there's an error generating embeddings
        """
        try:
            if not self.api_key and self.provider != ModelProvider.OLLAMA:
                raise ModelValidationError(f"API key is required for {self.provider} embeddings")
                
            if self.provider == ModelProvider.OLLAMA:
                response = await self._request(
                    "POST",
                    "/api/embeddings",
                    json={"model": model, "prompt": text}
                )
                if not isinstance(response, dict) or "embedding" not in response:
                    raise ModelResponseError("Invalid response format from embeddings API")
                return response["embedding"]
                
            elif self.provider == ModelProvider.OPENAI:
                response = await self._request(
                    "POST",
                    "/v1/embeddings",
                    json={"model": model, "input": text},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                if not isinstance(response, dict) or "data" not in response or not response["data"]:
                    raise ModelResponseError("Invalid response format from OpenAI embeddings API")
                return response["data"][0]["embedding"]
                
            else:
                raise ModelError(f"Embeddings not supported for provider: {self.provider}")
            
        except json.JSONDecodeError as e:
            raise ModelResponseError(f"Failed to parse embeddings response: {str(e)}")
        except aiohttp.ClientError as e:
            raise ModelConnectionError(f"Connection error during embeddings: {str(e)}")
        except Exception as e:
            if not isinstance(e, ModelError):
                raise ModelError(f"Error generating embeddings: {str(e)}") from e
            raise
    
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
