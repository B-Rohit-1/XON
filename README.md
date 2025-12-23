# Xon AI Code Assistant

A powerful, multimodal AI assistant with specialized models for different tasks, built on top of Ollama's local LLMs.

## Features

- **Multimodal Support**: Text, code, vision, and audio processing
- **Model Management**: Easily switch between different AI models
- **Code Generation**: Generate code from natural language descriptions
- **Code Debugging**: Identify and fix issues in your code
- **Image Understanding**: Analyze and describe images
- **Audio Processing**: Transcribe and understand audio content
- **File Management**: Load and save files directly
- **Interactive Shell**: Simple command-line interface
- **Local-First**: Runs entirely on your machine for privacy

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM (16GB recommended)
- At least 10GB free disk space for models

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/xon-ai-agent.git
   cd xon-ai-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install required system dependencies:
   - **Windows**: Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - **Linux**: `sudo apt-get install ffmpeg`
   - **Mac**: `brew install ffmpeg`

## Setup Models

1. Start the Ollama server (if not already running):
   ```bash
   ollama serve
   ```

2. In a new terminal, download the required models:
   ```bash
   # General purpose chat
   ollama pull llama3:8b
   
   # Code generation and understanding
   ollama pull codellama:7b
   
   # Multimodal (image understanding)
   ollama pull llava:7b
   
   # Audio processing
   ollama pull whisper:base
   
   # Embeddings
   ollama pull mxbai-embed-large
   ```

## Available Models

Xon AI comes with several pre-configured models for different tasks:

### Chat Models
- **llama3-8b**: General purpose chat model, good for most tasks
- **llama3-70b**: Larger, more capable model (requires more RAM/VRAM)

### Code Models
- **codellama-7b**: Specialized for code generation and understanding
- **codellama-13b**: Larger version with better performance

### Vision Models
- **llava-7b**: Multimodal model for image understanding

### Audio Models
- **whisper-base**: Speech recognition and audio processing

### Embedding Models
- **mxbai-embed-large**: High-quality text embeddings for semantic search

## Configuration

Create a `.env` file in the project root with your settings:

```env
# Ollama settings
OLLAMA_HOST=http://localhost:11434

# Default models for different tasks
DEFAULT_CHAT_MODEL=llama3-8b
DEFAULT_CODE_MODEL=codellama-7b
DEFAULT_VISION_MODEL=llava-7b
DEFAULT_AUDIO_MODEL=whisper-base
DEFAULT_EMBEDDING_MODEL=mxbai-embed-large

# Optional: Set to your preferred language (e.g., 'en', 'es', 'fr')
LANGUAGE=en
```

## Usage

1. Start the Xon AI Agent:
   ```bash
   python main.py
   ```

2. Interact with the agent using these commands:
   - `your message` - Chat with text
   - `code your request` - Generate or modify code
   - `debug your code` - Debug code with error messages
   - `load filename` - Load a file for processing
   - `models` - List available models
   - `models set default <task_type> <model_name>` - Set default model for a task
   - `quit` or `exit` - Close the application

### Model Management

List all available models:
```
models
```

Set default model for a specific task type:
```
models set default chat llama3-8b
models set default code codellama-7b
models set default vision llava-7b
models set default audio whisper-base
```

### Examples

Chat with the assistant:
```
Hello! How can you help me today?
```

Generate Python code:
```
code write a function to calculate factorial
```

Debug code:
```
debug def test():
    x = 5
    return x + '2'  # This will cause a type error
```

Analyze an image:
```
load image.jpg
What's in this image?
```

## Development

### Project Structure

- `agent.py` - Main agent class handling all interactions
- `ollama_client.py` - Client for communicating with Ollama API
- `model_manager.py` - Manages different AI models and their configurations
- `coding_agent.py` - Specialized agent for code-related tasks
- `main.py` - Command-line interface
- `requirements.txt` - Python dependencies
- `.env` - Configuration (not version controlled)

### Model Configuration

The `model_manager.py` file contains the `ModelManager` class that handles all model-related operations. It includes:
- Model configurations with parameters
- Default model settings
- Methods to list, add, and modify models
- Support for different task types (chat, code, vision, audio, embedding)

### Adding New Features

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them

3. Run tests (if available):
   ```bash
   python -m pytest
   ```

4. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature-name
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for making local LLMs accessible
- [LLaVA](https://llava-vl.github.io/) for multimodal capabilities
- [Whisper](https://openai.com/research/whisper) for audio transcription
