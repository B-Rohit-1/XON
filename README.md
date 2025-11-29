# Xon AI Agent

A powerful AI agent with multimodal capabilities including text, image, and audio processing, powered by local LLMs via Ollama.

## Features

- **Text Generation**: Chat with various language models
- **Image Understanding**: Process and analyze images using LLaVA
- **Audio Processing**: Transcribe audio using Whisper
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
   # For text generation (choose one)
   ollama pull llama3.2:3b  # Smaller, faster
   # or
   ollama pull llama3.2:7b  # Larger, better quality

   # For image understanding
   ollama pull llava:latest

   # For audio transcription
   ollama pull whisper:latest
   ```

## Configuration

Create a `.env` file in the project root with your settings:

```env
# Ollama settings
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3.2:3b

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
   - `image:/path/to/image.jpg` - Process an image
   - `audio:/path/to/audio.mp3` - Transcribe audio
   - `quit` or `exit` - Close the application

## Development

### Project Structure

- `agent.py` - Main agent class handling all interactions
- `ollama_client.py` - Client for communicating with Ollama API
- `multimodal_handler.py` - Handles image and audio processing
- `main.py` - Command-line interface
- `requirements.txt` - Python dependencies
- `.env` - Configuration (not version controlled)

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
