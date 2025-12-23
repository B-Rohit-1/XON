# Xon AI - Setup and Usage Guide

Welcome to Xon AI! This guide will help you set up and start using the Xon AI assistant.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- [Optional] Ollama (if you want to use local models)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd garuda
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Update the `.env` file with your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
     ```
   - Get an API key from [OpenAI](https://platform.openai.com/api-keys)

## Configuration

Edit the `config.yaml` file to customize your setup:

```yaml
# Model configurations
models:
  - name: "gpt-4"
    model_id: "gpt-4"
    task_type: "chat"
    description: "OpenAI's GPT-4 for general chat"
    source: "api"
    parameters:
      temperature: 0.7
      max_tokens: 2000
    is_default: true

# API Configuration
api_config:
  openai_api_key: "${OPENAI_API_KEY}"
  anthropic_api_key: "${ANTHROPIC_API_KEY}"

# Application settings
app:
  default_models:
    chat: "gpt-4"
    code: "gpt-4"
```

## Usage

### Starting the Application

```bash
python main.py
```

### Available Commands

- `help` - Show available commands
- `exit` or `quit` - Exit the application
- `models` - List available models
- `set model <model_name>` - Change the active model
- `clear` - Clear the chat history

### Example Usage

1. Start a chat:
   ```
   > Hello, how can you help me today?
   ```

2. Ask coding questions:
   ```
   > Write a Python function to sort a list of dictionaries by a specific key
   ```

3. Get help with debugging:
   ```
   > I'm getting this error: [paste error message]
   ```

## Advanced Configuration

### Using Local Models with Ollama

1. Install [Ollama](https://ollama.ai/)
2. Pull a model:
   ```bash
   ollama pull llama2
   ```
3. Update `.env` to use Ollama:
   ```
   TEXT_MODEL=llama2
   ```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | - |
| `TEXT_MODEL` | Default text model | `gpt-4` |
| `VISION_MODEL` | Default vision model | `gpt-4-vision-preview` |
| `AUDIO_MODEL` | Default audio model | `whisper-1` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FILE` | Log file path | `logs/app.log` |

## Troubleshooting

### API Key Issues
- Ensure your API key is correctly set in the `.env` file
- Verify the key is active and has sufficient credits
- Check for any typos or extra spaces

### Installation Issues
- Make sure you're using Python 3.8 or higher
- Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`
- Check for any error messages during installation

### Common Errors
- **"Authentication failed"**: Invalid or missing API key
- **Model not found**: The specified model doesn't exist or isn't accessible
- **Connection issues**: Check your internet connection and API endpoint URLs

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

## License

[Specify your license here]

---

For additional help, please [open an issue](https://github.com/yourusername/garuda/issues) in the repository.
