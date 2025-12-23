# Xon AI - Complete Model Documentation

This document provides comprehensive information about all models available in Xon AI, their configurations, and usage guidelines.

## Table of Contents
- [Model Categories](#model-categories)
- [Model Specifications](#model-specifications)
- [Default Configurations](#default-configurations)
- [Advanced Usage](#advanced-usage)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Model Categories

Xon AI supports multiple categories of models, each optimized for specific tasks:

### 1. Chat Models
For general conversation and text generation tasks.

### 2. Code Models
Specialized for code generation, completion, and understanding.

### 3. Vision Models
For image understanding and multimodal tasks.

### 4. Audio Models
For speech recognition and audio processing.

### 5. Embedding Models
For generating text embeddings and semantic search.

## Model Specifications

### Chat Models

| Model Name    | Parameters | Context Window | Best For | Recommended Hardware |
|---------------|------------|----------------|-----------|----------------------|
| llama3-8b     | 8B         | 8K tokens      | General chat, instruction following | 8GB+ RAM, 16GB+ VRAM |
| llama3-70b    | 70B        | 8K tokens      | Advanced reasoning, complex tasks | 64GB+ RAM, 80GB+ VRAM |

### Code Models

| Model Name    | Languages | Context Window | Special Features |
|---------------|-----------|----------------|-------------------|
| codellama-7b  | 20+       | 16K tokens     | Code completion, debugging |
| codellama-13b | 20+       | 16K tokens     | Better understanding, larger context |
| codellama-34b | 20+       | 16K tokens     | Production-level code generation |

### Vision Models

| Model Name | Vision Capabilities | Text Understanding | Best For |
|------------|---------------------|-------------------|-----------|
| llava-7b   | Image understanding | Strong            | Image captioning, VQA |
| llava-13b  | Enhanced resolution | Advanced          | Detailed image analysis |

### Audio Models

| Model Name   | Languages | Real-time | Best For |
|--------------|-----------|-----------|-----------|
| whisper-base | 99+       | No        | General transcription |
| whisper-tiny | 99+       | Yes       | Fast, low-resource |

### Embedding Models

| Model Name       | Dimensions | Languages | Best For |
|------------------|------------|-----------|-----------|
| mxbai-embed-large| 1024       | 100+      | Semantic search, clustering |
| nomic-embed-text | 768        | 100+      | General purpose embeddings |

## Default Configurations

### Chat Model Defaults
```python
{
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2000,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

### Code Model Defaults
```python
{
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 4000,
    "stop": ["```"]
}
```

### Vision Model Defaults
```python
{
    "temperature": 0.3,
    "max_tokens": 1000,
    "image_quality": "high"
}
```

## Advanced Usage

### Model Parameters

#### Temperature
- **Lower (0.1-0.3)**: More focused and deterministic
- **Medium (0.4-0.7)**: Balanced creativity and coherence
- **High (0.8-1.2)**: More creative but potentially less coherent

#### Top-p (Nucleus Sampling)
- Controls diversity via nucleus sampling
- Lower values make output more focused
- Recommended: 0.8-0.95

### Custom Model Configuration

1. Create a custom model config file (`custom_models.yaml`):
```yaml
models:
  - name: "my-code-model"
    model_id: "codellama:7b"
    task_type: "code"
    description: "Custom code model with specific parameters"
    parameters:
      temperature: 0.1
      top_p: 0.95
      max_tokens: 4096
    is_default: false
```

2. Load the configuration:
```python
model_manager = ModelManager(config_path="custom_models.yaml")
```

## Performance Tips

### Memory Optimization
- Use smaller models (7B) for development
- Enable 4-bit or 8-bit quantization
- Use smaller context windows when possible

### Speed Optimization
- Use `whisper-tiny` for real-time transcription
- Reduce `max_tokens` for faster responses
- Use streaming for long outputs

## Troubleshooting

### Common Issues

#### Model Not Found
```bash
Error: Model 'example-model' not found
```
**Solution:**
1. Check if the model exists: `ollama list`
2. Pull the model: `ollama pull model_name`
3. Verify model name in configuration

#### Out of Memory
```bash
CUDA out of memory
```
**Solutions:**
1. Use a smaller model
2. Reduce context length
3. Enable memory optimization flags
4. Close other memory-intensive applications

#### Slow Performance
**Solutions:**
1. Enable GPU acceleration
2. Use a more powerful machine
3. Reduce model size
4. Use batching for multiple requests

## Model Updates

### Adding New Models
1. Add model to `model_manager.py`
2. Update documentation
3. Test thoroughly before marking as default

### Updating Models
```bash
ollama pull model_name:tag
```

## License

All models are subject to their respective licenses. Please check the model cards for specific licensing information.

## Support

For issues and feature requests, please open an issue on our [GitHub repository](https://github.com/yourusername/xon-ai).
