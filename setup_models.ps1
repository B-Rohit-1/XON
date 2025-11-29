Write-Host "🚀 Setting up Xon AI Agent models..."
Write-Host "----------------------------------"

# Check if Ollama is installed
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Ollama is not installed. Please install it from https://ollama.ai/" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Ollama is installed" -ForegroundColor Green

# Function to download a model with progress
function Install-Model {
    param (
        [string]$modelName,
        [string]$description
    )
    
    Write-Host "\n📥 Downloading $description ($modelName)..." -ForegroundColor Cyan
    try {
        ollama pull $modelName
        Write-Host "✅ Successfully downloaded $description" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download $description: $_" -ForegroundColor Red
    }
}

# Download required models
Install-Model -modelName "llama3.2:3b" -description "Text Generation Model (Small)"
Install-Model -modelName "llava:latest" -description "Multimodal Model (Image Understanding)"
Install-Model -modelName "whisper:latest" -description "Speech-to-Text Model"

Write-Host "\n✨ Setup complete! You can now run the Xon AI Agent." -ForegroundColor Green
Write-Host "Start the agent with: python main.py" -ForegroundColor Cyan
