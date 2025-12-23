"""
Xon AI - Main Application with Multimodal and Coding Support
"""
import os
import sys
import yaml
import logging
import argparse
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dotenv import load_dotenv

from agent import XonAgent
from coding_agent import CodingAgent
from logger import setup_logger
from ollama_client import OllamaClient, ModelProvider
from model_manager import ModelManager, model_manager, ModelConfig
from config_manager import get_config

class XonAI:
    """Main Xon AI application with multimodal and coding support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Xon AI with all components
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.logger = setup_logger("XonAI")
        self.config = get_config()
        
        # Override with provided config if any
        if config:
            # This is a simplified merge - in a real app, you might want a deep merge
            for key, value in config.items():
                setattr(self.config, key, value)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize agents with models from config
        self.agent = XonAgent(model=self.config.model_settings.text_model)
        self.coding_agent = CodingAgent(model=self.config.model_settings.text_model)
        self.current_file: Optional[str] = None
        
        self.logger.info("Xon AI initialized with full functionality and model support")
        self.contexts: List[Dict[str, Any]] = []
        
        # Ensure we have at least one default model for chat and code tasks
        if not self.model_manager.get_default_model('chat'):
            self.logger.warning("No default chat model found, adding default model")
            chat_model = ModelConfig(
                name='llama3-8b-local',
                model_id='llama3:8b',
                task_type='chat',
                description='Local Llama 3 8B model for chat',
                is_default=True,
                source=ModelProvider.OLLAMA
            )
            self.model_manager.add_model(chat_model)
        
        if not self.model_manager.get_default_model('code'):
            self.logger.warning("No default code model found, adding default model")
            code_model = ModelConfig(
                name='codellama-7b-local',
                model_id='codellama:7b',
                task_type='code',
                description='Local CodeLlama 7B model for code generation',
                is_default=True,
                source=ModelProvider.OLLAMA
            )
            self.model_manager.add_model(code_model)
        
        self.logger.info("Xon AI initialized with full functionality and model support")
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get the appropriate model for a given task type"""
        model = self.model_manager.get_default_model(task_type)
        if not model:
            self.logger.warning(f"No model found for task type: {task_type}. Using default chat model.")
            model = self.model_manager.get_default_model("chat")
        return model.model_id

    def load_file(self, file_path: str) -> bool:
        """Load a file into the appropriate context"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add to contexts (max 5)
            if len(self.contexts) >= 5:
                self.contexts.pop(0)
                
            self.contexts.append({
                "file_path": file_path,
                "content": content,
                "type": self._detect_file_type(file_path)
            })
            
            self.current_file = file_path
            self.logger.info(f"Loaded {file_path} as {self.contexts[-1]['type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            return False

    def _detect_file_type(self, file_path: str) -> str:
        """Detect the type of file based on extension"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.py', '.js', '.java', '.c', '.cpp', '.go', '.rs']:
            return 'code'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            return 'image'
        elif ext in ['.mp3', '.wav', '.ogg']:
            return 'audio'
        elif ext in ['.mp4', '.avi', '.mov']:
            return 'video'
        elif ext in ['.pdf', '.docx', '.txt']:
            return 'document'
        return 'text'

    async def _process_input_async(self, user_input: str) -> str:
        """Async implementation of process_input"""
        try:
            user_input = user_input.strip()
            
            if not user_input:
                return "Please enter a valid command or message."
                
            # Handle commands
            if user_input.startswith('!'):
                command = user_input[1:].split()[0].lower()
                if command == 'help':
                    return self._show_help()
                elif command == 'exit' or command == 'quit':
                    return "exit"
                elif command == 'models':
                    return self._handle_models_command(' '.join(user_input.split()[1:]))
                else:
                    return f"Unknown command: {command}. Type '!help' for available commands."
            
            # Handle file loading
            if user_input.lower().startswith('load '):
                file_path = user_input[5:].strip()
                if not file_path:
                    return "Please specify a file path to load."
                if self.load_file(file_path):
                    return f"✅ Loaded {file_path}"
                else:
                    return f"❌ Failed to load {file_path}"
                
            # Handle vision tasks
            if any(phrase in user_input.lower() for phrase in ['analyze image', 'describe image']):
                response = await self.agent.chat(
                    prompt=user_input,
                    model=self.get_model_for_task("vision"),
                    temperature=0.7
                )
                return response
                
            # Handle audio tasks
            if any(phrase in user_input.lower() for phrase in ['transcribe', 'audio', 'speech']):
                response = await self.agent.chat(
                    prompt=user_input,
                    model=self.get_model_for_task("audio"),
                    temperature=0.7
                )
                return response
                
            # Handle regular chat
            response = await self.agent.chat(
                prompt=user_input,
                model=self.get_model_for_task("chat"),
                temperature=0.7
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}", exc_info=True)
            return f"❌ An error occurred: {str(e)}"

    def process_input(self, user_input: str) -> str:
        """Process user input and route to appropriate handler"""
        import asyncio
        return asyncio.run(self._process_input_async(user_input))

    def _handle_coding_query(self, query: str) -> str:
        """Handle coding-related queries using the coding agent"""
        try:
            # Get relevant context
            code_context = next((ctx for ctx in self.contexts if ctx['type'] == 'code'), None)
            
            if code_context:
                self.coding_agent.context.code = code_context['content']
                self.coding_agent.context.file_path = code_context['file_path']
            
            # Process the query
            if 'debug' in query.lower():
                result = self.coding_agent.debug_code(query)
            else:
                result = self.coding_agent.generate_code(query)
            
            # Format the response
            response = []
            if result.get('code_blocks'):
                for i, block in enumerate(result['code_blocks'], 1):
                    response.append(f"```{block.get('language', 'python')}")
                    response.append(block.get('code', ''))
                    response.append("```")
            elif result.get('response'):
                response.append(result['response'])
                
            return "\n".join(response)
            
        except Exception as e:
            self.logger.error(f"Error in coding agent: {e}", exc_info=True)
            return f"❌ Error processing coding request: {str(e)}"

    def _handle_models_command(self, command: str) -> str:
        """Handle models-related commands"""
        parts = command.lower().split()
        
        if len(parts) == 1:  # Just 'models' command
            return self._list_models()
            
        if len(parts) >= 3 and parts[1] == "set" and parts[2] == "default":
            if len(parts) < 5:
                return "❌ Usage: models set default <task_type> <model_name>"
            task_type = parts[3]
            model_name = parts[4]
            return self._set_default_model(task_type, model_name)
            
        return f"❌ Unknown models command: {command}"
    
    def _list_models(self) -> str:
        """List all available models"""
        output = ["\n📊 Available Models:"]
        
        # Group models by task type
        task_models = {}
        for model in self.model_manager.models.values():
            if model.task_type not in task_models:
                task_models[model.task_type] = []
            task_models[model.task_type].append(model)
        
        # Format output by task type
        for task_type, models in task_models.items():
            output.append(f"\n🔧 {task_type.upper()} Models:")
            for model in models:
                default_marker = " (default)" if model.is_default else ""
                output.append(f"  • {model.name}{default_marker}")
                output.append(f"    ID: {model.model_id}")
                if model.description:
                    output.append(f"    {model.description}")
                if model.parameters:
                    params = ", ".join(f"{k}={v}" for k, v in model.parameters.items())
                    output.append(f"    Parameters: {params}")
                output.append("")
        
        return "\n".join(output)
    
    def _set_default_model(self, task_type: str, model_name: str) -> str:
        """Set the default model for a task type"""
        if self.model_manager.set_default_model(model_name, task_type):
            # Update agent models if needed
            if task_type == "chat":
                self.agent = XonAgent(model=model_name)
            elif task_type == "code":
                self.coding_agent = CodingAgent(model=model_name)
                
            return f"✅ Set default {task_type} model to {model_name}"
        return f"❌ Failed to set default {task_type} model to {model_name}"

def print_help():
    """Print help information for all available commands"""
    print("\n📝 Available Commands:")
    print("  General Commands:")
    print("    help           - Show this help message")
    print("    exit/quit      - Exit the program")
    print("    clear          - Clear the screen")
    print("\n  File Operations:")
    print("    load <file>    - Load a file (code, image, audio, video, document)")
    print("    save [file]    - Save current content to file")
    print("\n  Chat & AI Features:")
    print("    chat <message> - Send a message to the AI (default)")
    print("    code <task>    - Generate or modify code")
    print("    debug [error]  - Debug code with optional error message")
    print("    analyze image  - Analyze an image file")
    print("    transcribe     - Transcribe audio file to text")
    print("\n  Context Management:")
    print("    context show   - Show current context")
    print("    context clear  - Clear current context")
    print("\n  Model Management:")
    print("    models                  - List all available models")
    print("    models set default <type> <name> - Set default model for a task type")
    print("\n💡 You can also type naturally, like:")
    print("  - 'How does this work?'")
    print("  - 'Write a function to sort a list'")
    print("  - 'Explain this code: <paste code>'")
    print("  - 'Help me debug this error: ...'")

def main():
    """Main function to run Xon AI"""
    # Load environment variables from .env file
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description='Xon AI - Multimodal Assistant')
    parser.add_argument('--file', '-f', help='Load a file on startup')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            return 1
    else:
        print(f"ℹ️ Config file not found: {args.config}")
        config = {}
    
    print("🚀 Xon AI - Multimodal Assistant")
    print(f"Using configuration from: {args.config}" if os.path.exists(args.config) else "Using default configuration")
    print("Type 'help' for available commands\n")
    
    # Initialize XonAI with the loaded configuration
    assistant = XonAI(config=config)
    
    # Load file if specified
    if args.file:
        if not assistant.load_file(args.file):
            print(f"❌ Failed to load file: {args.file}")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ('exit', 'quit'):
                print("\n👋 Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print_help()
                continue
                
            # Process the input
            response = assistant.process_input(user_input)
            print(f"\n{response}")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            assistant.logger.error(f"Error in main loop: {e}", exc_info=True)

if __name__ == "__main__":
    sys.exit(main() or 0)