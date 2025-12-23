"""
Coding Agent for Xon AI - Specialized in programming tasks
"""
import os
import json
import inspect
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Local imports
from agent import XonAgent
from logger import setup_logger
from config_manager import get_config

@dataclass
class CodeContext:
    """Represents the context for a coding task"""
    file_path: Optional[str] = None
    language: Optional[str] = None
    code: Optional[str] = None
    requirements: List[str] = None
    error: Optional[str] = None
    test_cases: List[Dict] = None

class CodingAgent:
    """Specialized agent for handling coding-related tasks"""
    
    def __init__(self, base_agent: XonAgent = None, model: str = None):
        """Initialize the Coding Agent
        
        Args:
            base_agent: Optional XonAgent instance to use. If not provided, a new one will be created.
            model: The model to use for coding tasks. If not provided, uses default from config.
        """
        self.logger = setup_logger("CodingAgent")
        self.config = get_config()
        
        # Use provided model or get from config
        self.model_name = model or self.config.model_settings.text_model
        
        # Only create a new agent if one isn't provided
        if base_agent is None:
            self.base_agent = XonAgent(model=self.model_name)
            self.logger.info(f"Initialized new CodingAgent with model: {self.model_name}")
        else:
            self.base_agent = base_agent
            self.logger.info(f"Initialized CodingAgent with existing agent using model: {self.model_name}")
        
        # Initialize context
        self.context = CodeContext()
        
        # System prompt optimized for coding tasks
        self.system_prompt = """You are a senior software engineer AI assistant specialized in writing clean, 
efficient, and well-documented code. Follow these guidelines:
1. Only respond to coding-related requests
2. For non-coding questions, say "I'm a coding assistant. Please ask me about programming."
3. Write production-quality code with proper error handling
4. Include type hints and docstrings
5. Follow PEP 8 style guide for Python
5. Optimize for both readability and performance
6. Suggest test cases for your code
7. Explain complex logic with comments
8. Consider security best practices
9. Use modern Python features when appropriate
10. Keep dependencies to a minimum

For any coding task, first analyze the requirements, then provide:
1. A clear solution approach
2. The implementation
3. Example usage
4. Test cases
5. Potential improvements or alternatives
"""
        self.base_agent.system_prompt = self.system_prompt
    
    def set_context(self, context: CodeContext) -> None:
        """Set the coding context for the agent"""
        self.context = context
        if context.file_path:
            self._load_file(context.file_path)
    
    def _load_file(self, file_path: str) -> None:
        """Load a file into the coding context"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.context.code = f.read()
            self.context.language = Path(file_path).suffix[1:].lower()
            self.logger.info(f"Loaded {file_path} ({self.context.language})")
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def generate_code(self, task: str) -> Dict[str, Any]:
        """Generate code based on the given task"""
        self.logger.info(f"Generating code for task: {task[:100]}...")
        
        try:
            # Only process if it's a coding-related request
            if not any(word in task.lower() for word in ['code', 'write', 'create', 'function', 'class', 'script', 'program', 'debug', 'fix']):
                return {"response": "I'm a coding assistant. Please ask me about programming or use the 'code' command."}
                
            # Use the base agent's chat method with the configured model
            response = self.base_agent.chat(task)
            self.logger.debug(f"Raw response: {response}")
            
            # If response is a string, try to parse it as JSON
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    # If it's not JSON, use the string as is
                    pass
            
            # If response is a dictionary with a 'response' key, use that
            if isinstance(response, dict) and 'response' in response:
                response = response['response']
            
            return self._parse_code_response(str(response))
            
        except Exception as e:
            error_msg = f"Error in generate_code: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"response": error_msg, "code_blocks": []}
    
    def debug_code(self, error_message: str = None) -> Dict[str, Any]:
        """Debug existing code with or without an error message"""
        if not self.context.code:
            return {"error": "No code in context to debug"}
            
        prompt = self._build_debug_prompt(error_message)
        response = self.base_agent.chat(prompt)
        return self._parse_code_response(response)
    
    def _build_code_generation_prompt(self, task: str) -> str:
        """Build a prompt for code generation"""
        prompt = [
            f"Task: {task}\n\n",
            "Please provide a complete solution with the following sections:\n"
            "1. Approach: Explain your solution approach\n"
            "2. Implementation: The actual code\n"
            "3. Usage: Example of how to use the code\n"
            "4. Tests: Test cases to verify the implementation\n"
        ]
        
        if self.context.requirements:
            prompt.append("\nRequirements:")
            for req in self.context.requirements:
                prompt.append(f"- {req}")
            prompt.append("\n")
            
        return "".join(prompt)
    
    def _build_debug_prompt(self, error_message: str = None) -> str:
        """Build a prompt for debugging code"""
        prompt = [
            "I need help debugging the following code. "
            "Please analyze it and provide a fix.\n\n"
            f"Language: {self.context.language or 'Not specified'}\n"
            f"Code:\n```{self.context.language or ''}\n{self.context.code}\n```\n\n"
        ]
        
        if error_message:
            prompt.append(f"Error message: {error_message}\n\n")
            
        prompt.append(
            "Please provide:\n"
            "1. The root cause of the issue\n"
            "2. The fixed code with explanations of the changes\n"
            "3. How to prevent this issue in the future\n"
        )
        
        return "".join(prompt)
    
    def _parse_code_response(self, response: str) -> Dict[str, Any]:
        """Parse the AI's response into a structured format"""
        # This is a simple parser that can be enhanced based on the response format
        return {
            "response": response,
            "code_blocks": self._extract_code_blocks(response),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown-formatted text"""
        import re
        code_blocks = []
        pattern = r"```(?:\w*\n)?([\s\S]*?)```"
        
        for i, match in enumerate(re.finditer(pattern, text)):
            code_blocks.append({
                "code": match.group(1).strip(),
                "language": "python"  # Default to python if not specified
            })
            
        return code_blocks

    def format_code(self, code: str = None, language: str = None) -> str:
        """Format code according to language-specific style guidelines"""
        code = code or self.context.code
        language = language or self.context.language or 'python'
        
        if not code:
            return "No code provided to format"
            
        prompt = (
            f"Please format the following {language} code according to best practices. "
            "Ensure proper indentation, spacing, and line breaks. "
            "Return only the formatted code without any additional explanation.\n\n"
            f"```{language}\n{code}\n```"
        )
        
        response = self.base_agent.chat(prompt)
        # Extract just the code block from the response
        code_blocks = self._extract_code_blocks(response)
        return code_blocks[0]["code"] if code_blocks else code

# Example usage
if __name__ == "__main__":
    # Initialize the coding agent
    coder = CodingAgent()
    
    # Example 1: Generate code from scratch
    task = "Create a function that calculates the nth Fibonacci number with memoization"
    result = coder.generate_code(task)
    print("Generated code:")
    print(result["code_blocks"][0]["code"] if result["code_blocks"] else result["response"])
    
    # Example 2: Debug existing code
    coder.context.code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""
    debug_result = coder.debug_code("Division by zero error")
    print("\nDebug result:")
    print(debug_result["response"])
    
    # Example 3: Format code
    messy_code = 'def hello_world():\n    print(\'Hello, world!\')'
    print("\nFormatted code:")
    print(coder.format_code(messy_code))
