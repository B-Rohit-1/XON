"""
Context Manager for maintaining conversation context
"""
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Message:
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)

class ContextManager:
    """Manages conversation context and memory"""
    
    def __init__(self, max_context_length: int = 10):
        self.logger = logging.getLogger("ContextManager")
        self.max_context_length = max_context_length
        self.context: List[Message] = []
        self.memory: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a message to the context"""
        if metadata is None:
            metadata = {}
            
        message = Message(role=role, content=content, metadata=metadata)
        self.context.append(message)
        
        # Trim context if it gets too long
        if len(self.context) > self.max_context_length:
            self.context = self.context[-self.max_context_length:]
    
    def get_context(self) -> List[Dict[str, Any]]:
        """Get the current conversation context"""
        return [msg.to_dict() for msg in self.context]
    
    def clear_context(self) -> None:
        """Clear the conversation context"""
        self.context = []
    
    def add_to_memory(self, memory_item: Dict[str, Any]) -> None:
        """Add an item to long-term memory"""
        if 'timestamp' not in memory_item:
            memory_item['timestamp'] = datetime.utcnow().isoformat()
        self.memory.append(memory_item)
    
    def search_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search through long-term memory"""
        # In a real implementation, this would use vector search or similar
        # For now, just return the most recent items that match the query
        query = query.lower()
        results = []
        
        for item in reversed(self.memory):
            if query in str(item).lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    def save_to_file(self, filepath: str) -> bool:
        """Save context and memory to a file"""
        try:
            data = {
                'context': [msg.to_dict() for msg in self.context],
                'memory': self.memory,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving context to file: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load context and memory from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.context = [Message.from_dict(msg) for msg in data.get('context', [])]
            self.memory = data.get('memory', [])
            
            return True
            
        except FileNotFoundError:
            self.logger.warning(f"No existing context file found at {filepath}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading context from file: {e}")
            return False
