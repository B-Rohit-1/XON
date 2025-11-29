# Supporting Code Files for Xon Ollama Agent

## tools.py

```python
"""
Tool implementations for Xon AI Agent
Implements: Search, Reminders, Graph Queries
"""
import json
from typing import Dict, List, Any
from datetime import datetime
from neo4j import GraphDatabase


class SearchTool:
    """Search tool for knowledge base queries"""
    
    def __init__(self):
        self.knowledge_base = {
            "scholarships": [
                {"name": "Merit Scholarship", "amount": "$5000", "deadline": "March 31"},
                {"name": "Need-Based Grant", "amount": "$3000", "deadline": "April 15"},
                {"name": "STEM Excellence Award", "amount": "$10000", "deadline": "May 1"},
            ],
            "courses": [
                {"name": "Machine Learning", "credits": 4, "instructor": "Dr. Smith"},
                {"name": "Data Structures", "credits": 3, "instructor": "Prof. Johnson"},
                {"name": "Deep Learning", "credits": 4, "instructor": "Dr. Chen"},
            ],
            "jobs": [
                {"title": "Software Engineer Intern", "company": "Tech Corp", "location": "Remote"},
                {"title": "Data Analyst", "company": "Analytics Inc", "location": "NYC"},
                {"title": "ML Engineer", "company": "AI Startup", "location": "SF"},
            ]
        }
    
    def search(self, query: str, context: str = None) -> Dict[str, Any]:
        """Search knowledge base"""
        results = []
        query_lower = query.lower()
        
        if context and context.lower() in self.knowledge_base:
            items = self.knowledge_base[context.lower()]
            for item in items:
                if any(query_lower in str(v).lower() for v in item.values()):
                    results.append(item)
        else:
            for category, items in self.knowledge_base.items():
                for item in items:
                    if any(query_lower in str(v).lower() for v in item.values()):
                        results.append({**item, "category": category})
        
        return {
            "query": query,
            "context": context,
            "results_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }


class ReminderTool:
    """Reminder management tool"""
    
    def __init__(self):
        self.reminders = []
    
    def set_reminder(self, title: str, datetime_str: str, description: str = "") -> Dict[str, Any]:
        """Set a new reminder"""
        reminder = {
            "id": len(self.reminders) + 1,
            "title": title,
            "datetime": datetime_str,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        self.reminders.append(reminder)
        
        return {
            "success": True,
            "reminder": reminder,
            "message": f"Reminder '{title}' set for {datetime_str}"
        }
    
    def list_reminders(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all reminders"""
        if active_only:
            return [r for r in self.reminders if r["status"] == "active"]
        return self.reminders


class GraphQueryTool:
    """Neo4j graph database query tool"""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
        # Sample in-memory graph for demo
        self.graph_data = {
            "users": [
                {"id": "u1", "name": "Alice", "major": "Computer Science"},
                {"id": "u2", "name": "Bob", "major": "Data Science"},
            ],
            "courses": [
                {"id": "c1", "name": "Machine Learning", "credits": 4},
                {"id": "c2", "name": "Algorithms", "credits": 3},
            ],
            "enrollments": [
                {"user": "u1", "course": "c1", "grade": "A"},
                {"user": "u1", "course": "c2", "grade": "B+"},
                {"user": "u2", "course": "c1", "grade": "A-"},
            ]
        }
    
    def query(self, query_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge graph"""
        if query_type == "find_user":
            user_id = params.get("user_id")
            users = [u for u in self.graph_data["users"] if u["id"] == user_id]
            return {"query_type": query_type, "results": users}
        
        elif query_type == "get_user_courses":
            user_id = params.get("user_id")
            enrollments = [e for e in self.graph_data["enrollments"] if e["user"] == user_id]
            courses = []
            for enrollment in enrollments:
                course = next((c for c in self.graph_data["courses"] if c["id"] == enrollment["course"]), None)
                if course:
                    courses.append({**course, "grade": enrollment["grade"]})
            return {"query_type": query_type, "user_id": user_id, "results": courses}
        
        return {"query_type": query_type, "results": [], "error": "Unknown query type"}


class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.tools = {
            "search": SearchTool(),
            "reminders": ReminderTool(),
            "graph_queries": GraphQueryTool()
        }
    
    def get_tool(self, tool_name: str):
        """Get tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())
```

## context_manager.py

```python
"""
Context Manager for page-aware assistance
"""
from typing import Dict, Any, Optional
from datetime import datetime


class ContextManager:
    """Manages page context for context-aware responses"""
    
    PAGE_CONTEXTS = {
        "scholarships": {
            "name": "Scholarships",
            "description": "User browsing scholarship opportunities",
            "relevant_topics": ["applications", "eligibility", "deadlines", "financial aid"],
            "suggested_actions": ["search scholarships", "check deadlines", "view requirements"]
        },
        "jobs": {
            "name": "Jobs",
            "description": "User exploring job postings",
            "relevant_topics": ["applications", "resume", "interviews", "career development"],
            "suggested_actions": ["search jobs", "update resume", "prepare for interview"]
        },
        "courses": {
            "name": "Courses",
            "description": "User viewing available courses",
            "relevant_topics": ["enrollment", "curriculum", "prerequisites", "schedule"],
            "suggested_actions": ["browse courses", "check prerequisites", "enroll"]
        },
        "study_groups": {
            "name": "Study Groups",
            "description": "User in study group discussion",
            "relevant_topics": ["collaboration", "projects", "discussions", "resources"],
            "suggested_actions": ["join group", "share notes", "schedule meeting"]
        },
        "dashboard": {
            "name": "Dashboard",
            "description": "User viewing their profile and progress",
            "relevant_topics": ["progress", "achievements", "settings", "profile"],
            "suggested_actions": ["view progress", "update profile", "check achievements"]
        }
    }
    
    def __init__(self):
        self.current_context = None
        self.context_history = []
    
    def set_context(self, page: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Set current page context"""
        page_lower = page.lower().replace(" ", "_")
        
        if page_lower in self.PAGE_CONTEXTS:
            context = self.PAGE_CONTEXTS[page_lower].copy()
            context["metadata"] = metadata or {}
            context["timestamp"] = datetime.now().isoformat()
            
            self.current_context = context
            self.context_history.append({
                "page": page_lower,
                "timestamp": context["timestamp"]
            })
            
            return context
        
        return {
            "name": page,
            "description": f"User on {page} page",
            "relevant_topics": [],
            "suggested_actions": [],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
    
    def get_context(self) -> Optional[Dict[str, Any]]:
        """Get current context"""
        return self.current_context
    
    def get_context_string(self) -> str:
        """Get context as formatted string for prompt injection"""
        if not self.current_context:
            return "No specific page context."
        
        context = self.current_context
        return f"""Current Page: {context['name']}
Description: {context['description']}
Relevant Topics: {', '.join(context['relevant_topics'])}
Suggested Actions: {', '.join(context['suggested_actions'])}"""
    
    def get_context_history(self) -> list:
        """Get navigation history"""
        return self.context_history
```

## logger.py

```python
"""
Logging and Monitoring for Xon AI Agent
"""
import json
import time
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict


class AgentLogger:
    """Comprehensive logging and monitoring system"""
    
    def __init__(self):
        self.logs = {
            "messages": [],
            "tool_usage": [],
            "metrics": defaultdict(list),
            "errors": [],
            "events": []
        }
        self.start_time = time.time()
    
    def log_message(self, role: str, content: str):
        """Log a conversation message"""
        self.logs["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_tool_use(self, tool_name: str, params: Dict[str, Any]):
        """Log tool execution"""
        self.logs["tool_usage"].append({
            "tool": tool_name,
            "params": params,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_metric(self, metric_name: str, value: float):
        """Log a performance metric"""
        self.logs["metrics"][metric_name].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_error(self, error_message: str):
        """Log an error"""
        self.logs["errors"].append({
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_event(self, event_name: str, data: Dict[str, Any]):
        """Log a system event"""
        self.logs["events"].append({
            "event": event_name,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        metrics = {}
        
        # Latency statistics
        latencies = [m["value"] for m in self.logs["metrics"]["response_latency"]]
        if latencies:
            metrics["latency"] = {
                "p50": sorted(latencies)[len(latencies)//2] if latencies else 0,
                "p95": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else 0,
                "avg": sum(latencies) / len(latencies),
                "count": len(latencies)
            }
        
        # Message counts
        metrics["messages"] = {
            "total": len(self.logs["messages"]),
            "user": len([m for m in self.logs["messages"] if m["role"] == "user"]),
            "assistant": len([m for m in self.logs["messages"] if m["role"] == "assistant"])
        }
        
        # Tool usage
        metrics["tools"] = {
            "total_calls": len(self.logs["tool_usage"]),
            "by_tool": {}
        }
        for tool_use in self.logs["tool_usage"]:
            tool_name = tool_use["tool"]
            metrics["tools"]["by_tool"][tool_name] = metrics["tools"]["by_tool"].get(tool_name, 0) + 1
        
        metrics["errors"] = len(self.logs["errors"])
        metrics["session_duration"] = time.time() - self.start_time
        
        return metrics
    
    def export_logs(self, filepath: str) -> bool:
        """Export all logs to JSON file"""
        try:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "metrics": self.get_metrics(),
                "logs": self.logs
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to export logs: {e}")
            return False
```

## main.py

```python
"""
Main entry point and example usage of Xon AI Agent with Ollama
"""
from agent import XonAgent
import os


def main():
    """Example usage of Xon AI Agent with Ollama"""
    
    print("=" * 70)
    print("XON AI AGENT - OLLAMA MULTIMODAL IMPLEMENTATION")
    print("=" * 70)
    print()
    
    # Initialize agent
    print("Initializing agent...")
    agent = XonAgent()
    print("✓ Agent initialized with Ollama")
    print()
    
    # Set page context
    agent.set_page_context("scholarships")
    print("✓ Context set to: Scholarships page")
    print()
    
    # Example 1: Text conversation
    print("-" * 70)
    print("EXAMPLE 1: Text Conversation")
    print("-" * 70)
    
    test_queries = [
        "What scholarships are available?",
        "Find me machine learning courses",
        "What courses am I enrolled in?"
    ]
    
    for query in test_queries:
        print(f"\nUSER: {query}")
        print("XON: ", end="", flush=True)
        response = agent.chat(query, stream=True)
        print(f"⏱️  Latency: {response.get('latency', 0):.2f}s")
    
    # Example 2: Image analysis (if image exists)
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Image Analysis (Multimodal)")
    print("-" * 70)
    
    test_image = "data/images/test.jpg"
    if os.path.exists(test_image):
        print(f"\nAnalyzing image: {test_image}")
        response = agent.chat_with_image(
            prompt="Describe what you see in this image in detail",
            image_path=test_image
        )
        print(f"Analysis: {response.get('message', 'N/A')}")
        print(f"⏱️  Latency: {response.get('latency', 0):.2f}s")
    else:
        print(f"\n(Skipped - no test image found at {test_image})")
    
    # Example 3: Audio transcription (if audio exists)
    print("\n" + "-" * 70)
    print("EXAMPLE 3: Audio Transcription")
    print("-" * 70)
    
    test_audio = "data/audio/test.mp3"
    if os.path.exists(test_audio):
        print(f"\nTranscribing audio: {test_audio}")
        result = agent.transcribe_audio(test_audio)
        if result.get("success"):
            print(f"Transcription: {result.get('text', 'N/A')}")
            print(f"Language: {result.get('language', 'N/A')}")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print(f"\n(Skipped - no test audio found at {test_audio})")
    
    # Display metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    
    metrics = agent.get_metrics()
    
    if "latency" in metrics:
        print(f"\nLatency:")
        print(f"  P50: {metrics['latency']['p50']:.2f}s")
        print(f"  P95: {metrics['latency']['p95']:.2f}s")
        print(f"  Avg: {metrics['latency']['avg']:.2f}s")
    
    print(f"\nMessages:")
    print(f"  Total: {metrics['messages']['total']}")
    print(f"  User: {metrics['messages']['user']}")
    print(f"  Assistant: {metrics['messages']['assistant']}")
    
    print(f"\nTools:")
    print(f"  Total calls: {metrics['tools']['total_calls']}")
    for tool, count in metrics['tools']['by_tool'].items():
        print(f"  {tool}: {count}")
    
    print(f"\nSession duration: {metrics['session_duration']:.2f}s")
    
    # Export logs
    os.makedirs("logs", exist_ok=True)
    agent.export_logs("logs/conversation_log.json")
    print("\n✓ Logs exported to logs/conversation_log.json")
    print()


if __name__ == "__main__":
    main()
```

## .env.template

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Model Selection
TEXT_MODEL=llama3.2:3b
VISION_MODEL=llava:7b
CODE_MODEL=qwen2.5-coder:7b

# Neo4j Configuration (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Whisper Model
WHISPER_MODEL=base
```

## requirements.txt

```txt
# Ollama Python Client
ollama>=0.1.0

# Multimodal Support
pillow>=10.0.0
opencv-python>=4.8.0
openai-whisper>=20230918

# Text-to-Speech
pyttsx3>=2.90

# Neo4j (optional)
neo4j>=5.0.0

# Core Dependencies
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0

# Optional: Advanced video processing
moviepy>=1.0.3

# Optional: Better audio handling
soundfile>=0.12.0
librosa>=0.10.0
```

## Installation Script (setup.sh)

```bash
#!/bin/bash

echo "============================================"
echo "Xon AI Agent - Ollama Setup Script"
echo "============================================"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✓ Ollama is already installed"
fi

# Create project structure
echo ""
echo "Creating project directories..."
mkdir -p data/images
mkdir -p data/audio
mkdir -p data/video
mkdir -p data/frames
mkdir -p logs

echo "✓ Directories created"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✓ Virtual environment created"
echo ""
echo "To activate, run:"
echo "  source venv/bin/activate  # Mac/Linux"
echo "  venv\\Scripts\\activate    # Windows"

# Install Python packages
echo ""
echo "Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Python packages installed"

# Download Ollama models
echo ""
echo "Downloading Ollama models..."
echo "This may take a while depending on your internet connection..."
echo ""

ollama pull llama3.2:3b
ollama pull llava:7b

echo ""
echo "✓ Models downloaded"

# Copy environment template
if [ ! -f .env ]; then
    cp .env.template .env
    echo "✓ Created .env file (please configure)"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "============================================"
echo "Setup Complete! 🎉"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Review and update .env file if needed"
echo "3. Run the agent: python main.py"
echo ""
```

## Quick Start Commands

```bash
# 1. Clone/download the project
cd xon-ollama-agent

# 2. Make setup script executable (Mac/Linux)
chmod +x setup.sh

# 3. Run setup
./setup.sh

# 4. Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# 5. Run the agent
python main.py
```

## Windows Setup (PowerShell)

```powershell
# Download and install Ollama
# Visit: https://ollama.com/download/windows

# Create directories
New-Item -ItemType Directory -Force -Path data\images
New-Item -ItemType Directory -Force -Path data\audio
New-Item -ItemType Directory -Force -Path data\video
New-Item -ItemType Directory -Force -Path logs

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt

# Download models
ollama pull llama3.2:3b
ollama pull llava:7b

# Create .env from template
Copy-Item .env.template .env

# Run the agent
python main.py
```

---

All files are now ready for immediate use! This complete Ollama implementation gives you a fully local AI agent with multimodal capabilities.
