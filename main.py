"""
Xon AI Agent - Main Entry Point
"""
import os
import sys
import logging
from agent import XonAgent

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the Xon AI Agent"""
    print("🚀 Starting Xon AI Agent...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("main")
    
    try:
        # Initialize the agent
        agent = XonAgent()
        print("✅ Agent initialized successfully!")
        
        # Simple chat interface
        print("\n💬 Type your message or 'quit' to exit")
        print("🔍 Try asking about images, audio, or general questions")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ")
                
                # Check for exit command
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                # Process the input
                if user_input.startswith("image:"):
                    # Handle image processing
                    image_path = user_input[6:].strip()
                    if not os.path.exists(image_path):
                        print(f"❌ Image not found: {image_path}")
                        continue
                        
                    print(f"🖼️  Processing image: {image_path}")
                    result = agent.process_image(image_path)
                    print(f"🤖 {result.get('response', 'No response')}")
                    
                elif user_input.startswith("audio:"):
                    # Handle audio processing
                    audio_path = user_input[6:].strip()
                    if not os.path.exists(audio_path):
                        print(f"❌ Audio file not found: {audio_path}")
                        continue
                        
                    print(f"🎧 Processing audio: {audio_path}")
                    result = agent.process_audio(audio_path)
                    print(f"🔊 Transcription: {result.get('text', 'No transcription')}")
                    
                else:
                    # Handle text chat
                    print("💭 Thinking...")
                    try:
                        response = agent.chat(user_input)
                        if isinstance(response, dict):
                            print(f"🤖 {response.get('response', 'I received an empty response.')}")
                        else:
                            print(f"🤖 {str(response)[:500]}")
                    except Exception as e:
                        logger.error(f"Error in chat: {str(e)}")
                        print("🤖 I encountered an error processing your request. Please try again.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"❌ An error occurred: {e}")
                
    except Exception as e:
        logger.critical(f"Failed to start agent: {e}", exc_info=True)
        print(f"❌ Failed to start agent: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
