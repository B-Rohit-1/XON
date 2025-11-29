"""
Multimodal Handler for processing images, audio, and video
"""
import os
import base64
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import subprocess
import tempfile
import torch

# Try to import optional dependencies
IMPORT_ERROR = None
try:
    # Try importing torch first as it's a common dependency
    import torch
    # Then try transformers and its components
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor
    import torchaudio
    from PIL import Image
    import numpy as np
    try:
        import cv2
    except ImportError:
        # cv2 is optional for basic functionality
        pass
except ImportError as e:
    IMPORT_ERROR = str(e)

class MultimodalHandler:
    """Handle multimodal inputs: images, audio, video"""
    
    def __init__(self):
        """Initialize the multimodal handler"""
        self.logger = logging.getLogger("MultimodalHandler")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        self.whisper_processor = None
        
        if IMPORT_ERROR:
            self.logger.warning(f"Some dependencies not available: {IMPORT_ERROR}")
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
            
        Raises:
            ImportError: If required dependencies are not available
            Exception: For other errors during transcription
        """
        if IMPORT_ERROR:
            raise ImportError(f"Required dependencies not available: {IMPORT_ERROR}")
            
        try:
            # Load model and processor if not already loaded
            if self.whisper_processor is None or self.whisper_model is None:
                self.logger.info("Loading Whisper model...")
                model_name = "openai/whisper-tiny"
                try:
                    # Use AutoProcessor for better compatibility
                    self.whisper_processor = AutoProcessor.from_pretrained(model_name)
                    self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
                    self.whisper_model.eval()  # Set to evaluation mode
                except Exception as e:
                    self.logger.error(f"Failed to load Whisper model: {e}")
                    self.logger.error("This might be due to missing audio dependencies. Try: pip install torchaudio")
                    raise
            
            # Verify audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            self.logger.info(f"Transcribing audio file: {audio_path}")
            
            # Load audio file
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                waveform = waveform.to(self.device)
            except Exception as e:
                self.logger.error(f"Error loading audio file: {e}")
                raise
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=16000
                    ).to(self.device)
                    waveform = resampler(waveform)
                except Exception as e:
                    self.logger.error(f"Error resampling audio: {e}")
                    raise
            
            # Process audio
            try:
                input_features = self.whisper_processor(
                    waveform.squeeze().cpu().numpy(), 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(self.device)
                
                # Generate token ids
                with torch.no_grad():
                    predicted_ids = self.whisper_model.generate(input_features)
                
                # Decode token ids to text
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )
                
                if not transcription:
                    return "[No speech detected]"
                    
                return transcription[0]
                
            except Exception as e:
                self.logger.error(f"Error during transcription: {e}")
                raise
            
        except Exception as e:
            self.logger.error(f"Audio transcription failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    def extract_frames(self, video_path: str, frames_dir: str = None, fps: int = 1) -> List[str]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            frames_dir: Directory to save frames (default: temp directory)
            fps: Frames per second to extract
            
        Returns:
            List of paths to extracted frames
        """
        if IMPORT_ERROR:
            raise ImportError(f"Required dependencies not available: {IMPORT_ERROR}")
            
        try:
            # Create output directory if not exists
            if frames_dir is None:
                frames_dir = tempfile.mkdtemp(prefix="video_frames_")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frame_count = 0
            saved_frames = []
            
            # Get video properties
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(frame_rate / fps)) if fps > 0 else 1
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Save frame at specified interval
                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved_frames.append(frame_path)
                
                frame_count += 1
            
            cap.release()
            return saved_frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            raise
    
    def process_video(self, video_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Process video file - extract frames and analyze
        
        Args:
            video_path: Path to video file
            prompt: Analysis prompt for the video
            
        Returns:
            Analysis results
        """
        try:
            if prompt is None:
                prompt = "Describe the key events in this video."
            
            # Extract frames
            frames = self.extract_frames(video_path)
            self.logger.info(f"Extracted {len(frames)} frames from {video_path}")
            
            # Analyze each frame (simple implementation - could be enhanced)
            analyses = []
            for i, frame_path in enumerate(frames[:10]):  # Limit to first 10 frames for demo
                analysis = {
                    "frame": i,
                    "analysis": f"Frame {i+1} of video"
                }
                analyses.append(analysis)
            
            return {
                "frames_analyzed": len(analyses),
                "analyses": analyses,
                "summary": f"Analyzed {len(analyses)} frames from the video."
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {"error": str(e)}
