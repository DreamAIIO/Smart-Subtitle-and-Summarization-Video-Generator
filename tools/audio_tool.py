"""
Audio processing tool for language detection and audio manipulation.
"""
import os
from typing import Dict, Optional, Any, Tuple

import langdetect
from pydub import AudioSegment
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


class AudioProcessingTool(Tool):
    """Tool for audio processing operations."""
    
    name = "audio_processor"
    description = """
    Process audio files, including language detection and audio manipulation.
    """
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Path to the audio file"
        },
        "operation": {
            "type": "string",
            "description": "Operation to perform (detect_language, normalize, etc.)"
        },
        "options": {
            "type": "object",
            "description": "Additional options for the operation",
            "nullable": True
        }
    }
    output_type = "string"
    
    @log_execution
    @handle_errors(default_return="Error processing audio")
    def forward(self, audio_path: str, operation: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Process the audio based on the specified operation.
        
        Args:
            audio_path: Path to the audio file
            operation: Operation to perform
            options: Additional options for the operation
            
        Returns:
            Result of the operation or path to the output file
        """
        options = options or {}
        
        # Validate the audio file
        if not os.path.exists(audio_path):
            return f"Audio file does not exist: {audio_path}"
        
        # Dispatch to the appropriate method based on the operation
        if operation == "detect_language":
            return self._detect_language(audio_path, options)
        elif operation == "normalize_audio":
            return self._normalize_audio(audio_path, options)
        else:
            return f"Unsupported operation: {operation}"
    
    @handle_errors()
    def _detect_language(self, audio_path: str, options: Dict[str, Any]) -> str:
        """Detect the language in an audio file.
        
        Args:
            audio_path: Path to the audio file
            options: Additional options for detection
            
        Returns:
            Detected language code
        """
        # To properly detect language, we need a transcription
        # However, to avoid circular dependencies with the transcription tool,
        # we'll assume we have a small sample transcription from a lightweight model
        
        # In a real implementation, you would use a lightweight ASR model here
        # or call a separate speech recognition service
        
        # For this example, we'll use a simulated approach
        sample_transcription = self._get_sample_transcription(audio_path)
        
        try:
            # Use langdetect to identify the language
            lang = langdetect.detect(sample_transcription)
            
            # Get confidence scores for multiple languages
            lang_probs = langdetect.detect_langs(sample_transcription)
            
            result = {
                "language_code": lang,
                "confidence_scores": str(lang_probs)
            }
            
            return str(result)
        except langdetect.LangDetectException as e:
            raise RuntimeError(f"Language detection error: {str(e)}") from e
    
    @handle_errors()
    def _normalize_audio(self, audio_path: str, options: Dict[str, Any]) -> str:
        """Normalize audio volume and quality.
        
        Args:
            audio_path: Path to the audio file
            options: Additional options for normalization
            
        Returns:
            Path to the normalized audio file
        """
        target_db = options.get("target_db", -20)
        output_path = options.get("output_path", None)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(audio_path)
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(dirname, f"{basename}_normalized.wav")
        
        try:
            # Load audio file with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize to target dB
            change_in_db = target_db - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_db)
            
            # Export the normalized audio
            normalized_audio.export(output_path, format="wav")
            
            return output_path
        except Exception as e:
            raise RuntimeError(f"Audio normalization error: {str(e)}") from e
    
    def _get_sample_transcription(self, audio_path: str) -> str:
        """Get a sample transcription for language detection.
        
        This is a simplified version. In a real implementation, 
        you would use a lightweight ASR model here.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            A sample transcription text
        """
        # In a real implementation, we would use a lightweight ASR model
        # For now, we'll just return a dummy text in English
        # This should be replaced with actual transcription
        return "This is a sample transcription for testing language detection."