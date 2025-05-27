"""
Core video processing pipeline for the subtitle generator application.
Implements a deterministic pipeline for video processing tasks.
"""
import os
import json
import time
import logging
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple

# Import tools
from tools.video_tool import VideoProcessingTool
from tools.audio_tool import AudioProcessingTool
from tools.transcription_tool import EnhancedTranscriptionTool
from tools.subtitle_tool import EnhancedSubtitleTool
from tools.file_tool import FileOperationsTool

# Import utilities
from utils.error_handling import handle_errors, log_execution
from utils.file_path_utils import PathManager

# Configure logger
logger = logging.getLogger("subtitle_generator")

class VideoProcessingPipeline:
    """Pipeline for deterministic video processing operations."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the video processing pipeline.
        
        Args:
            base_dir: Base directory for file operations
        """
        # Initialize path manager
        self.path_manager = PathManager(base_dir)
        
        # Initialize tools
        self.video_tool = VideoProcessingTool()
        self.audio_tool = AudioProcessingTool()
        self.transcription_tool = EnhancedTranscriptionTool()
        self.subtitle_tool = EnhancedSubtitleTool()
        self.file_tool = FileOperationsTool()
        
        logger.info("Initialized VideoProcessingPipeline")
    
    @log_execution
    @handle_errors(default_return={"error": "Pipeline execution failed"})
    def process_video(self, video_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video through the deterministic pipeline.
        
        Args:
            video_path: Path to the video file
            options: Processing options
            
        Returns:
            Dictionary with results from processing
        """
        # Process Streamlit path if needed
        if video_path and '/var/folders/' in video_path:
            logger.info(f"Detected Streamlit temporary path: {video_path}")
            video_path = self.path_manager.handle_streamlit_path(video_path)
            logger.info(f"Using processed path: {video_path}")
        
        # Normalize the video path to avoid path issues
        video_path = os.path.normpath(video_path)
        
        # Validate video path
        is_valid, message = self.path_manager.validate_file_path(video_path)
        if not is_valid:
            # Try to find an alternative path
            alt_path = self.path_manager.find_nearest_match(video_path)
            if alt_path:
                video_path = alt_path
                logger.info(f"Using alternative video path: {alt_path}")
            else:
                return {"error": message}
        
        # Get output paths
        output_paths = self.path_manager.build_output_paths_for_video(video_path)
        
        # Process tracking variables
        results = {"status": "success"}
        results.update(output_paths)
        
        # Step 1: Extract audio
        try:
            audio_path = self._extract_audio(video_path, output_paths["audio"], options)
            results["audio_path"] = audio_path
            logger.info(f"Audio extracted to {audio_path}")
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            results["status"] = "partial"
            results["error_audio"] = str(e)
            # Continue to next steps despite error
        
        # Step 2: Transcribe audio
        try:
            if not "error_audio" in results:
                audio_to_use = results.get("audio_path", output_paths["audio"])
                transcript_path = self._transcribe_audio(audio_to_use, output_paths["transcript_json"], options)
                results["transcript_path"] = transcript_path
                logger.info(f"Transcription saved to {transcript_path}")
            else:
                logger.warning("Skipping transcription due to audio extraction error")
                results["error_transcript"] = "Audio extraction failed"
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            results["status"] = "partial" 
            results["error_transcript"] = str(e)
            # Continue to next steps despite error
        
        # Step 3: Generate subtitles
        try:
            if not "error_transcript" in results:
                transcript_to_use = results.get("transcript_path", output_paths["transcript_json"])
                subtitle_path = self._generate_subtitles(transcript_to_use, output_paths["subtitle"], options)
                results["subtitle_path"] = subtitle_path
                logger.info(f"Subtitles saved to {subtitle_path}")
            else:
                logger.warning("Skipping subtitle generation due to transcription error")
                results["error_subtitles"] = "Transcription failed"
        except Exception as e:
            logger.error(f"Subtitle generation failed: {str(e)}")
            results["status"] = "partial"
            results["error_subtitles"] = str(e)
            # Continue to next steps despite error
        
        # Step 4: Add subtitles to video
        try:
            if not "error_subtitles" in results:
                subtitle_to_use = results.get("subtitle_path", output_paths["subtitle"])
                subtitled_video_path = self._add_subtitles_to_video(video_path, subtitle_to_use, output_paths["subtitled_video"], options)
                results["subtitled_video_path"] = subtitled_video_path
                logger.info(f"Subtitled video saved to {subtitled_video_path}")
            else:
                logger.warning("Skipping subtitle embedding due to subtitle generation error")
                results["error_subtitled_video"] = "Subtitle generation failed"
        except Exception as e:
            logger.error(f"Subtitle embedding failed: {str(e)}")
            results["status"] = "partial"
            results["error_subtitled_video"] = str(e)
        
        # Add language information if available
        if "language" in options:
            results["language"] = options["language"]
        elif hasattr(self, "_detected_language") and self._detected_language:
            results["detected_language"] = self._detected_language
        
        return results
    
    @handle_errors()
    def _extract_audio(self, video_path: str, output_path: str, options: Dict[str, Any]) -> str:
        """Extract audio from video file.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the audio file
            options: Processing options
            
        Returns:
            Path to the extracted audio file
        """
        # Set audio extraction options
        audio_format = options.get("audio_format", "wav")
        audio_options = {
            "format": audio_format,
            "output_path": output_path
        }
        
        # Extract audio using video tool
        return self.video_tool.forward(video_path=video_path, operation="extract_audio", options=audio_options)
    
    @handle_errors()
    def _transcribe_audio(self, audio_path: str, output_path: str, options: Dict[str, Any]) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save the transcript
            options: Processing options
            
        Returns:
            Path to the transcript file
        """
        # Get language option
        language = options.get("language", "auto")
        
        # If language is set to auto, try to detect it
        if language == "auto":
            try:
                language_result = self.audio_tool.forward(
                    audio_path=audio_path, 
                    operation="detect_language", 
                    options={}
                )
                
                # Parse the result to get the language code
                if isinstance(language_result, str) and "language_code" in language_result:
                    import re
                    match = re.search(r"'language_code': '([a-z]{2})'", language_result)
                    if match:
                        language = match.group(1)
                        self._detected_language = language
                        logger.info(f"Detected language: {language}")
            except Exception as e:
                logger.warning(f"Language detection failed: {str(e)}")
                # Continue with auto detection in the transcription tool
        
        # Set transcription options
        transcription_options = {
            "output_format": "json",
            "output_path": output_path,
            "use_multi_pass": options.get("use_multi_pass", True),
            "model_size": options.get("model_size", "medium")
        }
        
        # Transcribe audio
        return self.transcription_tool.forward(
            audio_path=audio_path,
            language=language,
            options=transcription_options
        )
    
    @handle_errors()
    def _generate_subtitles(self, transcript_path: str, output_path: str, options: Dict[str, Any]) -> str:
        """Generate subtitle file from transcript.
        
        Args:
            transcript_path: Path to the transcript file
            output_path: Path to save the subtitle file
            options: Processing options
            
        Returns:
            Path to the subtitle file
        """
        # Set subtitle options
        subtitle_options = {
            "output_path": output_path,
            "max_chars_per_line": options.get("max_chars_per_line", 42),
            "max_words_per_entry": options.get("max_words_per_entry", 14),
            "min_duration": options.get("min_duration", 1.0),
            "max_duration": options.get("max_duration", 7.0),
            "preserve_full_text": options.get("preserve_full_text", True)
        }
        
        # Generate subtitles
        return self.subtitle_tool.forward(
            input_path=transcript_path,
            operation="format_subtitles",
            options=subtitle_options
        )
    
    @handle_errors()
    def _add_subtitles_to_video(self, video_path: str, subtitle_path: str, output_path: str, options: Dict[str, Any]) -> str:
        """Add subtitles to video.
        
        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_path: Path to save the output video
            options: Processing options
            
        Returns:
            Path to the subtitled video
        """
        # Set subtitle embedding options
        subtitle_method = options.get("subtitle_method", "auto")
        
        # Add subtitles to video
        subtitle_options = {
            "subtitle_path": subtitle_path,
            "output_path": output_path,
            "subtitle_method": subtitle_method,
            "verbose": options.get("verbose", True),
            "ffmpeg_path": options.get("ffmpeg_path", None)
        }
        
        return self.video_tool.forward(
            video_path=video_path,
            operation="add_subtitles",
            options=subtitle_options
        )