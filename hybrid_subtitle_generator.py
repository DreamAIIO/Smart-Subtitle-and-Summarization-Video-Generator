"""
Hybrid agent controller that combines deterministic pipelines with selective agent use.
"""
import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional, List, Union

from smolagents import CodeAgent
from smolagents.models import Model

# Import Gemini model
from gemini_model import GeminiModel

# Import pipeline
from pipeline.video_processing_pipeline import VideoProcessingPipeline

# Import tools
from tools.summarization_tool import EnhancedSummarizationTool
from tools.speaker_tool import SpeakerRecognitionTool
from tools.translation_tool import TranslationTool
from tools.quality_verification_tool import QualityVerificationTool
from tools.file_tool import FileOperationsTool

# Import utilities
from utils.error_handling import handle_errors, log_execution
from utils.file_path_utils import PathManager

# Configure logger
logger = logging.getLogger("subtitle_generator")

class HybridSubtitleGenerator:
    """Hybrid system that combines deterministic pipelines with selective agent use."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.5-pro-preview-05-06",
        api_key: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        base_dir: Optional[str] = None
    ):
        """Initialize the hybrid subtitle generator system.
        
        Args:
            model_id: Model ID for Gemini
            api_key: API key for Google Generative AI
            model_kwargs: Additional keyword arguments for the model
            base_dir: Base directory for file operations
        """
        # Initialize path manager
        self.path_manager = PathManager(base_dir)
        
        # Prepare model kwargs
        model_kwargs = model_kwargs or {}
        
        # Initialize the model
        self.model = GeminiModel(
            model_id=model_id,
            api_key=api_key,
            **model_kwargs
        )
        
        # Initialize the deterministic pipeline
        self.pipeline = VideoProcessingPipeline(base_dir)
        
        # Initialize specialized tools
        self.summarization_tool = EnhancedSummarizationTool()
        self.speaker_tool = SpeakerRecognitionTool()
        self.translation_tool = TranslationTool()
        self.quality_tool = QualityVerificationTool()
        self.file_tool = FileOperationsTool()
        
        logger.info(f"Initialized HybridSubtitleGenerator with model {model_id}")
    
    def _create_summarization_agent(self) -> CodeAgent:
        """Create a specialized agent for content summarization.
        
        Returns:
            CodeAgent for summarization
        """
        # Import the simplified summarization approach
        from simplified_summarization import create_simple_summarization_agent
        
        # Create a simplified agent
        return create_simple_summarization_agent(
            model=self.model,
            summarization_tool=self.summarization_tool,
            file_tool=self.file_tool,
            quality_tool=self.quality_tool
        )
    
    def _create_speaker_recognition_agent(self) -> CodeAgent:
        """Create a specialized agent for speaker recognition.
        
        Returns:
            CodeAgent for speaker recognition
        """
        # Import the simplified speaker recognition approach
        from simplified_speaker_recognition import create_simple_speaker_recognition_agent
        
        # Create a simplified agent
        return create_simple_speaker_recognition_agent(
            model=self.model,
            speaker_tool=self.speaker_tool,
            file_tool=self.file_tool,
            quality_tool=self.quality_tool
        )
    
    def _create_translation_agent(self) -> CodeAgent:
        """Create a specialized agent for subtitle translation.
        
        Returns:
            CodeAgent for translation
        """
        # Common imports for the agent
        COMMON_IMPORTS = ["json", "os", "time", "re", "tempfile", "shutil", "uuid", "open", "subprocess"]
        
        agent = CodeAgent(
            tools=[self.translation_tool, self.file_tool, self.quality_tool],
            model=self.model,
            name="translation_agent",
            description="Translates subtitles to different languages.",
            max_steps=2,  # Minimal steps for this task
            planning_interval=None,  # Disable planning to reduce LLM calls
            additional_authorized_imports=COMMON_IMPORTS
        )
        return agent
    
    def _create_recovery_agent(self) -> CodeAgent:
        """Create a recovery agent for handling errors.
        
        Returns:
            CodeAgent for error recovery
        """
        # Include all tools for maximum flexibility in recovery
        ALL_IMPORTS = [
            "json", "os", "time", "re", "tempfile", "shutil", "uuid", "open", "subprocess",
            "pysrt", "ffmpeg", "nltk", "numpy", "datetime", "sys", "traceback",
            "sumy", "sumy.parsers", "sumy.nlp", "sumy.summarizers", "langdetect"
        ]
        
        agent = CodeAgent(
            tools=[
                self.pipeline.video_tool, 
                self.pipeline.audio_tool,
                self.pipeline.transcription_tool,
                self.pipeline.subtitle_tool,
                self.file_tool,
                self.quality_tool
            ],
            model=self.model,
            name="recovery_agent",
            description="Recovers from errors in the video processing pipeline.",
            max_steps=4,  # More steps for complex recovery
            planning_interval=2,  # Enable planning for complex recovery
            additional_authorized_imports=ALL_IMPORTS
        )
        return agent
    
    @log_execution
    @handle_errors(default_return={"error": "Processing error occurred"})
    def process_video(self, video_path: str, operations: List[str], 
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a video with hybrid approach.
        
        Args:
            video_path: Path to the video file
            operations: List of operations to perform
            options: Additional options for processing
            
        Returns:
            Dictionary with results from all operations
        """
        options = options or {}
        results = {}
        
        # Step 1: Run the deterministic pipeline for core operations
        try:
            pipeline_results = self.pipeline.process_video(video_path, options)
            results.update(pipeline_results)
            logger.info("Core pipeline completed")
        except Exception as e:
            logger.error(f"Core pipeline failed: {str(e)}")
            
            # Use recovery agent if core pipeline fails
            try:
                recovery_agent = self._create_recovery_agent()
                recovery_task = f"""
                The core video processing pipeline failed with the following error:
                {str(e)}
                
                Try to recover the process for the video file at {video_path}.
                Focus on generating at least a basic transcript and subtitle file.
                """
                
                recovery_result = recovery_agent.run(recovery_task)
                
                # Try to parse the recovery result
                if isinstance(recovery_result, dict):
                    results.update(recovery_result)
                else:
                    # Try to extract paths from text
                    results.update(self._parse_paths_from_text(recovery_result))
                    
                results["recovery_used"] = True
                logger.info("Recovery agent completed")
            except Exception as recovery_e:
                logger.error(f"Recovery agent failed: {str(recovery_e)}")
                results["error"] = f"Core pipeline failed: {str(e)}. Recovery failed: {str(recovery_e)}"
                return results
        
        # Step 2: Process additional operations with specialized agents
        
        # Summarization (if requested)
        if "summarization" in operations:
            try:
                # Get the transcript path from results
                transcript_path = results.get("transcript_path")
                if transcript_path and os.path.exists(transcript_path):
                    # Import the simplified summarization
                    from simplified_summarization import run_summarization
                    
                    # Create the summarization agent
                    summary_agent = self._create_summarization_agent()
                    
                    # Run the simplified summarization
                    summary_options = {
                        k: v for k, v in options.items() 
                        if k in ["summary_method", "summary_length", "verbose_summarization"]
                    }
                    
                    summary_result = run_summarization(
                        agent=summary_agent,
                        transcript_path=transcript_path,
                        options=summary_options
                    )
                    
                    # Update results with summary information
                    results.update(summary_result)
                    
                    logger.info("Summarization completed")
                else:
                    logger.warning("Skipping summarization: transcript not available")
                    results["error_summary"] = "Transcript not available"
            except Exception as e:
                logger.error(f"Summarization failed: {str(e)}")
                results["error_summary"] = str(e)
        
        # Speaker recognition (if requested)
        if "speaker_recognition" in operations:
            try:
                # Get the audio path from results
                audio_path = results.get("audio_path")
                transcript_path = results.get("transcript_path")
                
                if audio_path and os.path.exists(audio_path):
                    # Import the simplified speaker recognition
                    from simplified_speaker_recognition import run_speaker_recognition
                    
                    # Create the speaker recognition agent
                    speaker_agent = self._create_speaker_recognition_agent()
                    
                    # Prepare options for speaker recognition
                    speaker_options = {
                        k: v for k, v in options.items() 
                        if k in ["expected_speakers", "min_speakers", "max_speakers", "verbose"]
                    }
                    
                    # First try using pyannote directly if authentication is available
                    pyannote_result = None
                    auth_token = options.get("auth_token", os.environ.get("HF_TOKEN"))
                    
                    if auth_token:
                        try:
                            logger.info("Attempting speaker recognition with pyannote directly")
                            pyannote_result = self.speaker_tool.identify_speakers(
                                audio_path=audio_path, 
                                auth_token=auth_token
                            )
                            
                            if pyannote_result and "speakers" in pyannote_result:
                                logger.info("Successfully identified speakers with pyannote")
                                # Store the technical diarization for Gemini enhancement
                                speaker_options["pyannote_result"] = pyannote_result
                        except Exception as e:
                            logger.warning(f"Direct pyannote speaker recognition failed: {str(e)}")
                            # Fallback to agent-based approach
                    
                    # Run the simplified speaker recognition with the agent
                    logger.info("Running agent-based speaker recognition")
                    speaker_result = run_speaker_recognition(
                        agent=speaker_agent,
                        audio_path=audio_path,
                        transcript_path=transcript_path,
                        options=speaker_options
                    )
                    
                    # Combine the results from both approaches if available
                    if pyannote_result and "speakers" in pyannote_result:
                        # We have results from both approaches
                        enhanced_speakers = speaker_result.get("speakers", {})
                        technical_speakers = pyannote_result.get("speakers", {})
                        
                        # If Gemini provided rich information but technical data is more precise
                        # with timestamps, enhance the technical data with Gemini's insights
                        combined_speakers = {}
                        
                        # First, copy technical data
                        for speaker_id, speaker_info in technical_speakers.items():
                            combined_speakers[speaker_id] = speaker_info.copy()
                            
                            # Find matching speaker in enhanced data (if any)
                            matching_enhanced = None
                            for enhanced_id, enhanced_info in enhanced_speakers.items():
                                # Simplified matching by speaker number
                                if (speaker_id.replace("SPEAKER_", "") == 
                                    enhanced_id.replace("SPEAKER_", "")):
                                    matching_enhanced = enhanced_info
                                    break
                            
                            # Enhance with Gemini-provided information
                            if matching_enhanced:
                                # Add role and characteristics if available
                                if matching_enhanced.get("role") != "Unknown speaker":
                                    combined_speakers[speaker_id]["role"] = matching_enhanced["role"]
                                
                                if "No specific characteristics" not in matching_enhanced.get("characteristics", ""):
                                    combined_speakers[speaker_id]["characteristics"] = matching_enhanced["characteristics"]
                        
                        # Update the result with combined data
                        speaker_result["speakers"] = combined_speakers
                    
                    # Update results with speaker information
                    results.update({
                        k: v for k, v in speaker_result.items() 
                        if k in ["speakers", "json_path", "text_path"]
                    })
                    
                    logger.info("Speaker recognition completed")
                else:
                    logger.warning("Skipping speaker recognition: audio not available")
                    results["error_speaker"] = "Audio not available"
            except Exception as e:
                logger.error(f"Speaker recognition failed: {str(e)}")
                results["error_speaker"] = str(e)
        
        # Translation (if requested)
        if "translation" in operations:
            try:
                # Get the subtitle path from results
                subtitle_path = results.get("subtitle_path")
                
                if subtitle_path and os.path.exists(subtitle_path):
                    # Get translation options
                    target_language = options.get("target_language")
                    
                    if not target_language:
                        logger.warning("Skipping translation: no target language specified")
                        results["error_translation"] = "No target language specified"
                    else:
                        # Use the specialized translation agent
                        translation_agent = self._create_translation_agent()
                        translation_task = f"""
                        Translate the subtitles at {subtitle_path} to {target_language}.
                        Save the translated subtitles in the same format as the original.
                        """
                        
                        translation_result = translation_agent.run(translation_task)
                        
                        # Process the translation result
                        if isinstance(translation_result, dict):
                            # The agent returned a dictionary directly
                            results.update({k: v for k, v in translation_result.items() if "translate" in k or "translation" in k})
                        else:
                            # Try to extract translation path from text
                            translation_paths = self._parse_paths_from_text(translation_result, "translation")
                            results.update(translation_paths)
                        
                        logger.info("Translation completed")
                else:
                    logger.warning("Skipping translation: subtitles not available")
                    results["error_translation"] = "Subtitles not available"
            except Exception as e:
                logger.error(f"Translation failed: {str(e)}")
                results["error_translation"] = str(e)
        
        return results
    
    def _parse_paths_from_text(self, text: str, prefix: str = "") -> Dict[str, str]:
        """Extract file paths from text.
        
        Args:
            text: Text to parse
            prefix: Prefix for the path keys
            
        Returns:
            Dictionary with extracted paths
        """
        paths = {}
        
        # Common file patterns to look for
        video_pattern = r'(?:video|video with subtitles|subtitled video)(?:\s+path)?[:\s]+([^\s,]+\.mp4)'
        subtitle_pattern = r'(?:subtitle|subtitles)(?:\s+path)?[:\s]+([^\s,]+\.srt)'
        transcript_pattern = r'(?:transcript)(?:\s+path)?[:\s]+([^\s,]+\.(?:txt|json))'
        summary_pattern = r'(?:summary)(?:\s+path)?[:\s]+([^\s,]+\.(?:txt|json))'
        translation_pattern = r'(?:translate|translation|translated)(?:\s+path)?[:\s]+([^\s,]+\.(?:srt|txt))'
        
        # Extract file paths
        video_matches = re.findall(video_pattern, text, re.IGNORECASE)
        subtitle_matches = re.findall(subtitle_pattern, text, re.IGNORECASE)
        transcript_matches = re.findall(transcript_pattern, text, re.IGNORECASE)
        summary_matches = re.findall(summary_pattern, text, re.IGNORECASE)
        translation_matches = re.findall(translation_pattern, text, re.IGNORECASE)
        
        # Process video matches
        if video_matches and not prefix or prefix == "video":
            path = video_matches[0]
            found_path = self.path_manager.find_nearest_match(path)
            paths["subtitled_video_path"] = found_path or path
        
        # Process subtitle matches
        if subtitle_matches and not prefix or prefix == "subtitle":
            path = subtitle_matches[0]
            found_path = self.path_manager.find_nearest_match(path)
            paths["subtitle_path"] = found_path or path
        
        # Process transcript matches
        if transcript_matches and not prefix or prefix == "transcript":
            path = transcript_matches[0]
            found_path = self.path_manager.find_nearest_match(path)
            paths["transcript_path"] = found_path or path
        
        # Process summary matches
        if summary_matches and not prefix or prefix == "summary":
            path = summary_matches[0]
            found_path = self.path_manager.find_nearest_match(path)
            paths["summary_path"] = found_path or path
        
        # Process translation matches
        if translation_matches and not prefix or prefix == "translation":
            path = translation_matches[0]
            found_path = self.path_manager.find_nearest_match(path)
            paths["translated_subtitle_path"] = found_path or path
        
        # Extract language information if available
        language_match = re.search(r'(?:language|detected language)[:\s]+([a-z]{2})', text, re.IGNORECASE)
        if language_match:
            paths["language"] = language_match.group(1)
        
        # Try to extract summary content if mentioned
        if prefix == "summary" or not prefix:
            summary_content_match = re.search(r'Summary content:\s*(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
            if summary_content_match:
                paths["summary_content"] = summary_content_match.group(1).strip()
        
        return paths