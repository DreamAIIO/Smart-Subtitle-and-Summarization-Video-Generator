"""
Updated agent controller with enhanced path handling for Streamlit temp files.
"""
import os
import json
import re
import time
import shutil
import logging
from typing import Dict, Any, Optional, List, Union, Type

from smolagents import CodeAgent
from smolagents.models import Model

# Import model implementations
from smolagents import HfApiModel
try:
    from gemini_model import GeminiModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from tools.video_tool import VideoProcessingTool
from tools.audio_tool import AudioProcessingTool
from tools.transcription_tool import EnhancedTranscriptionTool
from tools.subtitle_tool import EnhancedSubtitleTool
from tools.translation_tool import TranslationTool
from tools.summarization_tool import EnhancedSummarizationTool
from tools.speaker_tool import SpeakerRecognitionTool
from tools.file_tool import FileOperationsTool
from tools.quality_verification_tool import QualityVerificationTool

from utils.error_handling import handle_errors, log_execution
from utils.file_path_utils import PathManager

# Configure logger
logger = logging.getLogger("subtitle_generator")

class SubtitleGeneratorAgent:
    """Agent system for orchestrating the subtitle generation workflow."""
    
    # Define available model types
    MODEL_TYPES = {
        "hf": HfApiModel,
        "gemini": GeminiModel if GEMINI_AVAILABLE else None
    }
    
    # Default model configurations
    DEFAULT_MODELS = {
        "hf": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "gemini": "gemini-2.5-pro-preview-05-06"
    }
    
    def __init__(
        self, 
        model_type: str = "hf",
        model_id: Optional[str] = None, 
        api_key: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        base_dir: Optional[str] = None
    ):
        """Initialize the enhanced subtitle generator agent system.
        
        Args:
            model_type: Type of model to use ("hf" for Hugging Face or "gemini" for Google Gemini)
            model_id: Model ID to use (default will use a predefined model based on model_type)
            api_key: API key for the model service (HF_TOKEN or GEMINI_API_KEY depending on model_type)
            model_kwargs: Additional keyword arguments to pass to the model constructor
            base_dir: Base directory for file operations (default: current working directory)
        
        Raises:
            ValueError: If an unsupported model type is specified or if Gemini is requested but not available
        """
        # Initialize path manager
        self.path_manager = PathManager(base_dir)
        
        # Validate model type
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. Choose from: {list(self.MODEL_TYPES.keys())}")
        
        # Check if Gemini is available
        if model_type == "gemini" and not GEMINI_AVAILABLE:
            raise ValueError(
                "Gemini models are not available. Please install the required dependencies: "
                "pip install google-generativeai"
            )
        
        # Get the model class
        model_class = self.MODEL_TYPES[model_type]
        
        # Get default model ID if none provided
        if model_id is None:
            model_id = self.DEFAULT_MODELS[model_type]
        
        # Handle API key lookup
        if api_key is None:
            if model_type == "hf":
                api_key = os.environ.get("HF_TOKEN")
            elif model_type == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
        
        # Prepare model kwargs
        model_kwargs = model_kwargs or {}
        
        # Initialize the model based on type
        if model_type == "hf":
            self.model = model_class(model_id=model_id, token=api_key, **model_kwargs)
        elif model_type == "gemini":
            self.model = model_class(model_id=model_id, api_key=api_key, **model_kwargs)
        
        # Initialize tools
        self.video_tool = VideoProcessingTool()
        self.audio_tool = AudioProcessingTool()
        self.transcription_tool = EnhancedTranscriptionTool()
        self.subtitle_tool = EnhancedSubtitleTool()
        self.translation_tool = TranslationTool()
        self.summarization_tool = EnhancedSummarizationTool()
        self.speaker_tool = SpeakerRecognitionTool()
        self.file_tool = FileOperationsTool()
        self.quality_tool = QualityVerificationTool()
        
        # Create specialized agents
        self.transcription_agent = self._create_transcription_agent()
        self.subtitle_agent = self._create_subtitle_agent()
        self.summary_agent = self._create_summary_agent()
        
        # Initialize the manager agent
        self.manager_agent = self._create_manager_agent()
    
    def _create_transcription_agent(self) -> CodeAgent:
        """Create a specialized agent for audio extraction and transcription."""
        # Common imports that should be available to most agents
        COMMON_IMPORTS = ["json", "os", "time", "re", "tempfile", "shutil", "uuid", "open", "subprocess"]

        agent = CodeAgent(
            tools=[self.video_tool, self.audio_tool, self.transcription_tool, self.quality_tool, self.file_tool],
            model=self.model,
            name="transcription_agent",
            description="Extracts audio from video and performs accurate transcription with quality verification.",
            max_steps=3,
            planning_interval=2,  # Add planning steps for better quality control
            additional_authorized_imports=COMMON_IMPORTS
        )
        return agent
    
    def _create_subtitle_agent(self) -> CodeAgent:
        """Create a specialized agent for subtitle creation and formatting."""
        # Common imports plus specific ones needed for subtitle processing
        COMMON_IMPORTS = ["json", "os", "time", "re", "tempfile", "shutil", "uuid", "open", "subprocess"]
        SUBTITLE_IMPORTS = ["pysrt", "ffmpeg"]

        agent = CodeAgent(
            tools=[self.subtitle_tool, self.file_tool, self.quality_tool, self.video_tool],
            model=self.model,
            name="subtitle_agent",
            description="Creates well-formatted subtitles with proper timing and adds them to videos.",
            max_steps=4,  # Increased from 3 to allow for more retries
            planning_interval=2,
            additional_authorized_imports=COMMON_IMPORTS + SUBTITLE_IMPORTS
        )
        return agent
    
    def _create_summary_agent(self) -> CodeAgent:
        """Create a specialized agent for content summarization."""
        # Common imports plus specific ones needed for summarization
        COMMON_IMPORTS = ["json", "os", "time", "re", "tempfile", "shutil", "uuid", "open", "subprocess"]
        SUMMARY_IMPORTS = [
            "nltk", "sumy", "sumy.parsers", "sumy.parsers.plaintext", 
            "sumy.nlp", "sumy.nlp.tokenizers", "sumy.nlp.stemmers", 
            "sumy.summarizers", "sumy.summarizers.lsa", "sumy.summarizers.lex_rank", 
            "sumy.summarizers.luhn", "sumy.utils", "langdetect"
        ]

        agent = CodeAgent(
            tools=[self.summarization_tool, self.file_tool, self.quality_tool],
            model=self.model,
            name="summary_agent",
            description="Generates high-quality summaries from transcripts, focusing on key content.",
            max_steps=4,  # Increased from 3 to allow for more retries
            planning_interval=2,
            additional_authorized_imports=COMMON_IMPORTS + SUMMARY_IMPORTS
        )
        return agent
    
    def _create_manager_agent(self) -> CodeAgent:
        """Create a manager agent to orchestrate the workflow with improved error handling."""
        # All imports that might be needed by the manager
        ALL_IMPORTS = [
            "json", "os", "time", "re", "tempfile", "shutil", "uuid", "open", "subprocess",
            "pysrt", "ffmpeg", "nltk", "numpy", "datetime", "sys", "traceback",
            "sumy", "sumy.parsers", "sumy.nlp", "sumy.summarizers", "langdetect"
        ]

        agent = CodeAgent(
            tools=[self.file_tool, self.quality_tool],
            model=self.model,
            managed_agents=[
                self.transcription_agent,
                self.subtitle_agent,
                self.summary_agent
            ],
            max_steps=7,  # Increased from 5 to allow for more recovery steps
            planning_interval=3,
            additional_authorized_imports=ALL_IMPORTS
        )
        return agent
    
    def _handle_streamlit_path(self, streamlit_path: str) -> str:
        """Handle paths from Streamlit temporary uploads.
        
        Args:
            streamlit_path: Original path from Streamlit
            
        Returns:
            Working path that can be accessed by tools
        """
        if not streamlit_path:
            logger.warning("Empty streamlit path provided")
            return streamlit_path
            
        try:
            # Normalize the path to remove any issues
            norm_path = os.path.normpath(streamlit_path)
            
            # Check if accessible
            is_valid, message = self.path_manager.validate_file_path(norm_path)
            if is_valid:
                logger.info(f"Streamlit path is valid and accessible: {norm_path}")
                return norm_path
                
            logger.warning(f"Streamlit path issue: {message}")
            
            # If not accessible, copy to our temp directory
            temp_dir = os.path.join(self.path_manager.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            filename = os.path.basename(norm_path)
            new_path = os.path.join(temp_dir, filename)
            
            # Try to copy the file
            try:
                shutil.copy2(norm_path, new_path)
                logger.info(f"Copied Streamlit file to accessible location: {new_path}")
                return new_path
            except Exception as e:
                logger.error(f"Failed to copy Streamlit file: {str(e)}")
                
                # As a last resort, try to create a symlink
                try:
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.symlink(norm_path, new_path)
                    logger.info(f"Created symlink to Streamlit file: {new_path}")
                    return new_path
                except Exception as link_err:
                    logger.error(f"Failed to create symlink: {str(link_err)}")
            
            # If all else fails, return the original path
            return norm_path
        except Exception as e:
            logger.error(f"Error handling Streamlit path: {str(e)}")
            return streamlit_path  # Return original path as fallback
    
    @log_execution
    @handle_errors()
    def process_video(self, video_path: str, operations: List[str], 
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a video with multiple operations with improved error recovery.
        
        Args:
            video_path: Path to the video file
            operations: List of operations to perform
            options: Additional options for processing
            
        Returns:
            Dictionary with results from all operations
        """
        options = options or {}
        
        # Process Streamlit path if needed
        if video_path and '/var/folders/' in video_path:
            logger.info(f"Detected Streamlit temporary path: {video_path}")
            video_path = self._handle_streamlit_path(video_path)
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
        
        # Build output paths
        output_paths = self.path_manager.build_output_paths_for_video(video_path)
        
        # Define the task for the manager agent with explicit error handling guidance
        task = self._build_multi_operation_task(video_path, operations, options, output_paths)
        
        # Run the manager agent with retries for resilience
        max_retries = 3  # Increased from 2 to allow for more retry attempts
        for attempt in range(max_retries + 1):
            try:
                result = self.manager_agent.run(task)
                # Parse the result
                parsed_result = self._parse_agent_result(result)
                
                # Enhance result with output paths information
                parsed_result.update(output_paths)
                
                # Check if we have a valid result with expected keys
                if self._validate_result(parsed_result, operations):
                    return parsed_result
                
                # If validation fails and we have more attempts, continue to retry
                if attempt < max_retries:
                    logger.warning(f"Incomplete result on attempt {attempt+1}, retrying...")
                    time.sleep(2)  # Add a small delay before retrying
                    continue
                else:
                    # No more retries, return what we have with a warning
                    parsed_result["warning"] = "Results may be incomplete"
                    return parsed_result
                    
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Error on attempt {attempt+1}: {str(e)}, retrying...")
                    time.sleep(2)  # Add a small delay before retrying
                    continue
                else:
                    # Last attempt failed, return error
                    return {"error": str(e)}

    def _validate_result(self, result: Dict[str, Any], operations: List[str]) -> bool:
        """Validate that the result contains expected keys based on operations."""
        if not isinstance(result, dict):
            return False
            
        # For subtitles operation, we expect subtitle-related files
        if "subtitles" in operations:
            if not any(key in result for key in ["subtitle_path", "subtitled_video_path"]):
                # Try looking up actual files on disk using the path manager
                for key in ["subtitle_path", "subtitled_video_path"]:
                    if key in result:
                        path = result[key]
                        found_path = self.path_manager.find_nearest_match(path)
                        if found_path:
                            result[key] = found_path
                
                # Check again after attempting file lookup
                if not any(key in result for key in ["subtitle_path", "subtitled_video_path"]):
                    return False
                
        # For summarization, we expect summary-related files
        if "summarization" in operations:
            if not any(key in result for key in ["summary_path", "summary_content"]):
                # Try looking up actual files on disk
                if "summary_path" in result:
                    path = result["summary_path"]
                    found_path = self.path_manager.find_nearest_match(path)
                    if found_path:
                        result["summary_path"] = found_path
                
                # Check again after attempting file lookup
                if not any(key in result for key in ["summary_path", "summary_content"]):
                    return False
        
        return True
    
    def _build_multi_operation_task(self, video_path: str, operations: List[str], 
                                  options: Dict[str, Any], output_paths: Dict[str, str]) -> str:
        """Build the task with additional error handling instructions."""
        operations_text = "\n".join([f"- {op}" for op in operations])
        
        # Include output paths information
        paths_text = "\n".join([f"- {key}: {path}" for key, path in output_paths.items()])
        
        # Construct a more detailed task with quality verification steps
        task = f"""
        Process the video file at {video_path} with the following operations:
        
        {operations_text}
        
        Use the following output paths for generated files:
        
        {paths_text}
        
        IMPORTANT WORKFLOW GUIDELINES:
        
        1. Start by having the transcription_agent extract audio and generate a transcript
           - After audio extraction, verify audio quality before proceeding
           - For transcription, use a multi-pass approach:
              a) First pass: Full transcription
              b) Second pass: Review problematic segments and improve accuracy
              c) Verify the transcript quality before proceeding
           - If transcription fails, create a placeholder transcript to allow other operations to continue
        
        2. Once a transcript is available (even if placeholder), use the subtitle_agent to:
           - Create properly formatted subtitles with accurate timing
           - Perform timing adjustments if needed to match speech
           - Ensure subtitles are properly added to the video
           - Verify subtitle quality and timing accuracy
           - If subtitle embedding fails, try multiple alternative methods
           - IMPORTANT: When embedding subtitles, try ALL of the following methods until one works:
              a) Use FFmpeg with the simpler command: ffmpeg -i video.mp4 -vf subtitles=subs.srt output.mp4
              b) Use FFmpeg with absolute paths and explicit map commands
              c) Try with subtitle filter using copied files in a temp directory
              d) If all else fails, create a basic subtitled video
        
        3. If summarization is requested, use the summary_agent to:
           - Generate a comprehensive summary using the transcript
           - Utilize both extractive and abstractive summarization approaches
           - Focus on capturing key information from the content
           - Verify summary quality before finalizing
           - If the transcript is incomplete/placeholder, still create a basic summary indicating the limitation
           - Make sure to double-check the summary file creation and verify its contents
           - Always output a JSON file containing both the plain text summary and structured data
        
        4. For any operation that fails, always attempt alternative methods:
           - Use different parameters for tools to improve results
           - Retry failed operations with adjusted settings
           - Document any issues in the final output
           - Always produce some result file, even if it's just a placeholder
        
        Language settings: {"Use language detection" if options.get("language") == "auto" else f"Use language: {options.get('language')}"}
        
        Additional options:
        {json.dumps(options, indent=2)}
        
        Return a detailed JSON object with paths to all generated files and quality metrics.
        Make sure to handle all errors gracefully and always provide usable outputs for each requested operation.
        """
        
        return task
    
    def _parse_agent_result(self, result: str) -> Dict[str, Any]:
        """Parse the agent result to extract relevant information with enhanced robustness.
        
        Args:
            result: Result string from the agent
            
        Returns:
            Parsed result as a dictionary
        """
        # If result is already a dictionary
        if isinstance(result, dict):
            return result
            
        # Try to parse as JSON
        try:
            # Look for JSON in the string using a more robust pattern
            json_pattern = r'({[\s\S]*?})'
            json_matches = re.findall(json_pattern, result.replace('\\n', ' '))
            
            for json_str in json_matches:
                try:
                    # Try progressively larger chunks to find valid JSON
                    parsed_result = json.loads(json_str)
                    if isinstance(parsed_result, dict):
                        return parsed_result
                except json.JSONDecodeError:
                    continue
            
            # If no full JSON found, try finding JSON-like structures
            dict_pattern = r'"([^"]+)":\s*(?:"([^"]*)"|\d+|\{[^}]+\}|\[[^\]]+\])'
            matches = re.findall(dict_pattern, result)
            if matches:
                parsed_dict = {}
                for key, value in matches:
                    # Try to convert to appropriate types
                    if value.isdigit():
                        parsed_dict[key] = int(value)
                    elif value.lower() in ('true', 'false'):
                        parsed_dict[key] = value.lower() == 'true'
                    else:
                        parsed_dict[key] = value
                
                if parsed_dict:
                    return parsed_dict
        except Exception as e:
            logger.error(f"Error parsing result: {str(e)}")
        
        # Look for file paths in the text if JSON parsing fails
        try:
            result_dict = {}
            
            # Common file patterns to look for
            video_pattern = r'(?:video|video with subtitles|subtitled video)(?:\s+path)?[:\s]+([^\s,]+\.mp4)'
            subtitle_pattern = r'(?:subtitle|subtitles)(?:\s+path)?[:\s]+([^\s,]+\.srt)'
            transcript_pattern = r'(?:transcript)(?:\s+path)?[:\s]+([^\s,]+\.(?:txt|json))'
            summary_pattern = r'(?:summary)(?:\s+path)?[:\s]+([^\s,]+\.(?:txt|json))'
            
            # Extract file paths
            video_matches = re.findall(video_pattern, result, re.IGNORECASE)
            subtitle_matches = re.findall(subtitle_pattern, result, re.IGNORECASE)
            transcript_matches = re.findall(transcript_pattern, result, re.IGNORECASE)
            summary_matches = re.findall(summary_pattern, result, re.IGNORECASE)
            
            # For each matched path, validate it exists or find a nearby alternative
            if video_matches:
                path = video_matches[0]
                found_path = self.path_manager.find_nearest_match(path)
                result_dict["subtitled_video_path"] = found_path or path
            
            if subtitle_matches:
                path = subtitle_matches[0]
                found_path = self.path_manager.find_nearest_match(path)
                result_dict["subtitle_path"] = found_path or path
            
            if transcript_matches:
                path = transcript_matches[0]
                found_path = self.path_manager.find_nearest_match(path)
                result_dict["transcript_path"] = found_path or path
            
            if summary_matches:
                path = summary_matches[0]
                found_path = self.path_manager.find_nearest_match(path)
                result_dict["summary_path"] = found_path or path
            
            # Extract language information if available
            language_match = re.search(r'(?:language|detected language)[:\s]+([a-z]{2})', result, re.IGNORECASE)
            if language_match:
                result_dict["language"] = language_match.group(1)
            
            # Try to extract summary content if mentioned
            summary_content_match = re.search(r'Summary content:\s*(.*?)(?:\n\n|\Z)', result, re.DOTALL | re.IGNORECASE)
            if summary_content_match:
                result_dict["summary_content"] = summary_content_match.group(1).strip()
            
            if result_dict:
                result_dict["raw_output"] = result
                return result_dict
        except Exception as e:
            logger.error(f"Error extracting paths from result: {str(e)}")
        
        # Return a default dict with the raw result
        return {"result": result, "raw_output": result}