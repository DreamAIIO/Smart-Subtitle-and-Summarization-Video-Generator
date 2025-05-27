"""
Utility module for handling file paths in the subtitle generator application.
Centralizes path management to ensure consistency across the application.
"""
import os
import tempfile
import logging
from typing import Dict, List, Optional, Tuple
import shutil

logger = logging.getLogger("subtitle_generator")

class PathManager:
    """Manages file paths for the subtitle generator application."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the path manager with a base directory.
        
        Args:
            base_dir: Base directory for the application. If None, will use a default.
        """
        if base_dir is None:
            # Use current working directory as default
            self.base_dir = os.path.abspath(os.getcwd())
        else:
            self.base_dir = os.path.abspath(base_dir)
            
        # Create output directory structure
        self.output_dir = os.path.join(self.base_dir, "output")
        self._ensure_directory_structure()
        
        # Dictionary to track temporary files that need cleanup
        self.temp_files = {}
    
    def _ensure_directory_structure(self) -> None:
        """Ensure the output directory structure exists."""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["videos", "audios", "audio", "transcripts", "subtitles", "summaries", "temp"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
            
        logger.info(f"Directory structure created at {self.output_dir}")
    
    def get_output_path(self, category: str, input_file_path: str, extension: str = None) -> str:
        """Get an output path for a given category and input file.
        
        Args:
            category: Category of the file (videos, audios, transcripts, etc.)
            input_file_path: Original input file path
            extension: Optional extension to use instead of the original
            
        Returns:
            Absolute path for the output file
        """
        # Extract the base filename from the input path
        basename = os.path.basename(input_file_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Use the provided extension or keep the original
        if extension:
            # Ensure extension starts with a dot
            if not extension.startswith('.'):
                extension = '.' + extension
            output_filename = f"{name_without_ext}{extension}"
        else:
            # Keep the original extension
            ext = os.path.splitext(basename)[1]
            output_filename = f"{name_without_ext}{ext}"
        
        # Build the output path
        output_path = os.path.join(self.output_dir, category, output_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        return output_path
    
    def handle_streamlit_path(self, streamlit_path: str) -> str:
        """Handle paths from Streamlit temporary uploads.
        
        Args:
            streamlit_path: Original path from Streamlit
            
        Returns:
            Working path that can be accessed by tools
        """
        # Normalize the path to remove any issues
        norm_path = os.path.normpath(streamlit_path)
        
        # Check if accessible
        is_valid, message = self.validate_file_path(norm_path)
        if is_valid:
            return norm_path
            
        # If not accessible, copy to our temp directory
        try:
            temp_dir = os.path.join(self.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            filename = os.path.basename(norm_path)
            new_path = os.path.join(temp_dir, filename)
            
            # Try to copy the file
            shutil.copy2(norm_path, new_path)
            return new_path
        except Exception as e:
            logger.error(f"Failed to copy Streamlit file: {str(e)}")
            return norm_path  # Return original path as fallback
    
    def get_temp_file_path(self, prefix: str = "", suffix: str = "") -> str:
        """Get a path for a temporary file that will be automatically tracked for cleanup.
        
        Args:
            prefix: Prefix for the temporary file
            suffix: Suffix (including extension) for the temporary file
            
        Returns:
            Path to the temporary file
        """
        # Create the temp directory if it doesn't exist
        temp_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix, dir=temp_dir)
        temp_path = temp_file.name
        temp_file.close()
        
        # Track the temporary file for later cleanup
        self.temp_files[temp_path] = True
        
        return temp_path
    
    def build_output_paths_for_video(self, video_path: str) -> Dict[str, str]:
        """Build a dictionary of all output paths for a given video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of output paths for different file types
        """
        video_basename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(video_basename)[0]
        
        paths = {
            "video": video_path,
            "audio": self.get_output_path("audio", video_path, ".wav"),
            "transcript_json": self.get_output_path("transcripts", video_path, ".json"),
            "transcript_txt": self.get_output_path("transcripts", video_path, ".txt"),
            "subtitle": self.get_output_path("subtitles", video_path, ".srt"),
            "subtitled_video": self.get_output_path("videos", video_path, "_subtitled.mp4"),
            "summary": self.get_output_path("summaries", video_path, ".json"),
            "summary_txt": self.get_output_path("summaries", video_path, ".txt"),
            "translated_subtitle": self.get_output_path("subtitles", video_path, "_translated.srt")
        }
        
        return paths
    
    def clean_temp_files(self) -> None:
        """Clean up temporary files created by the path manager."""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")
        
        # Clear the dictionary after cleanup
        self.temp_files = {}
    
    def validate_file_path(self, file_path: str) -> Tuple[bool, str]:
        """Validate if a file path exists and is accessible.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not file_path:
            return False, "File path is empty"
            
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"
            
        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"
            
        # Check if the file is accessible
        try:
            with open(file_path, 'rb') as f:
                # Just try to read 1 byte to verify access
                f.read(1)
            return True, "File is valid and accessible"
        except Exception as e:
            return False, f"File exists but is not accessible: {str(e)}"
    
    def find_nearest_match(self, expected_path: str, directory: Optional[str] = None) -> Optional[str]:
        """Find the nearest matching file if the expected path doesn't exist.
        
        Args:
            expected_path: The expected file path
            directory: Optional directory to search in
            
        Returns:
            Path to the nearest matching file or None if not found
        """
        if os.path.exists(expected_path):
            return expected_path
            
        # Get the filename and extension
        filename = os.path.basename(expected_path)
        name_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        
        # Define search directories
        search_dirs = []
        if directory:
            search_dirs.append(directory)
        else:
            # If no specific directory, search in the expected directory and output dirs
            expected_dir = os.path.dirname(expected_path)
            search_dirs.append(expected_dir)
            for subdir in ["videos", "audios", "audio", "transcripts", "subtitles", "summaries", "temp"]:
                search_dirs.append(os.path.join(self.output_dir, subdir))
        
        # Look for files with similar names in each directory
        for search_dir in search_dirs:
            if not os.path.exists(search_dir) or not os.path.isdir(search_dir):
                continue
                
            # Look for exact match first
            exact_path = os.path.join(search_dir, filename)
            if os.path.exists(exact_path):
                return exact_path
                
            # Look for files with similar names
            for file in os.listdir(search_dir):
                if extension and file.endswith(extension) and name_without_ext in file:
                    return os.path.join(search_dir, file)
        
        return None