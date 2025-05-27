"""
Updated file handling utilities for the subtitle generator application.
"""
import os
import shutil
import tempfile
import uuid
import logging
from typing import Optional, Dict, Any, List
import streamlit as st

from utils.error_handling import handle_errors

logger = logging.getLogger("subtitle_generator")


class TempFileManager:
    """Enhanced temporary file manager for the subtitle generator application."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the temporary file manager.
        
        Args:
            base_dir: Optional base directory for temp files
        """
        if base_dir:
            self.temp_dir = os.path.join(base_dir, "temp")
        else:
            # Use system temp directory as a fallback
            self.temp_dir = os.path.join(tempfile.gettempdir(), "subtitle_generator")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Dictionary to track temp files
        self.temp_files = {}
        
        logger.debug(f"Initialized TempFileManager with temp dir: {self.temp_dir}")
    
    @handle_errors(default_return=None)
    def save_uploaded_file(self, uploaded_file: Any) -> Optional[str]:
        """Save an uploaded file from Streamlit to a temporary location."""
        if not uploaded_file:
            return None
            
        # Create a unique filename while preserving original extension
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_id = uuid.uuid4().hex
        temp_filename = f"{unique_id}_{uploaded_file.name}"
        
        # First try the system temp directory which has proper permissions
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.getbuffer())
            
            logger.info(f"Saved uploaded file to system temp: {temp_path}")
            return temp_path
        except Exception as e:
            logger.warning(f"Could not save to system temp: {str(e)}")
        
        # Fallback to our temp directory
        try:
            temp_path = os.path.join(self.temp_dir, temp_filename)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"Saved uploaded file to app temp: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            return None
    
    @handle_errors(default_return=None)
    def create_temp_file(self, content: str, prefix: str = "", suffix: str = "") -> Optional[str]:
        """Create a temporary file with the given content.
        
        Args:
            content: Content to write to the file
            prefix: Prefix for the temporary file name
            suffix: Suffix for the temporary file name
            
        Returns:
            Path to the created temporary file
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, prefix=prefix, suffix=suffix, dir=self.temp_dir
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content.encode('utf-8'))
            
            # Keep track of this file
            self.temp_files[temp_path] = {
                "original_name": os.path.basename(temp_path),
                "timestamp": os.path.getmtime(temp_path)
            }
            
            logger.debug(f"Created temp file at {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temp file: {str(e)}")
            return None
    
    @handle_errors()
    def cleanup_old_files(self, max_age_seconds: int = 3600) -> None:
        """Clean up old temporary files.
        
        Args:
            max_age_seconds: Maximum age of files to keep (in seconds)
        """
        import time
        current_time = time.time()
        
        # Identify files to delete
        files_to_delete = []
        for file_path, file_info in self.temp_files.items():
            if current_time - file_info.get("timestamp", 0) > max_age_seconds:
                files_to_delete.append(file_path)
        
        # Delete the files
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed old temp file: {file_path}")
                
                # Remove from tracking
                del self.temp_files[file_path]
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {str(e)}")
    
    @handle_errors()
    def build_output_directories(self, video_path: str) -> Dict[str, str]:
        """Build necessary output directories based on the video path.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of created output directories
        """
        # Determine base directory - handle both absolute and temp paths
        if os.path.isabs(video_path):
            base_dir = os.path.dirname(os.path.dirname(video_path)) 
        else:
            # Use the current directory as fallback
            base_dir = os.getcwd()
        
        # Create output base directory
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = {
            "videos": os.path.join(output_dir, "videos"),
            "audio": os.path.join(output_dir, "audio"),
            "transcripts": os.path.join(output_dir, "transcripts"),
            "subtitles": os.path.join(output_dir, "subtitles"),
            "summaries": os.path.join(output_dir, "summaries"),
            "temp": os.path.join(output_dir, "temp")
        }
        
        # Create each subdirectory
        for name, path in subdirs.items():
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Created directory: {path}")
        
        # Add the base output directory
        subdirs["output"] = output_dir
        
        return subdirs
    
    @handle_errors()
    def find_or_suggest_file(self, expected_path: str, file_type: str) -> Optional[str]:
        """Find a file at the expected path or suggest an alternative.
        
        Args:
            expected_path: Expected path to the file
            file_type: Type of file (e.g., 'transcript', 'subtitle')
            
        Returns:
            Path to the found file or None if not found
        """
        if os.path.exists(expected_path):
            return expected_path
            
        # Extract filename components
        dir_path = os.path.dirname(expected_path)
        basename = os.path.basename(expected_path)
        name_without_ext, ext = os.path.splitext(basename)
        
        # Try alternative extensions based on file type
        alt_extensions = []
        if file_type == 'transcript':
            alt_extensions = ['.txt', '.json', '.srt']
        elif file_type == 'subtitle':
            alt_extensions = ['.srt', '.vtt', '.sub']
        elif file_type == 'summary':
            alt_extensions = ['.txt', '.json', '.md']
        elif file_type == 'video':
            alt_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Try the same name with different extensions
        for alt_ext in alt_extensions:
            alt_path = os.path.join(dir_path, name_without_ext + alt_ext)
            if os.path.exists(alt_path):
                logger.info(f"Found alternative file: {alt_path}")
                return alt_path
        
        # Try similar names in the directory
        try:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                for filename in os.listdir(dir_path):
                    if name_without_ext.lower() in filename.lower():
                        alt_path = os.path.join(dir_path, filename)
                        logger.info(f"Found similar file: {alt_path}")
                        return alt_path
        except Exception as e:
            logger.warning(f"Error searching directory {dir_path}: {str(e)}")
        
        # Try to find a file with similar content in parent directories
        try:
            # Look in parent directories
            parent_dir = os.path.dirname(dir_path)
            parent_parent_dir = os.path.dirname(parent_dir)
            
            for search_dir in [parent_dir, parent_parent_dir]:
                if not os.path.exists(search_dir) or not os.path.isdir(search_dir):
                    continue
                    
                # Look in all subdirectories
                for root, dirs, files in os.walk(search_dir):
                    for filename in files:
                        if name_without_ext.lower() in filename.lower() or ext.lower() == os.path.splitext(filename)[1].lower():
                            alt_path = os.path.join(root, filename)
                            logger.info(f"Found similar file in parent directory: {alt_path}")
                            return alt_path
        except Exception as e:
            logger.warning(f"Error searching parent directories: {str(e)}")
            
        return None
    
    def __del__(self):
        """Cleanup temporary files when the object is destroyed."""
        self.cleanup_old_files()


def validate_video_file(file_path: str) -> tuple[bool, str]:
    """
    Validates if a file is a valid video file.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    import os
    
    # Check if file exists
    if not file_path:
        return False, "File path is empty"
        
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
        
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
        
    # Check if it has a video extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in valid_extensions:
        return False, f"Not a supported video format: {file_ext}"
        
    # Check if the file size is valid
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Video file is empty"
        if file_size > 500 * 1024 * 1024:  # 500 MB
            return False, f"Video file is too large: {file_size / (1024 * 1024):.1f} MB"
    except Exception as e:
        return False, f"Error checking file size: {str(e)}"
        
    # Try to verify file integrity using ffprobe if available
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', file_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Not a valid video file according to ffprobe: {result.stderr}"
    except Exception:
        # If ffprobe check fails, we'll still accept the file but log a warning
        logger.warning(f"Could not verify video integrity with ffprobe for {file_path}")
        
    return True, "Valid video file"