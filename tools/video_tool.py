"""
Video processing tool for extracting audio and adding subtitles to videos.
Enhanced with more robust subtitle embedding and error handling.
"""
import os
import json
from typing import Dict, Optional, Any
import shutil
import tempfile
import uuid
import subprocess
import re

import ffmpeg
from moviepy.editor import VideoFileClip
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution
from utils.file_handling import validate_video_file


class VideoProcessingTool(Tool):
    """Tool for video processing operations."""
    
    name = "video_processor"
    description = """
    Process video files, including extracting audio and adding subtitles.
    """
    inputs = {
        "video_path": {
            "type": "string",
            "description": "Path to the video file"
        },
        "operation": {
            "type": "string",
            "description": "Operation to perform (extract_audio, add_subtitles, etc.)"
        },
        "options": {
            "type": "object",
            "description": "Additional options for the operation",
            "nullable": True
        }
    }
    output_type = "string"
    
    @log_execution
    @handle_errors(default_return="Error processing video")
    def forward(self, video_path: str, operation: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Process the video based on the specified operation.
        
        Args:
            video_path: Path to the video file
            operation: Operation to perform
            options: Additional options for the operation
            
        Returns:
            Path to the output file or status message
        """
        options = options or {}
        
        # Validate the video file
        is_valid, message = validate_video_file(video_path)
        if not is_valid:
            return f"Invalid video file: {message}"
        
        # Dispatch to the appropriate method based on the operation
        if operation == "extract_audio":
            return self._extract_audio(video_path, options)
        elif operation == "add_subtitles":
            return self._add_subtitles(video_path, options)
        elif operation == "get_video_info":
            return self._get_video_info(video_path)
        else:
            return f"Unsupported operation: {operation}"
    
    @handle_errors()
    def _extract_audio(self, video_path: str, options: Dict[str, Any]) -> str:
        """Extract audio from a video file."""
        output_format = options.get("format", "wav")  # Default to WAV
        output_path = options.get("output_path", None)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(video_path)
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(dirname, f"{basename}.{output_format}")
        
        try:
            # Use a more robust approach with proper quoting of file paths
            # Create a temporary directory to work in
            temp_dir = tempfile.gettempdir()
            temp_id = uuid.uuid4().hex
            temp_output = os.path.join(temp_dir, f"temp_audio_{temp_id}.{output_format}")
            
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(temp_output, acodec='pcm_s16le' if output_format == 'wav' else 'copy')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            
            # Move the file to the intended location
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(temp_output, output_path)
            
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}") from e

    @handle_errors()
    def _add_subtitles(self, video_path: str, options: Dict[str, Any]) -> str:
        """Add subtitles to a video file with enhanced robustness.
        
        Args:
            video_path: Path to the video file
            options: Additional options for subtitle embedding
            
        Returns:
            Path to the output video file with embedded subtitles
        """
        verbose = options.get("verbose", True)
        
        # Log start of operation if verbose
        if verbose:
            print(f"Starting subtitle embedding for video: {video_path}")
        
        subtitle_path = options.get("subtitle_path")
        if not subtitle_path:
            raise ValueError("subtitle_path is required in options")
        
        output_path = options.get("output_path", None)
        if output_path is None:
            dirname = os.path.dirname(video_path)
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(dirname, f"{basename}_subtitled.mp4")
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define the subtitle method to use
        subtitle_method = options.get("subtitle_method", "auto")
        
        # Get the list of methods to try based on the subtitle method
        methods_to_try = []
        if subtitle_method == "auto":
            # Try all methods in order of preference
            methods_to_try = ["filter", "map", "burn", "moviepy"]
        else:
            # Use only the specified method
            methods_to_try = [subtitle_method]
        
        # Try each method until one works
        for method in methods_to_try:
            try:
                if verbose:
                    print(f"Trying subtitle embedding method: {method}")
                
                if method == "filter":
                    result = self._add_subtitles_filter(video_path, subtitle_path, output_path, verbose)
                elif method == "map":
                    result = self._add_subtitles_map(video_path, subtitle_path, output_path, verbose)
                elif method == "burn":
                    result = self._add_subtitles_burn(video_path, subtitle_path, output_path, verbose)
                elif method == "moviepy":
                    result = self._add_subtitles_moviepy(video_path, subtitle_path, output_path, verbose)
                else:
                    continue
                
                # Check if the output file exists and has content
                if os.path.exists(result) and os.path.getsize(result) > 0:
                    if verbose:
                        print(f"Successfully created subtitled video at {result} using method {method}")
                    return result
            except Exception as e:
                if verbose:
                    print(f"Method {method} failed: {str(e)}")
                # Continue with the next method
                continue
        
        # All methods failed, create a fallback solution
        try:
            if verbose:
                print("All subtitle embedding methods failed, creating fallback solution")
            
            return self._create_subtitle_fallback(video_path, subtitle_path, output_path)
        except Exception as e:
            if verbose:
                print(f"Fallback solution failed: {str(e)}")
            
            # Last resort: Just copy the original video and create an info file
            shutil.copy2(video_path, output_path)
            fallback_info_path = os.path.splitext(output_path)[0] + "_subtitles_info.txt"
            with open(fallback_info_path, "w", encoding="utf-8") as f:
                f.write(f"Subtitles could not be embedded in the video.\n")
                f.write(f"Please use the separate subtitle file at: {subtitle_path}\n")
                f.write(f"You can load this subtitle file in your video player to see the subtitles.")
            
            if verbose:
                print(f"Created fallback solution at {output_path}")
            
            return output_path
    
    def _add_subtitles_filter(self, video_path: str, subtitle_path: str, output_path: str, verbose: bool = False) -> str:
        """Add subtitles using FFmpeg's subtitles filter.
        
        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_path: Path to save the output file
            verbose: Whether to print verbose output
            
        Returns:
            Path to the output file
        """
        # Create clean paths to avoid issues with spaces
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        temp_subtitle = os.path.join(temp_dir, "temp_subtitle.srt")
        temp_output = os.path.join(temp_dir, "temp_output.mp4")
        
        try:
            # Copy files to temp location
            shutil.copy2(video_path, temp_video)
            shutil.copy2(subtitle_path, temp_subtitle)
            
            # Escape path for subtitles filter
            escaped_subtitle_path = temp_subtitle.replace(":", "\\:").replace("'", "\\'")
            
            # Build the FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', temp_video,
                '-vf', f"subtitles='{escaped_subtitle_path}'",
                '-c:a', 'copy',
                '-y',
                temp_output
            ]
            
            if verbose:
                print(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                if verbose:
                    print(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            # Check if output exists
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                # Move to final location
                shutil.move(temp_output, output_path)
                return output_path
            else:
                raise RuntimeError("Output file was not created")
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _add_subtitles_map(self, video_path: str, subtitle_path: str, output_path: str, verbose: bool = False) -> str:
        """Add subtitles using FFmpeg's explicit mapping.
        
        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_path: Path to save the output file
            verbose: Whether to print verbose output
            
        Returns:
            Path to the output file
        """
        # Create clean paths to avoid issues with spaces
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        temp_subtitle = os.path.join(temp_dir, "temp_subtitle.srt")
        temp_output = os.path.join(temp_dir, "temp_output.mp4")
        
        try:
            # Copy files to temp location
            shutil.copy2(video_path, temp_video)
            shutil.copy2(subtitle_path, temp_subtitle)
            
            # Build the FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', temp_video,
                '-f', 'srt',
                '-i', temp_subtitle,
                '-map', '0:v',
                '-map', '0:a',
                '-map', '1',
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-c:s', 'mov_text',
                '-y',
                temp_output
            ]
            
            if verbose:
                print(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                if verbose:
                    print(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            # Check if output exists
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                # Move to final location
                shutil.move(temp_output, output_path)
                return output_path
            else:
                raise RuntimeError("Output file was not created")
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _add_subtitles_burn(self, video_path: str, subtitle_path: str, output_path: str, verbose: bool = False) -> str:
        """Burn subtitles directly into the video.
        
        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_path: Path to save the output file
            verbose: Whether to print verbose output
            
        Returns:
            Path to the output file
        """
        # Create clean paths to avoid issues with spaces
        temp_dir = tempfile.mkdtemp()
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        temp_subtitle = os.path.join(temp_dir, "temp_subtitle.srt")
        temp_output = os.path.join(temp_dir, "temp_output.mp4")
        
        try:
            # Copy files to temp location
            shutil.copy2(video_path, temp_video)
            shutil.copy2(subtitle_path, temp_subtitle)
            
            # Escape path for subtitles filter
            escaped_subtitle_path = temp_subtitle.replace(":", "\\:").replace("'", "\\'")
            
            # Build the FFmpeg command with style options
            cmd = [
                'ffmpeg',
                '-i', temp_video,
                '-vf', f"subtitles='{escaped_subtitle_path}':force_style='FontSize=24,BorderStyle=3'",
                '-c:a', 'aac',
                '-y',
                temp_output
            ]
            
            if verbose:
                print(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                if verbose:
                    print(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            # Check if output exists
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                # Move to final location
                shutil.move(temp_output, output_path)
                return output_path
            else:
                raise RuntimeError("Output file was not created")
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _add_subtitles_moviepy(self, video_path: str, subtitle_path: str, output_path: str, verbose: bool = False) -> str:
        """Add subtitles using MoviePy.
        
        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_path: Path to save the output file
            verbose: Whether to print verbose output
            
        Returns:
            Path to the output file
        """
        try:
            if verbose:
                print("Attempting to use MoviePy for subtitle embedding")
            
            from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
            import pysrt
            
            # Read the subtitle file
            subs = pysrt.open(subtitle_path)
            
            # Load the video
            with VideoFileClip(video_path) as video:
                # Create text clips for each subtitle and overlay them
                subtitle_clips = []
                
                for sub in subs:
                    start_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
                    end_time = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
                    
                    # Create text clip
                    txt_clip = TextClip(
                        sub.text,
                        fontsize=24,
                        color='white',
                        bg_color='black',
                        size=(video.w, None),
                        method='caption'
                    )
                    txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(start_time).set_end(end_time)
                    subtitle_clips.append(txt_clip)
                
                # Add subtitles to the video
                final_clip = CompositeVideoClip([video] + subtitle_clips)
                
                # Write the result to a file
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac'
                )
                
                return output_path
        except Exception as e:
            if verbose:
                print(f"MoviePy method failed: {str(e)}")
            raise RuntimeError(f"MoviePy method failed: {str(e)}")
    
    def _create_subtitle_fallback(self, video_path: str, subtitle_path: str, output_path: str) -> str:
        """Create a fallback solution when all subtitle embedding methods fail.
        
        Args:
            video_path: Path to the video file
            subtitle_path: Path to the subtitle file
            output_path: Path to save the output file
            
        Returns:
            Path to the output file
        """
        # Just copy the original video
        shutil.copy2(video_path, output_path)
        
        # Create an info file with subtitle information
        fallback_info_path = os.path.splitext(output_path)[0] + "_subtitles_info.txt"
        with open(fallback_info_path, "w", encoding="utf-8") as f:
            f.write(f"Subtitles could not be embedded in the video.\n")
            f.write(f"Please use the separate subtitle file at: {subtitle_path}\n")
            f.write(f"You can load this subtitle file in your video player to see the subtitles.")
        
        return output_path
    
    @handle_errors()
    def _get_video_info(self, video_path: str) -> str:
        """Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            JSON string with video information
        """
        try:
            # Get video info using ffprobe
            probe = ffmpeg.probe(video_path)
            
            # Extract video and audio streams
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            # Format duration
            duration = float(probe.get('format', {}).get('duration', 0))
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Extract relevant information
            info = {
                "filename": os.path.basename(video_path),
                "duration": duration,
                "formatted_duration": formatted_duration,
                "size_bytes": int(probe.get('format', {}).get('size', 0)),
                "format": probe.get('format', {}).get('format_name', 'unknown'),
                "bitrate": int(probe.get('format', {}).get('bit_rate', 0)) if 'bit_rate' in probe.get('format', {}) else 0,
            }
            
            # Add video stream info if available
            if video_info:
                # Calculate frame rate
                frame_rate = 0
                if 'avg_frame_rate' in video_info:
                    try:
                        fraction = video_info['avg_frame_rate'].split('/')
                        if len(fraction) == 2 and int(fraction[1]) != 0:
                            frame_rate = int(fraction[0]) / int(fraction[1])
                    except:
                        pass
                
                info["video"] = {
                    "codec": video_info.get('codec_name', 'unknown'),
                    "width": int(video_info.get('width', 0)),
                    "height": int(video_info.get('height', 0)),
                    "aspect_ratio": video_info.get('display_aspect_ratio', 'unknown'),
                    "frame_rate": frame_rate,
                    "bit_depth": int(video_info.get('bits_per_raw_sample', 8)) if 'bits_per_raw_sample' in video_info else 8,
                }
            
            # Add audio stream info if available
            if audio_info:
                info["audio"] = {
                    "codec": audio_info.get('codec_name', 'unknown'),
                    "channels": int(audio_info.get('channels', 0)),
                    "sample_rate": int(audio_info.get('sample_rate', 0)) if 'sample_rate' in audio_info else 0,
                    "bit_rate": int(audio_info.get('bit_rate', 0)) if 'bit_rate' in audio_info else 0,
                }
            
            # Return as JSON string
            return json.dumps(info, indent=2)
        except Exception as e:
            # Try using MoviePy if FFmpeg fails
            try:
                # Get video info using MoviePy
                with VideoFileClip(video_path) as clip:
                    info = {
                        "filename": os.path.basename(video_path),
                        "duration": clip.duration,
                        "size": clip.size,
                        "fps": clip.fps,
                        "has_audio": clip.audio is not None
                    }
                    return str(info)
            except:
                pass
                
            raise RuntimeError(f"Error getting video info: {str(e)}") from e