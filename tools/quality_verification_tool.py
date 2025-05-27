"""
Quality verification tool for assessing and improving output quality.
"""
import os
import json
import re
from typing import Dict, Optional, Any, List, Tuple

import pysrt
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


class QualityVerificationTool(Tool):
    """Tool for verifying quality of generated outputs."""
    
    name = "quality_verifier"
    description = """
    Verifies the quality of transcriptions, subtitles, and summaries.
    Provides feedback and suggestions for improvement.
    """
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to verify"
        },
        "file_type": {
            "type": "string",
            "description": "Type of file (transcript, subtitle, summary, audio)"
        },
        "reference_path": {
            "type": "string",
            "description": "Optional path to a reference file for comparison",
            "nullable": True
        },
        "options": {
            "type": "object",
            "description": "Additional verification options",
            "nullable": True
        }
    }
    output_type = "string"
    
    @log_execution
    @handle_errors(default_return="Error verifying quality")
    def forward(self, file_path: str, file_type: str, reference_path: Optional[str] = None, 
                options: Optional[Dict[str, Any]] = None) -> str:
        """Verify the quality of generated outputs.
        
        Args:
            file_path: Path to the file to verify
            file_type: Type of file (transcript, subtitle, summary, audio)
            reference_path: Optional path to a reference file for comparison
            options: Additional verification options
            
        Returns:
            Quality assessment with suggestions for improvement
        """
        options = options or {}
        
        if not os.path.exists(file_path):
            return f"File does not exist: {file_path}"
        
        if reference_path and not os.path.exists(reference_path):
            return f"Reference file does not exist: {reference_path}"
        
        # Dispatch to the appropriate verification method based on the file type
        if file_type == "transcript":
            return self._verify_transcript(file_path, reference_path, options)
        elif file_type == "subtitle":
            return self._verify_subtitle(file_path, reference_path, options)
        elif file_type == "summary":
            return self._verify_summary(file_path, reference_path, options)
        elif file_type == "audio":
            return self._verify_audio(file_path, options)
        else:
            return f"Unsupported file type: {file_type}"
    
    @handle_errors()
    def _verify_transcript(self, transcript_path: str, reference_path: Optional[str], 
                          options: Dict[str, Any]) -> str:
        """Verify transcript quality and suggest improvements.
        
        Args:
            transcript_path: Path to the transcript file
            reference_path: Optional path to a reference transcript
            options: Additional verification options
            
        Returns:
            Quality assessment with suggestions
        """
        file_ext = os.path.splitext(transcript_path)[1].lower()
        
        # Load the transcript
        transcript_text = ""
        segments = []
        
        try:
            if file_ext == '.json':
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    
                    # Extract text and segments if available
                    if "text" in transcript_data:
                        transcript_text = transcript_data["text"]
                    
                    if "segments" in transcript_data:
                        segments = transcript_data["segments"]
                        if not transcript_text:
                            transcript_text = " ".join([segment.get("text", "") for segment in segments])
            elif file_ext == '.txt':
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
        except Exception as e:
            return f"Error loading transcript: {str(e)}"
        
        # Perform quality checks
        quality_issues = []
        quality_score = 0
        improvements = []
        
        # Check for empty or very short transcript
        if not transcript_text:
            quality_issues.append("Transcript is empty")
            quality_score = 0
            improvements.append("Regenerate transcript with different parameters")
        else:
            # Basic quality metrics
            word_count = len(transcript_text.split())
            
            # Check for very short transcript
            if word_count < 10:
                quality_issues.append(f"Transcript is very short ({word_count} words)")
                quality_score = max(10, quality_score)
                improvements.append("Retry transcription with different model or parameters")
            
            # Check for repetitive text patterns that might indicate issues
            words = transcript_text.lower().split()
            unique_words = len(set(words))
            
            if word_count > 0:
                repetition_ratio = unique_words / word_count
                if repetition_ratio < 0.3:  # High repetition
                    quality_issues.append(f"High text repetition detected (ratio: {repetition_ratio:.2f})")
                    quality_score = max(30, quality_score)
                    improvements.append("Check audio quality and retry with noise reduction")
            
            # Check for unnatural pauses in segments
            if segments:
                for i in range(1, len(segments)):
                    prev_end = segments[i-1].get("end", 0)
                    curr_start = segments[i].get("start", 0)
                    
                    # Gap of more than 2 seconds between segments
                    if curr_start - prev_end > 2.0:
                        quality_issues.append(f"Potential missed speech between {prev_end:.2f}s and {curr_start:.2f}s")
                        improvements.append(f"Review audio between {prev_end:.2f}s and {curr_start:.2f}s")
            
            # Overall quality assessment based on issues
            if not quality_issues:
                quality_score = 90
                improvements.append("Transcript appears to be of good quality")
            else:
                # Deduct points based on number of issues
                quality_score = max(0, 90 - len(quality_issues) * 10)
        
        # Format the result
        result = {
            "quality_score": quality_score,
            "issues": quality_issues,
            "word_count": word_count if 'word_count' in locals() else 0,
            "improvements": improvements
        }
        
        return json.dumps(result, indent=2)
    
    @handle_errors()
    def _verify_subtitle(self, subtitle_path: str, reference_path: Optional[str], 
                        options: Dict[str, Any]) -> str:
        """Verify subtitle quality and timing accuracy.
        
        Args:
            subtitle_path: Path to the subtitle file
            reference_path: Optional path to a reference subtitle file
            options: Additional verification options
            
        Returns:
            Quality assessment with suggestions
        """
        file_ext = os.path.splitext(subtitle_path)[1].lower()
        
        if file_ext not in ['.srt', '.vtt']:
            return f"Unsupported subtitle format: {file_ext}"
        
        try:
            # Load subtitles
            if file_ext == '.srt':
                subtitles = pysrt.open(subtitle_path)
            else:
                return "VTT verification not yet implemented"
                
            # Perform quality checks
            quality_issues = []
            quality_score = 0
            improvements = []
            timing_issues = []
            formatting_issues = []
            
            # Check for empty subtitle file
            if len(subtitles) == 0:
                quality_issues.append("Subtitle file is empty")
                quality_score = 0
                improvements.append("Regenerate subtitles from transcript")
                return json.dumps({
                    "quality_score": quality_score,
                    "issues": quality_issues,
                    "improvements": improvements
                }, indent=2)
            
            # Analyze subtitle entries
            short_duration_count = 0
            long_duration_count = 0
            line_length_issues = 0
            
            previous_end = 0
            
            for subtitle in subtitles:
                # Check duration (less than 0.7 seconds is too short)
                duration = subtitle.end.ordinal - subtitle.start.ordinal
                if duration < 700:  # Less than 0.7 seconds
                    short_duration_count += 1
                    timing_issues.append(f"Subtitle at {subtitle.start} is too short ({duration/1000:.2f}s)")
                
                # Check for long durations (more than 7 seconds)
                if duration > 7000:
                    long_duration_count += 1
                    timing_issues.append(f"Subtitle at {subtitle.start} is too long ({duration/1000:.2f}s)")
                
                # Check for line length (more than 42 chars per line is hard to read)
                for line in subtitle.text.split("\n"):
                    if len(line) > 42:
                        line_length_issues += 1
                        formatting_issues.append(f"Line too long at {subtitle.start}: {len(line)} chars")
                
                # Check for gaps between subtitles
                if subtitle.start.ordinal > previous_end + 2000:  # Gap of more than 2 seconds
                    timing_issues.append(f"Gap detected between {previous_end/1000:.2f}s and {subtitle.start.ordinal/1000:.2f}s")
                
                previous_end = subtitle.end.ordinal
            
            # Add issues to quality assessment
            if short_duration_count > 0:
                quality_issues.append(f"{short_duration_count} subtitles have very short duration")
                improvements.append("Adjust timing to ensure subtitles display for at least 0.7 seconds")
            
            if long_duration_count > 0:
                quality_issues.append(f"{long_duration_count} subtitles have very long duration")
                improvements.append("Split long subtitles into multiple shorter ones")
            
            if line_length_issues > 0:
                quality_issues.append(f"{line_length_issues} lines exceed recommended length")
                improvements.append("Reformat long lines to improve readability")
            
            # Calculate quality score based on issues
            short_ratio = short_duration_count / len(subtitles)
            long_ratio = long_duration_count / len(subtitles)
            format_ratio = line_length_issues / len(subtitles)
            
            # Deduct points based on issue ratios
            quality_score = 100
            quality_score -= int(short_ratio * 30)
            quality_score -= int(long_ratio * 20)
            quality_score -= int(format_ratio * 20)
            
            # Add suggestions if not already present
            if not improvements:
                if quality_score < 70:
                    improvements.append("Consider regenerating subtitles with better formatting")
                else:
                    improvements.append("Subtitles appear to be of good quality")
            
            result = {
                "quality_score": max(0, quality_score),
                "issues": quality_issues,
                "subtitle_count": len(subtitles),
                "timing_issues": timing_issues[:5],  # Limit to first 5
                "formatting_issues": formatting_issues[:5],  # Limit to first 5
                "improvements": improvements
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error verifying subtitles: {str(e)}"
    
    @handle_errors()
    def _verify_summary(self, summary_path: str, reference_path: Optional[str], 
                      options: Dict[str, Any]) -> str:
        """Verify summary quality and content coverage.
        
        Args:
            summary_path: Path to the summary file
            reference_path: Optional path to the source transcript for comparison
            options: Additional verification options
            
        Returns:
            Quality assessment with suggestions
        """
        file_ext = os.path.splitext(summary_path)[1].lower()
        
        # Load the summary
        summary_text = ""
        
        try:
            if file_ext == '.json':
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    
                    if "summary" in summary_data:
                        if isinstance(summary_data["summary"], list):
                            summary_text = " ".join(summary_data["summary"])
                        else:
                            summary_text = summary_data["summary"]
            else:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_text = f.read()
        except Exception as e:
            return f"Error loading summary: {str(e)}"
        
        # Check reference transcript if available
        transcript_text = ""
        if reference_path and os.path.exists(reference_path):
            try:
                ref_ext = os.path.splitext(reference_path)[1].lower()
                
                if ref_ext == '.json':
                    with open(reference_path, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)
                        
                        if "text" in transcript_data:
                            transcript_text = transcript_data["text"]
                        elif "segments" in transcript_data:
                            segments = transcript_data["segments"]
                            transcript_text = " ".join([segment.get("text", "") for segment in segments if isinstance(segment, dict)])
                else:
                    with open(reference_path, 'r', encoding='utf-8') as f:
                        transcript_text = f.read()
            except Exception as e:
                print(f"Warning: Could not load reference transcript: {str(e)}")
        
        # Perform quality checks
        quality_issues = []
        quality_score = 0
        improvements = []
        
        # Check for empty or very short summary
        if not summary_text:
            quality_issues.append("Summary is empty")
            quality_score = 0
            improvements.append("Regenerate summary with different parameters")
        else:
            # Basic quality metrics
            summary_word_count = len(summary_text.split())
            
            # Check for very short summary
            if summary_word_count < 5:
                quality_issues.append(f"Summary is very short ({summary_word_count} words)")
                quality_score = max(10, quality_score)
                improvements.append("Retry summarization with different parameters")
            
            # Check content coverage if transcript is available
            if transcript_text:
                transcript_word_count = len(transcript_text.split())
                
                # Check summary ratio (ideal is 5-15% of original)
                if transcript_word_count > 0:
                    summary_ratio = summary_word_count / transcript_word_count
                    
                    if summary_ratio < 0.01:  # Less than 1% of original
                        quality_issues.append(f"Summary is too short relative to transcript (ratio: {summary_ratio:.2f})")
                        improvements.append("Generate a more comprehensive summary")
                    elif summary_ratio > 0.4:  # More than 40% of original
                        quality_issues.append(f"Summary is too long relative to transcript (ratio: {summary_ratio:.2f})")
                        improvements.append("Generate a more concise summary")
                
                # Check keyword coverage
                # Extract important words from transcript
                transcript_words = re.findall(r'\b[A-Za-z]{4,}\b', transcript_text.lower())
                summary_words = re.findall(r'\b[A-Za-z]{4,}\b', summary_text.lower())
                
                # Count word frequencies in transcript
                word_freq = {}
                for word in transcript_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top keywords
                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                important_words = [word for word, _ in top_keywords]
                
                # Check how many important words are in summary
                keywords_found = sum(1 for word in important_words if word in summary_words)
                keyword_coverage = keywords_found / len(important_words) if important_words else 0
                
                if keyword_coverage < 0.3:
                    quality_issues.append(f"Summary covers only {keyword_coverage*100:.1f}% of important topics")
                    improvements.append("Include more key topics from the transcript")
            
            # Overall quality assessment
            if not quality_issues:
                quality_score = 90
                improvements.append("Summary appears to be of good quality")
            else:
                # Deduct points based on number of issues
                quality_score = max(0, 90 - len(quality_issues) * 15)
        
        # Format the result
        result = {
            "quality_score": quality_score,
            "issues": quality_issues,
            "word_count": summary_word_count if 'summary_word_count' in locals() else 0,
            "improvements": improvements
        }
        
        return json.dumps(result, indent=2)
    
    @handle_errors()
    def _verify_audio(self, audio_path: str, options: Dict[str, Any]) -> str:
        """Verify audio quality for transcription.
        
        Args:
            audio_path: Path to the audio file
            options: Additional verification options
            
        Returns:
            Quality assessment with suggestions
        """
        # Simple file existence and size check
        if not os.path.exists(audio_path):
            return json.dumps({
                "quality_score": 0,
                "issues": ["Audio file does not exist"],
                "improvements": ["Ensure the audio file path is correct"]
            }, indent=2)
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return json.dumps({
                "quality_score": 0,
                "issues": ["Audio file is empty"],
                "improvements": ["Re-extract audio from video source"]
            }, indent=2)
        
        # For a more comprehensive audio quality check, we'd need to analyze the audio
        # Using libraries like librosa, but for simplicity, we'll just check the file size
        quality_score = 70  # Default moderate score
        issues = []
        improvements = []
        
        if file_size < 1024 * 100:  # Less than 100KB
            quality_score = 30
            issues.append("Audio file is very small, potentially low quality")
            improvements.append("Extract audio at a higher bitrate")
        elif file_size > 1024 * 1024 * 100:  # More than 100MB
            quality_score = 90
            improvements.append("Audio file is large and likely of good quality")
        
        # Add metadata if available
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", audio_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                
                # Extract relevant audio information
                audio_info = {}
                
                if "format" in metadata:
                    audio_info["format"] = metadata["format"].get("format_name", "unknown")
                    audio_info["duration"] = float(metadata["format"].get("duration", 0))
                
                for stream in metadata.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        audio_info["codec"] = stream.get("codec_name", "unknown")
                        audio_info["sample_rate"] = stream.get("sample_rate", "unknown")
                        audio_info["channels"] = stream.get("channels", 0)
                        audio_info["bit_rate"] = stream.get("bit_rate", "unknown")
                        break
                
                # Adjust quality score based on audio properties
                if "sample_rate" in audio_info:
                    sample_rate = int(audio_info["sample_rate"]) if audio_info["sample_rate"].isdigit() else 0
                    if sample_rate < 16000:
                        quality_score = max(30, quality_score - 20)
                        issues.append(f"Low sample rate: {sample_rate}Hz")
                        improvements.append("Extract audio with at least 16kHz sample rate for better transcription")
                    elif sample_rate >= 44100:
                        quality_score = min(95, quality_score + 10)
                
                if "bit_rate" in audio_info and audio_info["bit_rate"].isdigit():
                    bit_rate = int(audio_info["bit_rate"]) / 1000  # Convert to kbps
                    if bit_rate < 64:
                        quality_score = max(30, quality_score - 15)
                        issues.append(f"Low bit rate: {bit_rate}kbps")
                        improvements.append("Extract audio with at least 128kbps bit rate")
                    elif bit_rate >= 128:
                        quality_score = min(95, quality_score + 5)
                
                # Add audio information to the result
                return json.dumps({
                    "quality_score": quality_score,
                    "issues": issues,
                    "improvements": improvements,
                    "audio_info": audio_info
                }, indent=2)
        except Exception as e:
            # If ffprobe fails, return basic assessment
            pass
        
        # Basic assessment if detailed analysis not available
        return json.dumps({
            "quality_score": quality_score,
            "issues": issues,
            "improvements": improvements,
            "file_size_kb": file_size / 1024
        }, indent=2)