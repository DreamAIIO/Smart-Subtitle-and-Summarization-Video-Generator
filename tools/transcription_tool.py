"""
Enhanced transcription tool for converting speech to text with multi-pass processing.
"""
import os
import json
import time
from typing import Dict, Optional, Any, List, Tuple

import whisperx
import torch
import numpy as np
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


class EnhancedTranscriptionTool(Tool):
    """Enhanced tool for transcribing audio to text with multi-pass processing."""
    
    name = "enhanced_transcriber"
    description = """
    Transcribes audio to text with timestamps using a multi-pass approach for
    improved accuracy. Can detect difficult segments and retry with different parameters.
    """
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Path to the audio file to transcribe"
        },
        "language": {
            "type": "string",
            "description": "Language code (optional, will be auto-detected if not provided)",
            "nullable": True
        },
        "options": {
            "type": "object",
            "description": "Additional transcription options",
            "nullable": True
        }
    }
    output_type = "string"
    
    # Cache models to avoid reloading
    _whisper_model = None
    _alignment_model = None
    
    @log_execution
    @handle_errors(default_return="Error transcribing audio")
    def forward(self, audio_path: str, language: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
        """Transcribe audio to text with a multi-pass approach.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (optional, will be auto-detected if not provided)
            options: Additional transcription options
            
        Returns:
            Path to the transcript file
        """
        options = options or {}
        
        if not os.path.exists(audio_path):
            return f"Audio file does not exist: {audio_path}"
        
        # Get output format
        output_format = options.get("output_format", "json")
        output_path = options.get("output_path", None)
        use_multi_pass = options.get("use_multi_pass", True)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(audio_path)
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(dirname, f"{basename}_transcript.{output_format}")
        
        try:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            # Get model size based on options
            model_size = options.get("model_size", "medium")
            
            # Phase 1: Initial transcription
            initial_result = self._perform_initial_transcription(
                audio_path, language, model_size, device, compute_type, options
            )
            
            # If multi-pass is disabled or not needed, return the initial result
            if not use_multi_pass or not initial_result.get("segments", []):
                self._save_transcript(initial_result, output_path, output_format)
                return output_path
            
            # Phase 2: Identify problematic segments
            problematic_segments = self._identify_problematic_segments(initial_result)
            
            if not problematic_segments:
                # No problematic segments found, save and return initial transcript
                self._save_transcript(initial_result, output_path, output_format)
                return output_path
            
            # Phase 3: Process problematic segments with different parameters
            improved_result = self._process_problematic_segments(
                audio_path, initial_result, problematic_segments,
                language, device, compute_type, options
            )
            
            # Save the improved transcript
            self._save_transcript(improved_result, output_path, output_format)
            
            return output_path
        except Exception as e:
            # Last-resort fallback: create a basic transcript
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    fallback_text = f"Error during transcription: {str(e)}\n\nThis is a placeholder transcript."
                    if output_format == "json":
                        json.dump({
                            "language": language or "en",
                            "text": fallback_text,
                            "segments": [{"start": 0, "end": 10, "text": fallback_text}]
                        }, f, ensure_ascii=False, indent=2)
                    else:
                        f.write(fallback_text)
                return output_path
            except:
                # If all else fails, just return the error
                raise RuntimeError(f"Transcription error: {str(e)}") from e
    
    @handle_errors()
    def _perform_initial_transcription(self, audio_path: str, language: Optional[str],
                                     model_size: str, device: str, compute_type: str,
                                     options: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the initial transcription of the audio with improved parameter handling.
        
        Args:
            audio_path: Path to the audio file
            language: Language code
            model_size: Whisper model size
            device: Device to use (cuda/cpu)
            compute_type: Compute type (float16/int8)
            options: Additional options
            
        Returns:
            Transcription results
        """
        # Load WhisperX model
        if self._whisper_model is None or options.get("reload_model", False):
            self._whisper_model = whisperx.load_model(
                model_size, 
                device=device,
                compute_type=compute_type,
                language=language,
                asr_options={"word_timestamps": True}
            )
        
        # Apply custom parameters for batch size
        batch_size = options.get("batch_size", 16)
        beam_size = options.get("beam_size", 5)
        
        # Try with beam_size parameter first, fallback to without it if needed
        try:
            result = self._whisper_model.transcribe(
                audio_path,
                batch_size=batch_size,
                beam_size=beam_size
            )
        except TypeError as e:
            if "unexpected keyword argument 'beam_size'" in str(e):
                # Retry without beam_size parameter
                print("Retrying transcription without beam_size parameter")
                result = self._whisper_model.transcribe(
                    audio_path,
                    batch_size=batch_size
                )
            else:
                raise e
        
        # Ensure the result has the expected format
        if not isinstance(result, dict):
            result = {"language": language or "en", "segments": [{"start": 0, "end": 10, "text": str(result)}]}
        
        # Ensure segments is a list of dictionaries
        if "segments" not in result or not isinstance(result["segments"], list):
            result["segments"] = [{"start": 0, "end": 10, "text": "Transcription completed but no segments found"}]
        
        # Validate each segment
        validated_segments = []
        for segment in result["segments"]:
            if not isinstance(segment, dict):
                segment = {"start": 0, "end": 10, "text": str(segment)}
            if "text" not in segment:
                segment["text"] = "No text in segment"
            if "start" not in segment:
                segment["start"] = 0
            if "end" not in segment:
                segment["end"] = 10
            validated_segments.append(segment)
        
        result["segments"] = validated_segments
        
        # Align timestamps at word level if possible
        try:
            if "language" in result and result["language"]:
                language_code = result["language"]
                
                if self._alignment_model is None or options.get("reload_model", False):
                    self._alignment_model = whisperx.load_align_model(language_code=language_code, device=device)
                
                aligned_result = whisperx.align(
                    result["segments"],
                    self._alignment_model,
                    language_code,
                    audio_path,
                    device
                )
                
                # Check if alignment succeeded and has the right format
                if isinstance(aligned_result, dict) and "segments" in aligned_result:
                    result = aligned_result
        except Exception as align_error:
            print(f"Warning: Could not align timestamps: {str(align_error)}")
        
        # Extract the full transcript text
        full_transcript = ""
        for segment in result["segments"]:
            if isinstance(segment, dict) and "text" in segment:
                full_transcript += segment["text"] + " "
        
        result["text"] = full_transcript.strip()
        
        return result
    
    @handle_errors()
    def _identify_problematic_segments(self, transcript: Dict[str, Any]) -> List[int]:
        """Identify problematic segments that may need reprocessing.
        
        Args:
            transcript: The transcript data
            
        Returns:
            List of indices of problematic segments
        """
        problematic_indices = []
        segments = transcript.get("segments", [])
        
        for i, segment in enumerate(segments):
            # Skip non-dict segments
            if not isinstance(segment, dict):
                continue
                
            text = segment.get("text", "").strip()
            
            # Check for potentially problematic segments
            has_problems = False
            
            # Very short segments with too little content
            if len(text.split()) <= 2 and len(text) < 10:
                has_problems = True
            
            # Segments with many non-alphabetic characters (potential noise)
            alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
            if len(text) > 5 and alpha_ratio < 0.5:
                has_problems = True
            
            # Repeated characters or words (stuttering or noise)
            words = text.lower().split()
            if len(words) >= 3:
                repeated_words = sum(words[i] == words[i+1] for i in range(len(words)-1))
                repeat_ratio = repeated_words / len(words)
                if repeat_ratio > 0.4:
                    has_problems = True
            
            # Segments with unusual timing
            duration = segment.get("end", 0) - segment.get("start", 0)
            if duration > 0:
                chars_per_second = len(text) / duration
                # Either extremely fast or slow speech could indicate issues
                if chars_per_second > 30 or (len(text) > 10 and chars_per_second < 3):
                    has_problems = True
            
            # Check for gaps between segments
            if i > 0 and isinstance(segments[i-1], dict):
                prev_end = segments[i-1].get("end", 0)
                curr_start = segment.get("start", 0)
                # Gap of more than 1 second but less than 5 seconds might indicate missed speech
                if 1.0 < curr_start - prev_end < 5.0:
                    has_problems = True
            
            if has_problems:
                problematic_indices.append(i)
        
        return problematic_indices
    
    @handle_errors()
    def _process_problematic_segments(self, audio_path: str, transcript: Dict[str, Any], 
                                    problematic_indices: List[int], language: Optional[str],
                                    device: str, compute_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process problematic segments with different parameters for improved accuracy.
        
        Args:
            audio_path: Path to the audio file
            transcript: Original transcript data
            problematic_indices: Indices of segments to reprocess
            language: Language code
            device: Device to use (cuda/cpu)
            compute_type: Compute type (float16/int8)
            options: Additional options
            
        Returns:
            Improved transcript
        """
        # Try to import librosa, but gracefully handle if not available
        try:
            import librosa
            librosa_available = True
        except ImportError:
            print("Warning: librosa not available, using fallback method for processing segments")
            librosa_available = False
        
        # Copy the original transcript to preserve structure
        improved_transcript = transcript.copy()
        segments = improved_transcript.get("segments", [])
        
        # Skip if no segments or no problematic segments
        if not segments or not problematic_indices:
            return improved_transcript
        
        # Group adjacent problematic segments for batch processing
        segment_groups = []
        current_group = [problematic_indices[0]]
        
        for i in range(1, len(problematic_indices)):
            # If segments are adjacent or close, add to the current group
            if problematic_indices[i] - problematic_indices[i-1] <= 2:
                current_group.append(problematic_indices[i])
            else:
                segment_groups.append(current_group)
                current_group = [problematic_indices[i]]
        
        # Add the last group
        if current_group:
            segment_groups.append(current_group)
        
        # Load audio file if librosa is available
        if librosa_available:
            try:
                audio_data, sample_rate = librosa.load(audio_path, sr=None)
            except Exception as e:
                print(f"Error loading audio with librosa: {str(e)}")
                librosa_available = False
        
        # Process each group of problematic segments
        for group in segment_groups:
            # Determine the start and end time for this group
            start_idx = max(0, min(group) - 1)
            end_idx = min(len(segments) - 1, max(group) + 1)
            
            start_time = segments[start_idx].get("start", 0) if isinstance(segments[start_idx], dict) else 0
            end_time = segments[end_idx].get("end", 0) if isinstance(segments[end_idx], dict) else 0
            
            # Add a small buffer
            start_time = max(0, start_time - 0.5)
            end_time = min(end_time + 0.5, float('inf'))
            
            # Handle the segment differently based on whether librosa is available
            if librosa_available:
                self._process_segment_with_librosa(
                    audio_data, sample_rate, start_time, end_time, 
                    group, segments, improved_transcript, language, device, compute_type, options
                )
            else:
                self._process_segment_fallback(
                    audio_path, start_time, end_time, 
                    group, segments, improved_transcript, language, device, compute_type, options
                )
        
        # Rebuild the full text from the improved segments
        full_text = ""
        for segment in improved_transcript["segments"]:
            if isinstance(segment, dict) and "text" in segment:
                full_text += segment["text"] + " "
        
        improved_transcript["text"] = full_text.strip()
        
        return improved_transcript
    
    def _process_segment_with_librosa(self, audio_data, sample_rate, start_time, end_time,
                                     group, segments, improved_transcript, language, 
                                     device, compute_type, options):
        """Process an audio segment using librosa for extraction."""
        # Extract the audio segment
        start_sample = int(start_time * sample_rate)
        end_sample = min(int(end_time * sample_rate), len(audio_data))
        
        if start_sample >= end_sample or start_sample >= len(audio_data):
            return
            
        segment_audio = audio_data[start_sample:end_sample]
        
        # Save the segment to a temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        try:
            import soundfile as sf
            sf.write(temp_filename, segment_audio, sample_rate)
            
            # Process this segment with different parameters
            model_size = options.get("model_size", "medium")
            if len(segment_audio) / sample_rate < 10:  # Very short segment
                model_size = options.get("fine_model_size", "large")
            
            # Use a larger beam size and more computational resources for this segment
            asr_options = {
                "word_timestamps": True
            }
            
            # Try to use beam_size if possible
            try:
                asr_options["beam_size"] = options.get("fine_beam_size", 10)
                asr_options["best_of"] = options.get("fine_best_of", 5)
            except:
                # Ignore if these parameters cause issues
                pass
            
            # Load a new model instance for fine-grained processing
            fine_model = whisperx.load_model(
                model_size, 
                device=device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options
            )
            
            # Try transcribing with beam_size first
            try:
                segment_result = fine_model.transcribe(
                    temp_filename,
                    batch_size=options.get("fine_batch_size", 8),
                    beam_size=options.get("fine_beam_size", 10)
                )
            except TypeError as e:
                if "unexpected keyword argument 'beam_size'" in str(e):
                    # Retry without beam_size
                    segment_result = fine_model.transcribe(
                        temp_filename,
                        batch_size=options.get("fine_batch_size", 8)
                    )
                else:
                    raise e
            
            # Update the segments
            self._update_transcript_with_segment_result(
                segment_result, start_time, group, segments, improved_transcript
            )
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    def _process_segment_fallback(self, audio_path, start_time, end_time,
                                 group, segments, improved_transcript, language, 
                                 device, compute_type, options):
        """Process an audio segment using FFmpeg for extraction as fallback."""
        import subprocess
        import tempfile
        import uuid
        
        # Create a temporary file for the segment
        temp_dir = tempfile.gettempdir()
        temp_id = uuid.uuid4().hex
        temp_filename = os.path.join(temp_dir, f"temp_segment_{temp_id}.wav")
        
        try:
            # Use FFmpeg to extract the segment
            duration = end_time - start_time
            subprocess.run([
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:a', 'pcm_s16le',
                '-y',
                temp_filename
            ], check=True, capture_output=True)
            
            # Process with WhisperX
            model_size = options.get("model_size", "medium")
            
            # Use a larger beam size and more computational resources for this segment
            asr_options = {
                "word_timestamps": True
            }
            
            # Load a new model instance for fine-grained processing
            fine_model = whisperx.load_model(
                model_size, 
                device=device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options
            )
            
            # Try transcribing with beam_size first
            try:
                segment_result = fine_model.transcribe(
                    temp_filename,
                    batch_size=options.get("fine_batch_size", 8),
                    beam_size=options.get("fine_beam_size", 10)
                )
            except TypeError as e:
                if "unexpected keyword argument 'beam_size'" in str(e):
                    # Retry without beam_size
                    segment_result = fine_model.transcribe(
                        temp_filename,
                        batch_size=options.get("fine_batch_size", 8)
                    )
                else:
                    raise e
            
            # Update the segments
            self._update_transcript_with_segment_result(
                segment_result, start_time, group, segments, improved_transcript
            )
            
        except Exception as e:
            print(f"Error in segment fallback processing: {str(e)}")
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except:
                pass
    
    def _update_transcript_with_segment_result(self, segment_result, start_time, group, 
                                             segments, improved_transcript):
        """Update the transcript with results from a processed segment."""
        # Align the new transcription times to the original timeline
        new_segments = segment_result.get("segments", [])
        for new_segment in new_segments:
            if isinstance(new_segment, dict):
                new_segment["start"] = start_time + new_segment.get("start", 0)
                new_segment["end"] = start_time + new_segment.get("end", 0)
        
        # Replace the problematic segments with the new ones
        start_idx = min(group)
        end_idx = max(group)
        
        # Remove the problematic segments
        updated_segments = []
        for i, segment in enumerate(segments):
            if i < start_idx or i > end_idx:
                updated_segments.append(segment)
        
        # Insert the new segments at the right position
        position = start_idx
        for new_segment in new_segments:
            updated_segments.insert(position, new_segment)
            position += 1
        
        # Update the transcript
        improved_transcript["segments"] = sorted(
            updated_segments,
            key=lambda x: x.get("start", 0) if isinstance(x, dict) else 0
        )
    
    @handle_errors()
    def _save_transcript(self, transcript: Dict[str, Any], output_path: str, output_format: str) -> None:
        """Save the transcript in the specified format.
        
        Args:
            transcript: Transcript data
            output_path: Path to save the transcript
            output_format: Output format (json, txt, srt)
        """
        # Extract the full text
        full_transcript = transcript.get("text", "")
        
        if not full_transcript and "segments" in transcript:
            full_transcript = " ".join([
                segment.get("text", "") for segment in transcript["segments"]
                if isinstance(segment, dict)
            ])
        
        # Save the transcript
        if output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
        elif output_format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_transcript)
                f.write("\n\n--- Segments with timestamps ---\n\n")
                for segment in transcript.get("segments", []):
                    if isinstance(segment, dict) and "text" in segment:
                        start = segment.get("start", 0)
                        end = segment.get("end", 0)
                        text = segment.get("text", "")
                        f.write(f"{start:.2f} -> {end:.2f}: {text}\n")
        elif output_format == "srt":
            self._save_as_srt(transcript.get("segments", []), output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    @staticmethod
    def _save_as_srt(segments: List[Dict[str, Any]], output_path: str) -> None:
        """Save transcript segments in SRT format.
        
        Args:
            segments: List of transcript segments
            output_path: Path to save the SRT file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    try:
                        # Get start and end times, handling both dictionary and direct access
                        if isinstance(segment, dict):
                            start_seconds = segment.get("start", 0)
                            end_seconds = segment.get("end", 0)
                            text = segment.get("text", "No text available")
                        else:
                            # If segment is not a dict, use default values
                            start_seconds = 0
                            end_seconds = 10
                            text = str(segment)
                        
                        # Convert timestamps to SRT format (HH:MM:SS,mmm)
                        start = _format_timestamp(start_seconds)
                        end = _format_timestamp(end_seconds)
                        
                        # Write SRT entry
                        f.write(f"{i}\n")
                        f.write(f"{start} --> {end}\n")
                        f.write(f"{text.strip()}\n\n")
                    except Exception as e:
                        # Handle individual segment errors
                        f.write(f"{i}\n")
                        f.write("00:00:00,000 --> 00:00:10,000\n")
                        f.write(f"Error processing segment: {str(e)}\n\n")
        except Exception as e:
            # If file writing fails completely, create a minimal SRT
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("1\n")
                    f.write("00:00:00,000 --> 00:00:10,000\n")
                    f.write(f"Error creating subtitles: {str(e)}\n\n")
            except:
                pass


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")