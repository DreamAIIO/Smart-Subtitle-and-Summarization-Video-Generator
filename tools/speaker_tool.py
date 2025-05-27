"""
Speaker recognition tool for identifying different speakers in audio.
"""
import os
import json
from typing import Dict, Optional, Any, List

import torch
import numpy as np
from pyannote.audio import Pipeline
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


class SpeakerRecognitionTool(Tool):
    """Tool for identifying different speakers in audio."""
    
    name = "speaker_recognizer"
    description = """
    Identifies and labels different speakers in audio content.
    """
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Path to the audio file"
        },
        "transcript_path": {
            "type": "string",
            "description": "Optional path to the transcript file for alignment",
            "nullable": True
        },
        "options": {
            "type": "object",
            "description": "Additional speaker recognition options",
            "nullable": True
        }
    }
    output_type = "string"
    
    # Cache for model to avoid reloading
    _diarization_pipeline = None
    
    @log_execution
    @handle_errors(default_return="Error recognizing speakers")
    def forward(self, audio_path: str, transcript_path: Optional[str] = None, 
                options: Optional[Dict[str, Any]] = None) -> str:
        """Identify speakers in audio.
        
        Args:
            audio_path: Path to the audio file
            transcript_path: Optional path to the transcript file for alignment
            options: Additional speaker recognition options
            
        Returns:
            Path to the diarized transcript file
        """
        options = options or {}
        
        if not os.path.exists(audio_path):
            return f"Audio file does not exist: {audio_path}"
        
        if transcript_path and not os.path.exists(transcript_path):
            return f"Transcript file does not exist: {transcript_path}"
        
        # Determine if we use the HuggingFace token from options or environment
        auth_token = options.get("auth_token", os.environ.get("HF_TOKEN"))
        if not auth_token:
            return "HF_TOKEN environment variable or auth_token option required for speaker recognition"
        
        # Load model
        if self._diarization_pipeline is None:
            self._diarization_pipeline = self._load_diarization_model(auth_token)
        
        # Perform speaker diarization
        diarization = self._perform_diarization(audio_path, options)
        
        # Save results with or without transcript alignment
        if transcript_path:
            output_path = self._align_with_transcript(diarization, transcript_path, options)
        else:
            output_path = self._save_diarization_only(diarization, audio_path, options)
        
        return output_path
    
    @handle_errors()
    def _load_diarization_model(self, auth_token: str) -> Pipeline:
        """Load the speaker diarization model.
        
        Args:
            auth_token: HuggingFace authentication token
            
        Returns:
            Pyannote.audio diarization pipeline
        """
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load pyannote pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        # Move to appropriate device
        pipeline.to(torch.device(device))
        
        return pipeline
    
    @handle_errors()
    def _perform_diarization(self, audio_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to the audio file
            options: Additional options
            
        Returns:
            Diarization results
        """
        # Set parameters
        num_speakers = options.get("num_speakers", None)
        min_speakers = options.get("min_speakers", None)
        max_speakers = options.get("max_speakers", None)
        
        # Apply diarization
        diarization = self._diarization_pipeline(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        return diarization
    
    @handle_errors()
    def _save_diarization_only(self, diarization: Dict[str, Any], audio_path: str, 
                                options: Dict[str, Any]) -> str:
        """Save diarization results without transcript alignment.
        
        Args:
            diarization: Diarization results
            audio_path: Path to the audio file
            options: Additional options
            
        Returns:
            Path to the output file
        """
        output_format = options.get("output_format", "json")
        output_path = options.get("output_path", None)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(audio_path)
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(dirname, f"{basename}_speakers.{output_format}")
        
        # Convert diarization results to a serializable format
        results = []
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
        
        # Sort by start time
        results.sort(key=lambda x: x["start"])
        
        # Save results
        if output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif output_format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(f"{item['start']:.2f} -> {item['end']:.2f}: {item['speaker']}\n")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_path
    
    @handle_errors()
    def _align_with_transcript(self, diarization: Dict[str, Any], transcript_path: str, 
                                options: Dict[str, Any]) -> str:
        """Align diarization results with a transcript.
        
        Args:
            diarization: Diarization results
            transcript_path: Path to the transcript file
            options: Additional options
            
        Returns:
            Path to the aligned transcript file
        """
        output_format = options.get("output_format", "json")
        output_path = options.get("output_path", None)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(transcript_path)
            basename = os.path.splitext(os.path.basename(transcript_path))[0]
            output_path = os.path.join(dirname, f"{basename}_with_speakers.{output_format}")
        
        # Load transcript
        file_ext = os.path.splitext(transcript_path)[1].lower()
        
        if file_ext == '.json':
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            
            # Extract segments from the transcript
            segments = transcript.get("segments", [])
            
            # Create a map of speaker segments
            speaker_segments = []
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker
                })
            
            # Assign speakers to transcript segments
            for segment in segments:
                segment_start = segment.get("start", 0)
                segment_end = segment.get("end", 0)
                
                # Find overlapping speaker segments
                overlapping_speakers = []
                max_overlap = 0
                assigned_speaker = None
                
                for speaker_segment in speaker_segments:
                    overlap_start = max(segment_start, speaker_segment["start"])
                    overlap_end = min(segment_end, speaker_segment["end"])
                    
                    if overlap_end > overlap_start:
                        overlap_duration = overlap_end - overlap_start
                        
                        if overlap_duration > max_overlap:
                            max_overlap = overlap_duration
                            assigned_speaker = speaker_segment["speaker"]
                
                # Assign the speaker with maximum overlap
                segment["speaker"] = assigned_speaker
            
            # Save the updated transcript
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
                
        elif file_ext == '.srt':
            import pysrt
            
            # Load subtitles
            subtitles = pysrt.open(transcript_path)
            
            # Create a map of speaker segments
            speaker_segments = []
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker
                })
            
            # Assign speakers to subtitles
            for subtitle in subtitles:
                # Get start and end time in seconds
                start_time = subtitle.start.hours * 3600 + subtitle.start.minutes * 60 + subtitle.start.seconds + subtitle.start.milliseconds / 1000
                end_time = subtitle.end.hours * 3600 + subtitle.end.minutes * 60 + subtitle.end.seconds + subtitle.end.milliseconds / 1000
                
                # Find overlapping speaker segments
                max_overlap = 0
                assigned_speaker = None
                
                for speaker_segment in speaker_segments:
                    overlap_start = max(start_time, speaker_segment["start"])
                    overlap_end = min(end_time, speaker_segment["end"])
                    
                    if overlap_end > overlap_start:
                        overlap_duration = overlap_end - overlap_start
                        
                        if overlap_duration > max_overlap:
                            max_overlap = overlap_duration
                            assigned_speaker = speaker_segment["speaker"]
                
                # Add speaker to subtitle text
                if assigned_speaker:
                    subtitle.text = f"[{assigned_speaker}] {subtitle.text}"
            
            # Save the updated subtitles
            subtitles.save(output_path, encoding='utf-8')
            
        else:
            raise ValueError(f"Unsupported transcript format: {file_ext}")
        
        return output_path
    
    @handle_errors()
    def identify_speakers(self, audio_path: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """Identify the number of speakers and their speaking time.
        
        Args:
            audio_path: Path to the audio file
            auth_token: Optional authentication token
            
        Returns:
            Dictionary with speaker information
        """
        # Use token from parameters or environment
        token = auth_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN environment variable or auth_token required")
        
        # Load model if needed
        if self._diarization_pipeline is None:
            self._diarization_pipeline = self._load_diarization_model(token)
        
        # Perform diarization
        diarization = self._diarization_pipeline(audio_path)
        
        # Extract speaker information
        speakers = {}
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            # Calculate segment duration
            duration = segment.end - segment.start
            
            # Update speaker stats
            if speaker not in speakers:
                speakers[speaker] = {
                    "total_time": 0,
                    "segments": []
                }
            
            speakers[speaker]["total_time"] += duration
            speakers[speaker]["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "duration": duration
            })
        
        # Calculate additional statistics
        result = {
            "num_speakers": len(speakers),
            "speakers": speakers
        }
        
        return result