"""
Enhanced subtitle tool for generating and manipulating subtitle files with improved accuracy.
Specifically addresses issues with complete text preservation and proper segment handling.
"""
import os
import json
import re
from typing import Dict, Optional, Any, List, Tuple

import pysrt
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


class EnhancedSubtitleTool(Tool):
    """Tool for generating and manipulating subtitle files with improved accuracy."""
    
    name = "enhanced_subtitle_processor"
    description = """
    Generates and manipulates subtitle files with enhanced formatting and timing accuracy.
    Supports multi-pass processing for improved quality and preserves complete text.
    """
    inputs = {
        "input_path": {
            "type": "string",
            "description": "Path to the input file (transcript or subtitle file)",
            "nullable": True
        },
        "operation": {
            "type": "string",
            "description": "Operation to perform (format_subtitles, adjust_timing, optimize_subtitles, etc.)"
        },
        "options": {
            "type": "object",
            "description": "Additional options for the operation",
            "nullable": True
        }
    }
    output_type = "string"
    
    # Maximum characters per subtitle line
    DEFAULT_MAX_CHARS_PER_LINE = 42
    
    # Maximum words per subtitle entry
    DEFAULT_MAX_WORDS_PER_ENTRY = 14
    
    # Default subtitle duration limits (in seconds)
    MIN_SUBTITLE_DURATION = 1.0
    MAX_SUBTITLE_DURATION = 7.0
    
    @log_execution
    @handle_errors(default_return="Error processing subtitles")
    def forward(self, input_path: Optional[str], operation: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Process subtitles based on the specified operation with enhanced quality."""
        options = options or {}
        
        # Dispatch to the appropriate method based on the operation
        try:
            if operation == "format_subtitles":
                return self._format_subtitles(input_path, options)
            elif operation == "adjust_timing":
                return self._adjust_timing(input_path, options)
            elif operation == "optimize_subtitles":
                return self._optimize_subtitles(input_path, options)
            elif operation == "verify_synchronization":
                return self._verify_synchronization(input_path, options)
            elif operation == "convert_format":
                return self._convert_format(input_path, options)
            elif operation == "style_subtitles":
                return self._style_subtitles(input_path, options)
            else:
                return f"Unsupported operation: {operation}"
        except Exception as e:
            # If the regular operation fails, try the fallback for format_subtitles
            if operation == "format_subtitles":
                try:
                    # Try to extract text content from input if possible
                    extracted_text = ""
                    if input_path and os.path.exists(input_path):
                        with open(input_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Try to find actual text content
                        if content.strip():
                            extracted_text = content
                    
                    # If we couldn't get text from file, check options
                    if not extracted_text and options.get("subtitle_text"):
                        extracted_text = options.get("subtitle_text")
                    elif not extracted_text and options.get("content"):
                        extracted_text = options.get("content")
                    
                    # If we still don't have text, use a placeholder
                    if not extracted_text:
                        extracted_text = "Subtitle generation failed. This is a placeholder."
                    
                    # Generate a fallback output path if needed
                    output_path = options.get("output_path", None)
                    if not output_path:
                        dirname = os.path.dirname(input_path) if input_path else os.getcwd()
                        basename = "fallback_subtitles.srt"
                        output_path = os.path.join(dirname, basename)
                    
                    # Create basic subtitles from the text
                    return self._create_basic_srt_from_text(extracted_text, output_path)
                except Exception as fallback_error:
                    # If even the fallback fails, return the original error
                    return f"Error processing subtitles: {str(e)}\nFallback error: {str(fallback_error)}"
            else:
                # For other operations, just return the error
                return f"Error in {operation}: {str(e)}"
    
    def _format_text_for_display(self, text: str, max_chars_per_line: int) -> str:
        """Format text for display in subtitles with appropriate line breaks.
        
        Args:
            text: Text to format
            max_chars_per_line: Maximum characters per line
            
        Returns:
            Formatted text with appropriate line breaks
        """
        # Remove extra whitespace
        text = text.strip()
        if not text:
            return ""
        
        # If the text is already short enough, return it as is
        if len(text) <= max_chars_per_line:
            return text
        
        # Split text into words
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed the max characters per line
            if len(current_line) + len(word) + (1 if current_line else 0) <= max_chars_per_line:
                # Add the word to the current line
                current_line = current_line + " " + word if current_line else word
            else:
                # Add the current line to the lines list and start a new line
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        # Add the last line if it exists
        if current_line:
            lines.append(current_line)
        
        # Join the lines with line breaks
        return "\n".join(lines)
    
    @handle_errors()
    def _format_subtitles(self, input_path: Optional[str], options: Dict[str, Any]) -> str:
        """Format subtitles for better readability and distribution with enhanced rules.
        
        Args:
            input_path: Path to the subtitle or transcript file
            options: Additional options for formatting
            
        Returns:
            Path to the formatted subtitle file
        """
        # Get configuration options with defaults
        max_chars_per_line = options.get("max_chars_per_line", self.DEFAULT_MAX_CHARS_PER_LINE)
        max_words_per_entry = options.get("max_words_per_entry", self.DEFAULT_MAX_WORDS_PER_ENTRY)
        min_duration = options.get("min_duration", self.MIN_SUBTITLE_DURATION)
        max_duration = options.get("max_duration", self.MAX_SUBTITLE_DURATION)
        preserve_full_text = options.get("preserve_full_text", True)
        reading_speed = options.get("reading_speed", 20)  # Characters per second
        
        output_path = options.get("output_path", None)
        subtitle_text = options.get("subtitle_text", options.get("content", None))
        
        if output_path is None:
            # Generate output path if not provided
            if input_path:
                dirname = os.path.dirname(input_path)
                basename = os.path.splitext(os.path.basename(input_path))[0]
            else:
                dirname = os.getcwd()
                basename = "generated"
            output_path = os.path.join(dirname, f"{basename}.srt")
        
        # Process based on input type
        if subtitle_text:
            # Create subtitles from text input
            return self._create_subtitles_from_text(subtitle_text, output_path, options)
        elif input_path and os.path.exists(input_path):
            # Process based on file type
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.json':
                return self._format_from_json_transcript(input_path, output_path, options)
            elif file_ext == '.srt':
                return self._format_from_srt(input_path, output_path, options)
            elif file_ext == '.txt':
                # Create subtitles from text file
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                return self._create_subtitles_from_text(text, output_path, options)
            else:
                raise ValueError(f"Unsupported input format: {file_ext}")
        else:
            # If no input file and no subtitle text, create a dummy subtitle
            return self._create_basic_srt_from_text(
                "Generated subtitle placeholder. Replace with actual content.",
                output_path
            )
    
    @handle_errors()
    def _format_from_json_transcript(self, json_path: str, output_path: str, options: Dict[str, Any]) -> str:
        """Format subtitles from a JSON transcript file with proper text handling.
        
        Args:
            json_path: Path to the JSON transcript file
            output_path: Path to save the formatted subtitle file
            options: Additional options for formatting
            
        Returns:
            Path to the formatted subtitle file
        """
        # Load the JSON transcript
        with open(json_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # Extract segments and text
        segments = transcript.get("segments", [])
        
        if not segments:
            raise ValueError("No segments found in the JSON transcript")
        
        # Create a subtitles object
        subtitles = pysrt.SubRipFile()
        
        # Get formatting options
        max_chars_per_line = options.get("max_chars_per_line", self.DEFAULT_MAX_CHARS_PER_LINE)
        max_words_per_entry = options.get("max_words_per_entry", self.DEFAULT_MAX_WORDS_PER_ENTRY)
        min_duration = options.get("min_duration", self.MIN_SUBTITLE_DURATION)
        max_duration = options.get("max_duration", self.MAX_SUBTITLE_DURATION)
        preserve_full_text = options.get("preserve_full_text", True)
        
        # Process each segment
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
                
            # Extract segment data
            text = segment.get("text", "").strip()
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            if not text:
                continue
                
            # Ensure minimum duration
            duration = end_time - start_time
            if duration < min_duration:
                end_time = start_time + min_duration
            
            # Ensure maximum duration
            if duration > max_duration:
                # Split the segment into multiple parts
                self._add_split_subtitle_entries(
                    subtitles, text, start_time, end_time, 
                    max_chars_per_line, max_words_per_entry, max_duration
                )
            else:
                # Add as a single entry with proper formatting
                subtitle_item = pysrt.SubRipItem()
                subtitle_item.index = len(subtitles) + 1
                
                # Set timing
                subtitle_item.start.seconds = int(start_time)
                subtitle_item.start.milliseconds = int((start_time % 1) * 1000)
                subtitle_item.end.seconds = int(end_time)
                subtitle_item.end.milliseconds = int((end_time % 1) * 1000)
                
                # Format text for better display
                subtitle_item.text = self._format_text_for_display(text, max_chars_per_line)
                
                # Add to subtitles
                subtitles.append(subtitle_item)
        
        # Save the formatted subtitles
        subtitles.save(output_path, encoding='utf-8')
        
        return output_path
    
    def _add_split_subtitle_entries(self, subtitles, text, start_time, end_time, 
                                   max_chars_per_line, max_words_per_entry, max_duration):
        """Split long text into multiple subtitle entries with appropriate timing.
        
        Args:
            subtitles: The SubRipFile object to add entries to
            text: The full text to split
            start_time: Start time of the segment
            end_time: End time of the segment
            max_chars_per_line: Maximum characters per line
            max_words_per_entry: Maximum words per entry
            max_duration: Maximum duration per entry
        """
        # Calculate total duration
        total_duration = end_time - start_time
        
        # Split text into words
        words = text.split()
        
        if not words:
            return
            
        # Calculate how many parts we need
        num_parts = max(1, min(len(words) // max_words_per_entry + 1, 
                              int(total_duration / max_duration) + 1))
        
        # Calculate words per part
        words_per_part = len(words) // num_parts
        if words_per_part < 1:
            words_per_part = 1
            
        # Calculate time per part
        time_per_part = total_duration / num_parts
        
        # Create subtitle entries
        for i in range(num_parts):
            # Calculate start index and end index for words
            start_idx = i * words_per_part
            end_idx = min(start_idx + words_per_part, len(words))
            
            # Last part gets all remaining words
            if i == num_parts - 1:
                end_idx = len(words)
            
            # Skip if no words in this part
            if start_idx >= end_idx:
                continue
                
            # Get the text for this part
            part_text = " ".join(words[start_idx:end_idx])
            
            # Calculate timing for this part
            part_start = start_time + (i * time_per_part)
            part_end = part_start + time_per_part
            
            # For the last part, ensure we end at the original end time
            if i == num_parts - 1:
                part_end = end_time
            
            # Create the subtitle item
            subtitle_item = pysrt.SubRipItem()
            subtitle_item.index = len(subtitles) + 1
            
            # Set timing
            subtitle_item.start.seconds = int(part_start)
            subtitle_item.start.milliseconds = int((part_start % 1) * 1000)
            subtitle_item.end.seconds = int(part_end)
            subtitle_item.end.milliseconds = int((part_end % 1) * 1000)
            
            # Format text for better display
            subtitle_item.text = self._format_text_for_display(part_text, max_chars_per_line)
            
            # Add to subtitles
            subtitles.append(subtitle_item)
    
    def _create_subtitles_from_text(self, text: str, output_path: str, options: Dict[str, Any]) -> str:
        """Create subtitles from text with automatic timing.
        
        Args:
            text: Text to convert to subtitles
            output_path: Path to save the subtitle file
            options: Additional options for formatting
            
        Returns:
            Path to the subtitle file
        """
        # Get configuration options with defaults
        max_chars_per_line = options.get("max_chars_per_line", self.DEFAULT_MAX_CHARS_PER_LINE)
        words_per_second = options.get("words_per_second", 2.5)  # Typical speaking rate
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Create subtitles
        subtitles = pysrt.SubRipFile()
        
        # Current time in seconds
        current_time = 0
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Create subtitle item
            subtitle_item = pysrt.SubRipItem()
            subtitle_item.index = len(subtitles) + 1
            
            # Estimate duration based on number of words
            words = sentence.split()
            duration = max(self.MIN_SUBTITLE_DURATION, len(words) / words_per_second)
            
            # Set timing
            subtitle_item.start.seconds = int(current_time)
            subtitle_item.start.milliseconds = int((current_time % 1) * 1000)
            subtitle_item.end.seconds = int(current_time + duration)
            subtitle_item.end.milliseconds = int(((current_time + duration) % 1) * 1000)
            
            # Format text for better display
            subtitle_item.text = self._format_text_for_display(sentence, max_chars_per_line)
            
            # Add to subtitles
            subtitles.append(subtitle_item)
            
            # Update current time
            current_time += duration
        
        # Save the subtitle file
        subtitles.save(output_path, encoding='utf-8')
        
        return output_path
    
    @handle_errors()
    def _style_subtitles(self, input_path: str, options: Dict[str, Any]) -> str:
        """Apply enhanced styling to subtitles.
        
        Args:
            input_path: Path to the subtitle file
            options: Additional options for styling
            
        Returns:
            Path to the styled subtitle file
        """
        if not os.path.exists(input_path):
            return f"File does not exist: {input_path}"
        
        style = options.get("style", "default")
        output_path = options.get("output_path", None)
        speaker_map = options.get("speaker_map", {})
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(input_path)
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(dirname, f"{basename}_styled.srt")
        
        # Load subtitles
        subtitles = pysrt.open(input_path)
        
        # Apply styling based on selected style
        if style == "default":
            # No styling changes
            pass
        elif style == "uppercase":
            # Convert all text to uppercase
            for subtitle in subtitles:
                subtitle.text = subtitle.text.upper()
        elif style == "italics":
            # Add italics formatting
            for subtitle in subtitles:
                subtitle.text = f"<i>{subtitle.text}</i>"
        elif style == "color":
            # Apply color formatting based on line position
            colors = options.get("colors", ["#FFFFFF", "#FFFF00", "#00FFFF", "#FF00FF"])
            for subtitle in subtitles:
                lines = subtitle.text.split('\n')
                styled_lines = []
                for i, line in enumerate(lines):
                    color_idx = min(i, len(colors) - 1)
                    styled_lines.append(f'<font color="{colors[color_idx]}">{line}</font>')
                subtitle.text = '\n'.join(styled_lines)
        elif style == "speaker":
            # Apply styling based on speaker identification
            if not speaker_map:
                # Default speaker colors if not provided
                speaker_colors = {
                    "SPEAKER_1": "#FFFFFF",
                    "SPEAKER_2": "#FFFF00",
                    "SPEAKER_3": "#00FFFF",
                    "SPEAKER_4": "#FF00FF"
                }
            else:
                speaker_colors = speaker_map
            
            for subtitle in subtitles:
                # Extract speaker label if present
                text = subtitle.text
                speaker_match = re.match(r'\[([^\]]+)\](.*)', text)
                
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
                    content = speaker_match.group(2).strip()
                    
                    # Apply color based on speaker
                    color = speaker_colors.get(speaker, "#FFFFFF")
                    subtitle.text = f'<font color="{color}">[{speaker}] {content}</font>'
        else:
            raise ValueError(f"Unsupported style: {style}")
        
        # Save styled subtitles
        subtitles.save(output_path, encoding='utf-8')
        
        return output_path
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using more sophisticated rules.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Clean up the text
        text = text.replace('\n', ' ').strip()
        
        try:
            # Try to use NLTK for better sentence splitting
            import nltk.tokenize
            try:
                sentences = nltk.tokenize.sent_tokenize(text)
            except LookupError:
                # If NLTK data is not available, download it
                nltk.download('punkt', quiet=True)
                sentences = nltk.tokenize.sent_tokenize(text)
        except (ImportError, LookupError):
            # Fallback to simple regex-based splitting
            sentence_endings = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_endings, text)
        
        return sentences
    
    def _create_basic_srt_from_text(self, text: str, output_path: str) -> str:
        """Create a basic SRT file from text content.
        
        Args:
            text: Text content to convert to subtitles
            output_path: Path to save the SRT file
            
        Returns:
            Path to the SRT file
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a clean SRT file with simple sentences
        subtitles = pysrt.SubRipFile()
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Create subtitle items (5 seconds per sentence)
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            item = pysrt.SubRipItem()
            item.index = i + 1
            start_sec = i * 5
            end_sec = start_sec + 5
            
            # Set timing
            item.start.seconds = int(start_sec)
            item.start.milliseconds = int((start_sec % 1) * 1000)
            item.end.seconds = int(end_sec)
            item.end.milliseconds = int((end_sec % 1) * 1000)
            
            # Set text (limit to 42 chars per line)
            item.text = self._format_text_for_display(sentence, self.DEFAULT_MAX_CHARS_PER_LINE)
            
            subtitles.append(item)
        
        # Save the subtitle file
        subtitles.save(output_path, encoding='utf-8')
        return output_path
    
    @handle_errors()
    def _format_from_srt(self, input_path: str, output_path: str, options: Dict[str, Any]) -> str:
        """Format an existing SRT file with improved display and timing.
        
        Args:
            input_path: Path to the SRT file
            output_path: Path to save the formatted SRT file
            options: Additional options for formatting
            
        Returns:
            Path to the formatted SRT file
        """
        # Get configuration options
        max_chars_per_line = options.get("max_chars_per_line", self.DEFAULT_MAX_CHARS_PER_LINE)
        min_duration = options.get("min_duration", self.MIN_SUBTITLE_DURATION)
        max_duration = options.get("max_duration", self.MAX_SUBTITLE_DURATION)
        
        # Load subtitles
        subtitles = pysrt.open(input_path)
        
        # Create a new subtitle file for the formatted subtitles
        formatted_subtitles = pysrt.SubRipFile()
        
        # Process each subtitle
        for subtitle in subtitles:
            # Calculate duration
            start_time = subtitle.start.ordinal / 1000
            end_time = subtitle.end.ordinal / 1000
            duration = end_time - start_time
            
            # Ensure minimum duration
            if duration < min_duration:
                subtitle.end.ordinal = subtitle.start.ordinal + int(min_duration * 1000)
            
            # Ensure maximum duration
            if duration > max_duration:
                # Split the subtitle for very long text
                text = subtitle.text
                text_length = len(text)
                
                if text_length > max_chars_per_line * 2:
                    # Split into multiple subtitles
                    parts = []
                    for i in range(0, text_length, max_chars_per_line * 2):
                        parts.append(text[i:i + max_chars_per_line * 2])
                    
                    part_duration = duration / len(parts)
                    
                    for i, part in enumerate(parts):
                        new_subtitle = pysrt.SubRipItem()
                        new_subtitle.index = len(formatted_subtitles) + 1
                        
                        # Set timing
                        part_start = start_time + (i * part_duration)
                        part_end = part_start + part_duration
                        
                        new_subtitle.start.seconds = int(part_start)
                        new_subtitle.start.milliseconds = int((part_start % 1) * 1000)
                        new_subtitle.end.seconds = int(part_end)
                        new_subtitle.end.milliseconds = int((part_end % 1) * 1000)
                        
                        # Format text
                        new_subtitle.text = self._format_text_for_display(part, max_chars_per_line)
                        
                        formatted_subtitles.append(new_subtitle)
                else:
                    # Just update the format
                    new_subtitle = subtitle.copy()
                    new_subtitle.index = len(formatted_subtitles) + 1
                    new_subtitle.text = self._format_text_for_display(text, max_chars_per_line)
                    formatted_subtitles.append(new_subtitle)
            else:
                # Just update the format
                new_subtitle = subtitle.copy()
                new_subtitle.index = len(formatted_subtitles) + 1
                new_subtitle.text = self._format_text_for_display(subtitle.text, max_chars_per_line)
                formatted_subtitles.append(new_subtitle)
        
        # Save the formatted subtitles
        formatted_subtitles.save(output_path, encoding='utf-8')
        
        return output_path
    
    @handle_errors()
    def _adjust_timing(self, input_path: str, options: Dict[str, Any]) -> str:
        """Adjust subtitle timing to better match speech.
        
        Args:
            input_path: Path to the subtitle file
            options: Additional options for timing adjustment
            
        Returns:
            Path to the adjusted subtitle file
        """
        # Get options
        offset = options.get("offset", 0.0)  # Time offset in seconds
        scale = options.get("scale", 1.0)  # Time scale factor
        output_path = options.get("output_path", None)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(input_path)
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(dirname, f"{basename}_adjusted.srt")
        
        # Load subtitles
        subtitles = pysrt.open(input_path)
        
        # Adjust timing
        for subtitle in subtitles:
            # Apply offset and scale to start and end times
            start_time = subtitle.start.ordinal / 1000.0
            end_time = subtitle.end.ordinal / 1000.0
            
            # Apply transformations
            new_start_time = (start_time * scale) + offset
            new_end_time = (end_time * scale) + offset
            
            # Convert back to milliseconds
            subtitle.start.ordinal = int(new_start_time * 1000)
            subtitle.end.ordinal = int(new_end_time * 1000)
        
        # Save adjusted subtitles
        subtitles.save(output_path, encoding='utf-8')
        
        return output_path
    
    @handle_errors()
    def _optimize_subtitles(self, input_path: str, options: Dict[str, Any]) -> str:
        """Optimize subtitle timing and formatting for better viewing experience.
        
        Args:
            input_path: Path to the subtitle file
            options: Additional options for optimization
            
        Returns:
            Path to the optimized subtitle file
        """
        # Get options
        output_path = options.get("output_path", None)
        max_chars_per_line = options.get("max_chars_per_line", self.DEFAULT_MAX_CHARS_PER_LINE)
        min_duration = options.get("min_duration", self.MIN_SUBTITLE_DURATION)
        max_duration = options.get("max_duration", self.MAX_SUBTITLE_DURATION)
        min_gap = options.get("min_gap", 0.2)  # Minimum gap between subtitles
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(input_path)
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(dirname, f"{basename}_optimized.srt")
        
        # Load subtitles
        subtitles = pysrt.open(input_path)
        
        # Create a new subtitle file for the optimized subtitles
        optimized_subtitles = pysrt.SubRipFile()
        
        # Process each subtitle
        for i, subtitle in enumerate(subtitles):
            # Format text
            formatted_text = self._format_text_for_display(subtitle.text, max_chars_per_line)
            
            # Calculate duration
            start_time = subtitle.start.ordinal / 1000.0
            end_time = subtitle.end.ordinal / 1000.0
            duration = end_time - start_time
            
            # Ensure minimum duration based on text length
            text_length = len(subtitle.text)
            words = subtitle.text.split()
            word_count = len(words)
            
            # Calculate minimum duration based on reading speed
            reading_speed = 0.3  # seconds per word
            min_duration_for_text = max(self.MIN_SUBTITLE_DURATION, word_count * reading_speed)
            
            # Ensure maximum duration
            actual_duration = min(max_duration, max(duration, min_duration_for_text))
            
            # Create new subtitle
            new_subtitle = pysrt.SubRipItem()
            new_subtitle.index = len(optimized_subtitles) + 1

            # Set timing
            new_subtitle.start.seconds = int(start_time)
            new_subtitle.start.milliseconds = int((start_time % 1) * 1000)
            new_subtitle.end.seconds = int(start_time + actual_duration)
            new_subtitle.end.milliseconds = int(((start_time + actual_duration) % 1) * 1000)
            
            # Set text
            new_subtitle.text = formatted_text
            
            # Adjust gap with next subtitle if needed
            if i < len(subtitles) - 1:
                next_start = subtitles[i+1].start.ordinal / 1000.0
                current_end = start_time + actual_duration
                
                if next_start - current_end < min_gap:
                    # Adjust end time to maintain minimum gap
                    new_end_time = next_start - min_gap
                    if new_end_time > start_time:  # Ensure end time is after start time
                        new_subtitle.end.seconds = int(new_end_time)
                        new_subtitle.end.milliseconds = int((new_end_time % 1) * 1000)
            
            optimized_subtitles.append(new_subtitle)
        
        # Save optimized subtitles
        optimized_subtitles.save(output_path, encoding='utf-8')
        
        return output_path
   
    @handle_errors()
    def _verify_synchronization(self, input_path: str, options: Dict[str, Any]) -> str:
        """Verify subtitles synchronization with audio/video.
        
        Args:
            input_path: Path to the subtitle file
            options: Additional options for verification
            
        Returns:
            JSON string with verification results
        """
        # Get options
        reference_path = options.get("reference_path", None)
        output_path = options.get("output_path", None)
        
        # Load subtitles
        subtitles = pysrt.open(input_path)
        
        # Check basic timing issues
        issues = []
        subtitle_count = len(subtitles)
        
        for i, subtitle in enumerate(subtitles):
            # Check for very short durations
            duration = (subtitle.end.ordinal - subtitle.start.ordinal) / 1000.0
            if duration < self.MIN_SUBTITLE_DURATION:
                issues.append(f"Subtitle #{i+1} has a very short duration: {duration:.2f}s")
            
            # Check for overlapping with next subtitle
            if i < subtitle_count - 1:
                next_subtitle = subtitles[i+1]
                if subtitle.end.ordinal > next_subtitle.start.ordinal:
                    issues.append(f"Subtitle #{i+1} overlaps with subtitle #{i+2}")
            
            # Check for very long gaps
            if i < subtitle_count - 1:
                next_subtitle = subtitles[i+1]
                gap = (next_subtitle.start.ordinal - subtitle.end.ordinal) / 1000.0
                if gap > 5.0:  # More than 5 seconds
                    issues.append(f"Large gap of {gap:.2f}s between subtitles #{i+1} and #{i+2}")
        
        # Check with reference if provided
        result = {
            "subtitle_count": subtitle_count,
            "issues": issues,
            "synchronization": "good" if not issues else "issues_detected"
        }
        
        if reference_path and os.path.exists(reference_path):
            # Compare with reference timestamps if available
            result["reference_comparison"] = self._compare_with_reference(subtitles, reference_path)
        
        # Save results to output file if needed
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return output_path
        
        # Return result as JSON string
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @handle_errors()
    def _convert_format(self, input_path: str, options: Dict[str, Any]) -> str:
        """Convert subtitle file format.
        
        Args:
            input_path: Path to the subtitle file
            options: Additional options for conversion
            
        Returns:
            Path to the converted file
        """
        # Get options
        target_format = options.get("target_format", "srt").lower()
        output_path = options.get("output_path", None)
        
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(input_path)
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(dirname, f"{basename}.{target_format}")
        
        # Supported formats for now: srt, vtt
        supported_formats = {"srt", "vtt"}
        if target_format not in supported_formats:
            raise ValueError(f"Unsupported target format: {target_format}. Supported formats: {', '.join(supported_formats)}")
        
        # Load subtitles
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext == ".srt":
            subtitles = pysrt.open(input_path)
            
            if target_format == "vtt":
                # Convert to VTT
                vtt_content = "WEBVTT\n\n"
                
                for subtitle in subtitles:
                    # Convert time format
                    start_time = f"{subtitle.start.hours:02d}:{subtitle.start.minutes:02d}:{subtitle.start.seconds:02d}.{subtitle.start.milliseconds:03d}"
                    end_time = f"{subtitle.end.hours:02d}:{subtitle.end.minutes:02d}:{subtitle.end.seconds:02d}.{subtitle.end.milliseconds:03d}"
                    
                    # Add to VTT content
                    vtt_content += f"{start_time} --> {end_time}\n"
                    vtt_content += f"{subtitle.text}\n\n"
                
                # Save VTT file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(vtt_content)
        
        elif file_ext == ".vtt":
            # Basic VTT to SRT conversion
            if target_format == "srt":
                subtitles = pysrt.SubRipFile()
                
                with open(input_path, "r", encoding="utf-8") as f:
                    vtt_content = f.read()
                
                # Parse VTT content
                # Skip WEBVTT header
                content_parts = vtt_content.split("\n\n")
                if content_parts[0].strip().startswith("WEBVTT"):
                    content_parts = content_parts[1:]
                
                for i, part in enumerate(content_parts):
                    lines = part.strip().split("\n")
                    if not lines:
                        continue
                    
                    # Skip comments and other VTT metadata
                    if lines[0].startswith("NOTE") or not "-->" in lines[0]:
                        continue
                    
                    # Parse timestamp line (first line with "-->")
                    timestamp_line = next((line for line in lines if "-->" in line), None)
                    if timestamp_line:
                        timestamps = timestamp_line.split("-->")
                        start_str = timestamps[0].strip().replace(".", ",")
                        end_str = timestamps[1].strip().replace(".", ",")
                        
                        # Get text lines (everything after the timestamp line)
                        text_lines = lines[lines.index(timestamp_line) + 1:]
                        text = "\n".join(text_lines)
                        
                        # Create subtitle item
                        item = pysrt.SubRipItem()
                        item.index = i + 1
                        
                        # Parse start and end times
                        try:
                            # Handle hours, minutes, seconds, milliseconds
                            start_parts = start_str.split(":")
                            if len(start_parts) == 3:  # HH:MM:SS,mmm
                                item.start.hours = int(start_parts[0])
                                item.start.minutes = int(start_parts[1])
                                sec_parts = start_parts[2].split(",")
                                item.start.seconds = int(sec_parts[0])
                                item.start.milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                            elif len(start_parts) == 2:  # MM:SS,mmm
                                item.start.hours = 0
                                item.start.minutes = int(start_parts[0])
                                sec_parts = start_parts[1].split(",")
                                item.start.seconds = int(sec_parts[0])
                                item.start.milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                            
                            # Similar for end time
                            end_parts = end_str.split(":")
                            if len(end_parts) == 3:  # HH:MM:SS,mmm
                                item.end.hours = int(end_parts[0])
                                item.end.minutes = int(end_parts[1])
                                sec_parts = end_parts[2].split(",")
                                item.end.seconds = int(sec_parts[0])
                                item.end.milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                            elif len(end_parts) == 2:  # MM:SS,mmm
                                item.end.hours = 0
                                item.end.minutes = int(end_parts[0])
                                sec_parts = end_parts[1].split(",")
                                item.end.seconds = int(sec_parts[0])
                                item.end.milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                            
                            # Set text
                            item.text = text
                            
                            # Add to subtitle file
                            subtitles.append(item)
                        except Exception as e:
                            print(f"Error parsing subtitle timing: {e}")
                
                # Save SRT file
                subtitles.save(output_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported input format: {file_ext}")
        
        return output_path
    
    @handle_errors()
    def _compare_with_reference(self, subtitles, reference_path: str) -> Dict[str, Any]:
        """Compare subtitles with a reference transcript or subtitle file.
        
        Args:
            subtitles: pysrt.SubRipFile object
            reference_path: Path to reference file
            
        Returns:
            Dictionary with comparison results
        """
        file_ext = os.path.splitext(reference_path)[1].lower()
        
        # Extract reference timing information
        reference_segments = []
        
        if file_ext == '.json':
            # Assume it's a transcript JSON
            with open(reference_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if "segments" in data:
                    for segment in data["segments"]:
                        if isinstance(segment, dict) and "start" in segment and "end" in segment:
                            reference_segments.append({
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment.get("text", "")
                            })
        elif file_ext == '.srt':
            # Another subtitle file
            ref_subtitles = pysrt.open(reference_path)
            for sub in ref_subtitles:
                reference_segments.append({
                    "start": sub.start.ordinal / 1000.0,
                    "end": sub.end.ordinal / 1000.0,
                    "text": sub.text
                })
        else:
            return {"error": f"Unsupported reference format: {file_ext}"}
        
        if not reference_segments:
            return {"error": "No valid segments found in reference file"}
        
        # Compare timing
        sync_issues = []
        timing_scores = []
        
        for i, subtitle in enumerate(subtitles):
            sub_start = subtitle.start.ordinal / 1000.0
            sub_end = subtitle.end.ordinal / 1000.0
            
            # Find best matching reference segment
            best_match = None
            best_match_score = 0
            
            for ref_segment in reference_segments:
                # Calculate overlap
                overlap_start = max(sub_start, ref_segment["start"])
                overlap_end = min(sub_end, ref_segment["end"])
                
                if overlap_end > overlap_start:
                    # There's some overlap
                    overlap_duration = overlap_end - overlap_start
                    sub_duration = sub_end - sub_start
                    ref_duration = ref_segment["end"] - ref_segment["start"]
                    
                    # Calculate overlap ratio relative to both durations
                    sub_ratio = overlap_duration / sub_duration
                    ref_ratio = overlap_duration / ref_duration
                    
                    # Combined score
                    match_score = (sub_ratio + ref_ratio) / 2
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match = ref_segment
            
            # Record timing issue if significant mismatch
            if best_match and best_match_score < 0.7:
                sync_issues.append({
                    "subtitle_index": i+1,
                    "subtitle_time": f"{sub_start:.2f}-{sub_end:.2f}",
                    "reference_time": f"{best_match['start']:.2f}-{best_match['end']:.2f}",
                    "match_score": best_match_score,
                    "text": subtitle.text
                })
            
            timing_scores.append(best_match_score if best_match else 0)
        
        # Calculate overall sync score
        avg_sync_score = sum(timing_scores) / len(timing_scores) if timing_scores else 0
        
        return {
            "sync_score": avg_sync_score,
            "sync_issues": sync_issues[:10],  # Limit to first 10 issues
            "sync_issue_count": len(sync_issues),
            "reference_segment_count": len(reference_segments),
            "subtitle_count": len(subtitles)
        }