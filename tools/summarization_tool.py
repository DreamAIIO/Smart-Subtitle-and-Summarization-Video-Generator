"""
Enhanced summarization tool for creating high-quality summaries using both extractive and abstractive methods.
Specifically improved to handle JSON transcripts properly and generate better summaries.
Fixed the error with LexRankSummarizer's rate_sentence method.
"""
import os
import json
import re
from typing import Dict, Optional, Any, List, Tuple

import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass  # Handle silently if download fails

try:
    nltk.data.find('stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass  # Handle silently if download fails


class EnhancedSummarizationTool(Tool):
    """Tool for creating high-quality summaries using both extractive and abstractive methods."""
    
    name = "enhanced_summarizer"
    description = """
    Creates comprehensive summaries of video content using both extractive and
    abstractive summarization methods for improved quality.
    """
    inputs = {
        "transcript_path": {
            "type": "string",
            "description": "Path to the transcript file"
        },
        "language": {
            "type": "string",
            "description": "Language code for the transcript",
            "nullable": True
        },
        "options": {
            "type": "object",
            "description": "Additional summarization options",
            "nullable": True
        }
    }
    output_type = "string"
    
    @log_execution
    @handle_errors(default_return="Error summarizing content")
    def forward(self, transcript_path: str, language: str = "en", options: Optional[Dict[str, Any]] = None) -> str:
        """Summarize video content based on transcript with improved fallback mechanisms.
        
        Args:
            transcript_path: Path to the transcript file
            language: Language code for the transcript
            options: Additional summarization options
            
        Returns:
            Path to the summary file or the summary text
        """
        options = options or {}
        verbose = options.get("verbose", False)
        if verbose:
            print(f"Starting summarization for transcript: {transcript_path}")
            print(f"Options: {options}")
        
        # Ensure output directory exists
        output_path = options.get("output_path", None)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Make sure nltk data is available
        self._ensure_nltk_data()
        
        try:
            # Check if transcript file exists
            if not os.path.exists(transcript_path):
                if verbose:
                    print(f"Transcript file not found: {transcript_path}")
                # Create a basic summary if transcript not found
                return self._create_fallback_summary(
                    "Transcript file not found for summarization.",
                    output_path, 
                    options
                )
            
            # Extract text content from transcript
            transcript_text, transcript_segments = self._extract_transcript_content(transcript_path)
            
            if verbose:
                print(f"Extracted transcript text length: {len(transcript_text)}")
                print(f"Number of transcript segments: {len(transcript_segments)}")
            
            # If the transcript appears to be a placeholder or error message, create a minimal summary
            if not transcript_text.strip() or "Error during transcription" in transcript_text:
                if verbose:
                    print("Transcript appears to be empty or contains an error message")
                # Extract any usable content from the error message
                potential_content = self._extract_content_from_error_transcript(transcript_text)
                return self._create_fallback_summary(
                    potential_content or "The transcript contains an error message or is empty.",
                    output_path,
                    options
                )
            
            # Determine summarization parameters
            method = options.get("method", "hybrid")
            sentences_count = options.get("sentences_count", 5)
            output_format = options.get("output_format", "json")
            include_timeline = options.get("include_timeline", True)
            
            if verbose:
                print(f"Using summarization method: {method}")
                print(f"Sentences count: {sentences_count}")
            
            # Phase 1: Analyze transcript to identify key segments
            key_segments = self._identify_key_segments(transcript_text, transcript_segments, language)
            
            if verbose:
                print(f"Identified {len(key_segments)} key segments")
            
            # Phase 2: Generate an extractive summary
            extractive_summary = self._generate_extractive_summary(
                transcript_text, 
                language, 
                method=options.get("extractive_method", "lexrank"), 
                sentences_count=sentences_count
            )
            
            if verbose:
                print(f"Generated extractive summary with {len(extractive_summary)} sentences")
            
            # Phase 3: Generate an abstractive summary (if available)
            abstractive_summary = self._generate_abstractive_summary(
                transcript_text,
                extractive_summary,
                language,
                options
            )
            
            if verbose:
                if abstractive_summary:
                    print(f"Generated abstractive summary with {len(abstractive_summary)} characters")
                else:
                    print("No abstractive summary generated")
            
            # Phase 4: Generate timeline-based summary if segments are available
            timeline_summary = None
            if include_timeline and transcript_segments:
                timeline_summary = self._generate_timeline_summary(
                    transcript_segments,
                    language,
                    options.get("timeline_points", 5)
                )
                
                if verbose:
                    if timeline_summary:
                        print(f"Generated timeline summary with {len(timeline_summary)} points")
                    else:
                        print("No timeline summary generated")
            
            # Combine summaries for the final output
            final_summary = self._create_final_summary(
                extractive_summary,
                abstractive_summary,
                timeline_summary,
                key_segments
            )
            
            if verbose:
                print("Created final summary")
                if "text" in final_summary:
                    print(f"Summary text length: {len(final_summary['text'])}")
            
            # Determine output handling
            if not output_path:
                # Generate output path
                dirname = os.path.dirname(transcript_path)
                basename = os.path.splitext(os.path.basename(transcript_path))[0]
                output_path = os.path.join(dirname, f"{basename}_summary.{output_format}")
                
                if verbose:
                    print(f"Generated output path: {output_path}")
            
            # Make sure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save summary
            if output_format == "txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_summary["text"])
                    
                if verbose:
                    print(f"Saved summary as text file: {output_path}")
            elif output_format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_summary, f, ensure_ascii=False, indent=2)
                    
                if verbose:
                    print(f"Saved summary as JSON file: {output_path}")
            else:
                return f"Unsupported output format: {output_format}"
            
            return output_path
        except Exception as e:
            if verbose:
                print(f"Error during summarization: {str(e)}")
            # Create a minimal summary even if everything fails
            return self._create_fallback_summary(
                f"Error generating summary: {str(e)}",
                output_path,
                options
            )
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available, downloading or creating fallbacks if necessary."""
        # Attempt to quietly download required data if not found
        for data_name in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else data_name)
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except:
                    # If download fails, create minimal fallbacks
                    if data_name == 'punkt':
                        # Create a simple tokenizer that splits on periods
                        nltk.tokenize._treebank_word_tokenizer = lambda text: text.split('.')
                    elif data_name == 'stopwords':
                        # Create a minimal set of English stopwords
                        from nltk.corpus import stopwords
                        if not hasattr(stopwords, 'words') or not stopwords.words('english'):
                            simple_stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 
                                               'when', 'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 
                                               'on', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                                               'have', 'has', 'had', 'do', 'does', 'did', 'I', 'you', 'he', 
                                               'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those']
                            stopwords.words = lambda lang: simple_stopwords if lang == 'english' else []
    
    def _create_fallback_summary(self, message: str, output_path: Optional[str], options: Dict[str, Any]) -> str:
        """Create a fallback summary when normal summarization fails."""
        output_format = options.get("output_format", "json")
        
        if not output_path:
            # Generate a default output path
            import tempfile
            output_path = os.path.join(tempfile.gettempdir(), f"fallback_summary.{output_format}")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a minimal summary containing the error message and some generic content
        if output_format == "json":
            fallback_content = {
                "text": "Summary could not be generated properly due to issues with the transcript. " + message,
                "extractive_summary": ["Summary could not be generated properly due to issues with the transcript."],
                "abstractive_summary": "Summary could not be generated properly due to issues with the transcript. " + message
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fallback_content, f, ensure_ascii=False, indent=2)
        else:
            # Text format
            fallback_content = f"Summary could not be generated properly due to issues with the transcript.\n\n{message}\n\nPlease ensure the transcription process completes successfully to generate a proper summary."
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(fallback_content)
        
        return output_path

    def _extract_content_from_error_transcript(self, error_transcript: str) -> Optional[str]:
        """Try to extract any usable content from an error transcript."""
        # Look for any content after the error message
        import re
        
        # Try to find any meaningful text after error messages
        matches = re.search(r"Error.*\n\n(.+)", error_transcript)
        if matches and matches.group(1).strip():
            return matches.group(1).strip()
        
        return None
    
    @handle_errors()
    def _extract_transcript_content(self, transcript_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text and segments from a transcript file.
        
        Args:
            transcript_path: Path to the transcript file
            
        Returns:
            Tuple of (transcript text, transcript segments)
        """
        file_ext = os.path.splitext(transcript_path)[1].lower()
        transcript_text = ""
        transcript_segments = []
        
        if file_ext == '.json':
            with open(transcript_path, 'r', encoding='utf-8') as f:
                try:
                    transcript_data = json.load(f)
                    
                    # Extract text if available directly
                    if "text" in transcript_data:
                        transcript_text = transcript_data["text"]
                    
                    # Extract segments if available
                    if "segments" in transcript_data:
                        segments = transcript_data["segments"]
                        transcript_segments = segments
                        
                        # If no direct text, build it from segments
                        if not transcript_text:
                            segment_texts = []
                            for segment in segments:
                                if isinstance(segment, dict) and "text" in segment:
                                    segment_texts.append(segment["text"])
                            transcript_text = " ".join(segment_texts)
                except json.JSONDecodeError:
                    # Not a valid JSON, read as plain text
                    f.seek(0)
                    transcript_text = f.read()
        elif file_ext == '.srt':
            import pysrt
            try:
                subtitles = pysrt.open(transcript_path)
                texts = []
                
                for subtitle in subtitles:
                    text = subtitle.text
                    transcript_segments.append({
                        "start": subtitle.start.hours * 3600 + subtitle.start.minutes * 60 + subtitle.start.seconds + subtitle.start.milliseconds / 1000,
                        "end": subtitle.end.hours * 3600 + subtitle.end.minutes * 60 + subtitle.end.seconds + subtitle.end.milliseconds / 1000,
                        "text": text
                    })
                    texts.append(text)
                
                transcript_text = " ".join(texts)
            except Exception as e:
                # If pysrt fails, try reading as text
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
        elif file_ext == '.txt':
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        else:
            raise ValueError(f"Unsupported transcript format: {file_ext}")
        
        return transcript_text, transcript_segments
    
    @handle_errors()
    def _identify_key_segments(self, transcript_text: str, transcript_segments: List[Dict[str, Any]], 
                             language: str) -> List[Dict[str, Any]]:
        """Identify key segments in the transcript.
        
        Args:
            transcript_text: Full transcript text
            transcript_segments: List of transcript segments
            language: Language code
            
        Returns:
            List of key segments with importance scores
        """
        # If no segments are available, return empty list
        if not transcript_segments:
            return []
        
        try:
            # Use NLTK to tokenize sentences
            tokenizer = Tokenizer(language)
            stemmer = Stemmer(language)
            parser = PlaintextParser.from_string(transcript_text, tokenizer)
            
            # Use LexRank to get sentence importances
            summarizer = LexRankSummarizer(stemmer)
            summarizer.stop_words = get_stop_words(language)
            
            # Get sentences from the document
            sentences = parser.document.sentences
            
            # Get key segments based on sentence importance
            key_segments = []
            
            # Added new implementation to replace the missing rate_sentence method
            # This computes sentence importances differently
            sentence_scores = {}
            
            # Step 1: Convert sentences to plain strings for comparison
            sentence_texts = [str(sentence) for sentence in sentences]
            
            # Step 2: Get top sentences using the summarizer directly
            # Instead of using rate_sentence, we'll use the summarizer to get top sentences
            important_sentences = summarizer(parser.document, len(sentence_texts))
            important_sentence_texts = [str(sent) for sent in important_sentences]
            
            # Step 3: Assign importance scores based on rank in important_sentences
            for i, sent_text in enumerate(important_sentence_texts):
                # Higher score for earlier sentences in the important list
                score = 1.0 - (i / len(important_sentence_texts)) if important_sentence_texts else 0.5
                
                # Find all occurrences of this sentence in our segments
                for j, segment in enumerate(transcript_segments):
                    if isinstance(segment, dict) and "text" in segment:
                        segment_text = segment["text"]
                        # Check if this important sentence is in this segment
                        if sent_text in segment_text or segment_text in sent_text:
                            sentence_scores[j] = {
                                "segment": segment,
                                "score": score
                            }
            
            # If we don't have scores yet (summarizer might have failed), use position-based scoring
            if not sentence_scores:
                # Fallback: Simple scoring based on position
                # Beginning and end of transcript often have important content
                total_segments = len(transcript_segments)
                for i, segment in enumerate(transcript_segments):
                    if isinstance(segment, dict) and "text" in segment:
                        # Higher scores for segments at the beginning or end
                        position_score = 1.0
                        if i < total_segments // 4:  # First quarter
                            position_score = 0.9 - (i / (total_segments // 4)) * 0.4
                        elif i > 3 * total_segments // 4:  # Last quarter
                            position_score = 0.5 + ((i - (3 * total_segments // 4)) / (total_segments // 4)) * 0.4
                        else:  # Middle
                            position_score = 0.5
                        
                        # Also factor in segment length - longer segments might be more important
                        text_length = len(segment.get("text", ""))
                        length_score = min(0.2, text_length / 500)  # Cap at 0.2
                        
                        # Combined score
                        sentence_scores[i] = {
                            "segment": segment,
                            "score": position_score + length_score
                        }
            
            # Sort segments by score
            key_segments = []
            for idx, item in sorted(sentence_scores.items(), key=lambda x: x[1]["score"], reverse=True):
                key_segments.append({
                    "start": item["segment"].get("start", 0),
                    "end": item["segment"].get("end", 0),
                    "text": item["segment"].get("text", ""),
                    "importance_score": float(item["score"])
                })
            
            # Return top segments (up to 1/3 of all segments)
            max_key_segments = max(3, len(transcript_segments) // 3)
            return key_segments[:max_key_segments]
        except Exception as e:
            print(f"Error identifying key segments: {str(e)}")
            return []
    
    @handle_errors()
    def _generate_extractive_summary(self, text: str, language: str, method: str = "lexrank", 
                                   sentences_count: int = 5) -> List[str]:
        """Generate an extractive summary of the text with proper error handling.
        
        Args:
            text: Text to summarize
            language: Language code
            method: Summarization method (lexrank, lsa, luhn)
            sentences_count: Number of sentences in the summary
            
        Returns:
            List of summary sentences
        """
        if not text.strip():
            return ["No text content available for summarization."]
            
        # Ensure minimum viable text size
        if len(text.split()) < 10:
            return [text.strip()]
            
        try:
            # Create parser
            parser = PlaintextParser.from_string(text, Tokenizer(language))
            
            # Create stemmer
            stemmer = Stemmer(language)
            
            # Choose summarizer based on method
            if method == "lexrank":
                summarizer = LexRankSummarizer(stemmer)
            elif method == "lsa":
                summarizer = LsaSummarizer(stemmer)
            elif method == "luhn":
                summarizer = LuhnSummarizer(stemmer)
            else:
                # Default to LexRank if method is unsupported
                summarizer = LexRankSummarizer(stemmer)
            
            # Add stop words for the language
            summarizer.stop_words = get_stop_words(language)
            
            # Ensure we don't ask for more sentences than exist
            num_sentences = len(parser.document.sentences)
            if num_sentences < sentences_count:
                sentences_count = max(1, num_sentences)
            
            # Generate summary
            summary = summarizer(parser.document, sentences_count)
            
            # Convert to list of strings
            return [str(sentence) for sentence in summary]
        except Exception as e:
            print(f"Error in extractive summarization: {str(e)}")
            # Fallback to a simple extractive approach
            sentences = text.split('.')
            if len(sentences) <= sentences_count:
                return [s.strip() + '.' for s in sentences if s.strip()]
            else:
                # Take sentences from the beginning, middle and end for a more representative summary
                beginning = sentences[:sentences_count//3]
                middle_start = len(sentences)//2 - sentences_count//6
                middle = sentences[middle_start:middle_start + sentences_count//3]
                end = sentences[-(sentences_count - len(beginning) - len(middle)):]
                return [s.strip() + '.' for s in beginning + middle + end if s.strip()]
    
    @handle_errors()
    def _generate_abstractive_summary(self, transcript_text: str, extractive_summary: List[str],
                                    language: str, options: Dict[str, Any]) -> Optional[str]:
        """Generate an abstractive summary using Gemini for better quality.
        
        Args:
            transcript_text: Full transcript text
            extractive_summary: Extractive summary sentences
            language: Language code
            options: Additional options
            
        Returns:
            Abstractive summary text or None if not available
        """
        if not extractive_summary:
            return None
            
        try:
            # Initialize Gemini model
            from gemini_model import GeminiModel
            model = GeminiModel(
                model_id="gemini-2.5-pro-preview-05-06",
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            # Create a prompt for abstractive summarization
            prompt = f"""Create a comprehensive, thoughtful summary of the following content. 
            The summary should:
            1. Explain the main topic and purpose in your own words
            2. Highlight the core message and significance
            3. Synthesize the key points into a coherent narrative
            4. NOT merely repeat phrases from the original text
            5. Be clear, concise, and easy to understand
            
            Here is the content to summarize:
            
            {transcript_text}
            
            Key points identified:
            {chr(10).join(f"- {point}" for point in extractive_summary)}
            
            Please provide a well-structured summary that captures the essence of the content."""
            
            # Generate the summary using Gemini
            response = model([{"role": "user", "content": prompt}])
            
            if response and response.content:
                summary = response.content.strip()
                
                # Basic cleanup
                summary = re.sub(r'\s+', ' ', summary).strip()
                
                # Ensure proper sentence endings
                if not summary.endswith(('.', '!', '?')):
                    summary += '.'
                
                return summary
                
            return None
            
        except Exception as e:
            print(f"Error in abstractive summarization: {str(e)}")
            # Fallback to basic cleanup of extractive summary
            if extractive_summary:
                combined_text = " ".join(extractive_summary)
                cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
                return cleaned_text
            return None
    
    @handle_errors()
    def _generate_timeline_summary(self, segments: List[Dict[str, Any]], language: str, 
                                 num_points: int = 5) -> List[Dict[str, Any]]:
        """Generate a timeline-based summary with key points at intervals.
        
        Args:
            segments: List of transcript segments
            language: Language code
            num_points: Number of timeline points to generate
            
        Returns:
            List of timeline points with time and summary
        """
        if not segments:
            return []
        
        # Determine video duration and interval size
        start_time = segments[0].get("start", 0) if isinstance(segments[0], dict) else 0
        end_time = segments[-1].get("end", 0) if isinstance(segments[-1], dict) else 0
        duration = end_time - start_time
        
        # Ensure minimum duration
        if duration < 10:
            return []
        
        # Divide into equal intervals
        interval_size = duration / num_points
        
        # Create summaries for each interval
        timeline_points = []
        
        for i in range(num_points):
            interval_start = start_time + (i * interval_size)
            interval_end = interval_start + interval_size
            
            # Collect segments in this interval
            interval_segments = [
                segment for segment in segments
                if isinstance(segment, dict) and 
                segment.get("start", 0) >= interval_start and 
                segment.get("end", 0) <= interval_end
            ]
            
            if not interval_segments:
                continue
            
            # Extract text from interval segments
            interval_text = " ".join([segment.get("text", "") for segment in interval_segments])
            
            if not interval_text.strip():
                continue
            
            # Generate summary for this interval
            try:
                # Create parser and summarizer
                parser = PlaintextParser.from_string(interval_text, Tokenizer(language))
                summarizer = LexRankSummarizer(Stemmer(language))
                summarizer.stop_words = get_stop_words(language)
                
                # Get a single summary sentence for this interval
                try:
                    summary_sentences = list(summarizer(parser.document, 1))
                    if summary_sentences:
                        interval_summary = str(summary_sentences[0])
                    else:
                        # Fallback if summarizer fails
                        words = interval_text.split()
                        if len(words) > 15:
                            interval_summary = " ".join(words[:15]) + "..."
                        else:
                            interval_summary = interval_text
                except:
                    # Another fallback
                    interval_summary = interval_text[:100] + "..." if len(interval_text) > 100 else interval_text
            except Exception:
                # If summarization fails, use the first part of the text
                interval_summary = interval_text[:100] + "..." if len(interval_text) > 100 else interval_text
            
            # Format timestamp for display (MM:SS)
            mins_start = int(interval_start / 60)
            secs_start = int(interval_start % 60)
            timestamp = f"{mins_start:02d}:{secs_start:02d}"
            
            timeline_points.append({
                "time": interval_start,
                "timestamp": timestamp,
                "summary": interval_summary
            })
        
        return timeline_points
    
    @handle_errors()
    def _create_final_summary(self, extractive_summary: List[str], 
                            abstractive_summary: Optional[str],
                            timeline_summary: Optional[List[Dict[str, Any]]],
                            key_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create the final combined summary output.
        
        Args:
            extractive_summary: List of extractive summary sentences
            abstractive_summary: Abstractive summary text
            timeline_summary: List of timeline points
            key_segments: List of key segments
            
        Returns:
            Final summary data
        """
        # Ensure we have at least some summary content
        if not extractive_summary:
            extractive_summary = ["No summary available."]
        
        # Combine the summaries into a cohesive output
        result = {
            "extractive_summary": extractive_summary,
            "key_segments": key_segments
        }
        
        # Add abstractive summary if available
        if abstractive_summary:
            result["abstractive_summary"] = abstractive_summary
            # Use abstractive as the main summary text
            result["text"] = abstractive_summary
        else:
            # Use extractive as the main summary text
            result["text"] = " ".join(extractive_summary)
        
        # Add timeline if available
        if timeline_summary:
            result["timeline"] = timeline_summary
        
        return result