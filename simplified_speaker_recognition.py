"""
A simplified speaker recognition agent that combines technical diarization with Gemini-enhanced analysis.
"""
import os
import json
import re
import logging
from typing import Dict, Any, Optional, List

from smolagents import CodeAgent
from smolagents.models import Model

# Configure logging
logger = logging.getLogger("subtitle_generator")

def extract_code_block(llm_output: str) -> str:
    """
    Extracts a Python code block from LLM output, supporting several fallback patterns.
    Returns the code as a string, or raises ValueError if not found.
    """
    import re
    match = re.search(r'Code:\s*```(?:py|python)?\n(.*?)```', llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```(?:py|python)?\n(.*?)```', llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```[^\n]*\n(.*?)```', llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError("No code block found in LLM output")

def create_simple_speaker_recognition_agent(model: Model, speaker_tool, file_tool, quality_tool):
    """Create a simplified agent for speaker recognition.
    
    Args:
        model: The LLM model to use
        speaker_tool: The speaker recognition tool to use
        file_tool: The file operations tool
        quality_tool: The quality verification tool
        
    Returns:
        CodeAgent for speaker recognition
    """
    # Common imports for the agent
    COMMON_IMPORTS = ["json", "os", "re", "tempfile", "shutil", "uuid", "open", "subprocess", "globals"]
    
    # Create the agent with a simplified prompt
    agent = CodeAgent(
        tools=[speaker_tool, file_tool, quality_tool],
        model=model,
        name="speaker_recognition_agent",
        description="Identifies and characterizes different speakers in audio content.",
        max_steps=3,
        planning_interval=None,  # Disable planning to reduce LLM calls
        additional_authorized_imports=COMMON_IMPORTS
    )
    
    return agent

def run_speaker_recognition(agent: CodeAgent, audio_path: str, transcript_path: Optional[str] = None, 
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the speaker recognition agent with a simplified approach.
    
    Args:
        agent: The CodeAgent to use
        audio_path: Path to the audio file
        transcript_path: Optional path to the transcript file
        options: Additional options for speaker recognition
        
    Returns:
        Dictionary with speaker recognition results
    """
    options = options or {}
    verbose = options.get("verbose", False)
    
    # Create a detailed task for the agent
    task = f"""
    Identify and characterize the different speakers in the audio file at {audio_path}.
    
    IMPORTANT INSTRUCTIONS:
    
    1. First, use the speaker_recognizer tool to get technical speaker diarization results
    2. Then analyze these results along with any transcript data to provide enhanced speaker information
    3. For each identified speaker, try to determine:
       - Speaker role or identity (if mentioned in transcript)
       - Speaker characteristics (gender, approximate age if determinable)
       - Speaker's importance in the content (main presenter, guest, etc.)
    4. Save the enhanced speaker information in both JSON and text formats
    
    TECHNICAL INSTRUCTIONS:
    - Use the speaker_recognizer tool to perform initial diarization
    - If the initial diarization fails, try adjusting parameters like min_speakers or max_speakers
    - Always save your final results using file_operator tool to write JSON output
    - Use quality_verifier tool to check the quality of your results
    """
    
    # Add transcript information if available
    if transcript_path and os.path.exists(transcript_path):
        task += f"""
        
        Use the transcript file at {transcript_path} to help identify speaker names and roles.
        Look for patterns like "Speaker: Text" or mentions of names followed by statements.
        
        TRANSCRIPT ANALYSIS TECHNIQUES:
        1. Look for direct speaker identifications like "John: Hello everyone"
        2. Look for self-introductions like "Hello, my name is Sarah"
        3. Look for third-party introductions like "Let me introduce Dr. Thompson"
        4. Match speaker segments with these identified names
        5. For unidentified speakers, assign clear roles based on context
        """
        
        # Add instruction to attempt loading the transcript file
        task += f"""
        
        STEPS TO FOLLOW:
        1. Read the transcript file using the file_operator tool
        2. Perform speaker diarization using the speaker_recognizer tool
        3. Analyze the transcript text to identify speaker names
        4. Map speaker IDs from diarization to names from transcript
        5. Save the enhanced speaker information
        
        # Example code to read the transcript:
        transcript_content = file_operator(operation="read", file_path="{transcript_path}")
        """
    
    # Add other options to the task if provided
    if "min_speakers" in options and "max_speakers" in options:
        min_speakers = options["min_speakers"]
        max_speakers = options["max_speakers"]
        task += f"""
        
        The audio contains between {min_speakers} and {max_speakers} speakers.
        
        # Example code to use these parameters:
        diarization_result = speaker_recognizer(
            audio_path="{audio_path}",
            options={{"min_speakers": {min_speakers}, "max_speakers": {max_speakers}}}
        )
        """
    elif "expected_speakers" in options:
        expected_speakers = options["expected_speakers"]
        task += f"""
        
        The audio is expected to have approximately {expected_speakers} speakers.
        
        # Example code to use this parameter:
        diarization_result = speaker_recognizer(
            audio_path="{audio_path}",
            options={{"expected_speakers": {expected_speakers}}}
        )
        """
    else:
        # Add default example code
        task += f"""
        
        # Example code to perform speaker diarization:
        diarization_result = speaker_recognizer(audio_path="{audio_path}")
        """
        
    # Add pyannote result if available
    pyannote_result = options.get("pyannote_result")
    if pyannote_result and isinstance(pyannote_result, dict) and "speakers" in pyannote_result:
        task += f"""
        
        I already have some technical speaker diarization results that you can enhance:
        
        ```json
        {json.dumps(pyannote_result, indent=2)}
        ```
        
        Please use this information as a starting point, but enhance it with:
        1. Actual speaker identities if you can determine them from the transcript
        2. More detailed speaker characteristics
        3. Analysis of speaker relationships and speaking patterns
        
        The technical data provides precise timing information that you should preserve,
        while adding valuable semantic insights about each speaker's role and identity.
        
        # Example code to read and enhance the pyannote results:
        pyannote_speakers = {json.dumps(pyannote_result.get("speakers", {}), indent=2)}
        enhanced_speakers = {{}}
        
        for speaker_id, speaker_info in pyannote_speakers.items():
            # Copy technical data
            enhanced_speakers[speaker_id] = speaker_info.copy()
            
            # Add enhanced information
            enhanced_speakers[speaker_id]["role"] = "Determine role/identity based on transcript"
            enhanced_speakers[speaker_id]["characteristics"] = "Add detailed speaker characteristics"
        
        # Save the enhanced speaker information
        output_path = os.path.join(os.path.dirname("{audio_path}"), "enhanced_speakers.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({{"speakers": enhanced_speakers}}, f, indent=2)
        """
    
    # Add specific instructions about output structure
    task += """
    
    ANALYSIS GUIDELINES:
    1. Be precise about speaker transitions
    2. If you can determine speaker identities, use them instead of generic labels
    3. Note any characteristic speech patterns or vocal qualities
    4. Provide confidence levels for your identifications
    
    WRITING GUIDELINES:
    1. Use clear, concise descriptions
    2. Format timestamps consistently
    3. Organize speakers by their speaking time (most to least)
    4. Include both technical data and semantic insights
    
    EXPECTED OUTPUT FORMAT:
    {
        "speakers": {
            "SPEAKER_1": {
                "role": "Host",
                "characteristics": "Male, formal tone, leads discussion",
                "total_time": 142.5,
                "segments": [
                    {"start": 0.0, "end": 15.5, "text": "..."},
                    {"start": 35.2, "end": 55.8, "text": "..."}
                ]
            },
            "SPEAKER_2": {
                "role": "Guest - Dr. Jane Smith",
                "characteristics": "Female, technical vocabulary, expert voice",
                "total_time": 95.3,
                "segments": [
                    {"start": 16.0, "end": 35.0, "text": "..."},
                    {"start": 56.0, "end": 75.5, "text": "..."}
                ]
            }
        }
    }
    
    Remember, good speaker identification combines technical analysis with contextual understanding.
    """
    
    try:
        logger.info("Running speaker recognition agent with enhanced instructions")
        result = agent.run(task)

        if isinstance(result, str):
            try:
                code = extract_code_block(result)
                # You can now use `code` if you need to execute or inspect it
            except Exception:
                pass  # No code block found, continue as normal

        # Process the result to extract the path to output files
        json_path = None
        txt_path = None
        
        # If result is a dictionary, extract paths directly
        if isinstance(result, dict):
            json_path = result.get("json_output_path") or result.get("speaker_data_path")
            txt_path = result.get("text_output_path") or result.get("speaker_text_path")
            
            # If we have speaker data directly in the result, use that
            speakers_data = result.get("speakers", {})
            if speakers_data:
                return {
                    "speakers": speakers_data,
                    "json_path": json_path,
                    "text_path": txt_path
                }
        
        # If no direct dictionary, try to extract paths from string result
        if not json_path:
            json_match = re.search(r'(/[^\s]+(?:_speakers|_speaker_data|with_speakers)\.json)', str(result))
            if json_match:
                json_path = json_match.group(1)
        
        if not txt_path:
            txt_match = re.search(r'(/[^\s]+(?:_speakers|_speaker_data|with_speakers)\.txt)', str(result))
            if txt_match:
                txt_path = txt_match.group(1)
        
        # Try to infer output paths if still not found
        if not json_path and transcript_path:
            json_path = transcript_path.replace(".json", "_with_speakers.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(os.path.dirname(transcript_path), "speaker_data.json")
        
        if not txt_path and transcript_path:
            txt_path = transcript_path.replace(".json", "_speakers.txt")
            if not os.path.exists(txt_path):
                txt_path = os.path.join(os.path.dirname(transcript_path), "speaker_data.txt")
        
        # Try to read the speaker data from the JSON file
        speakers_data = {}
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "speakers" in data:
                        speakers_data = data["speakers"]
                    elif isinstance(data, dict) and any(k.startswith("SPEAKER_") or k.startswith("Speaker ") for k in data.keys()):
                        speakers_data = data
            except Exception as e:
                logger.error(f"Error reading speaker data from JSON: {str(e)}")
                if verbose:
                    print(f"Error reading speaker data from JSON: {str(e)}")
        
        # Try to extract speaker information from the agent's response if JSON parsing failed
        if not speakers_data:
            # Extract speaker information from text output if available
            if txt_path and os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    
                    # Try to parse speaker information from text
                    speakers_data = parse_speakers_from_text(text_content)
                except Exception as e:
                    logger.error(f"Error reading speaker data from text: {str(e)}")
                    if verbose:
                        print(f"Error reading speaker data from text: {str(e)}")
        
        # If we still don't have speaker data, try to parse it from the agent's response
        if not speakers_data and isinstance(result, str):
            speakers_data = parse_speakers_from_text(result)
        
        # If we have no speaker data at all, create a minimal result
        if not speakers_data:
            speakers_data = {
                "SPEAKER_1": {
                    "role": "Unknown speaker",
                    "characteristics": "No characteristics identified",
                    "total_time": 0,
                    "segments": []
                }
            }
        
        return {
            "speakers": speakers_data,
            "json_path": json_path,
            "text_path": txt_path
        }

    except Exception as e:
        logger.error(f"Error in speaker recognition: {str(e)}")
        return {
            "error": str(e),
            "raw_error": str(e)
        }

def parse_speakers_from_text(text: str) -> Dict[str, Any]:
    """Parse speaker information from text content.
    
    Args:
        text: Text content to parse
        
    Returns:
        Dictionary with speaker information
    """
    speakers = {}
    
    # Pattern to identify speaker sections - more robust version
    speaker_section_pattern = r'(?:Speaker|SPEAKER)[\s_](\w+|"\w+"|[\d]+)(?:\s*[-:]\s*|\s+)(.*?)(?=(?:Speaker|SPEAKER)[\s_](?:\w+|"\w+"|[\d]+)(?:\s*[-:]\s*|\s+)|\Z)'
    speaker_matches = re.finditer(speaker_section_pattern, text, re.DOTALL | re.IGNORECASE)
    
    speaker_count = 0
    for match in speaker_matches:
        speaker_count += 1
        speaker_id = match.group(1).strip('" ')
        description = match.group(2).strip()
        
        # Create speaker entry
        speaker_key = f"SPEAKER_{speaker_id}" if not speaker_id.startswith("SPEAKER_") else speaker_id
        
        # Extract role/identity from description
        role = "Unknown"
        role_patterns = [
            r'(?:Role|Identity|Name)[:\s]+([^\n\.]+)',
            r'(?:identified as|appears to be)[:\s]+([^\n\.]+)',
            r'(?:this is|this speaker is)[:\s]+([^\n\.]+)'
        ]
        
        for pattern in role_patterns:
            role_match = re.search(pattern, description, re.IGNORECASE)
            if role_match:
                role = role_match.group(1).strip()
                break
                
        # If no explicit role found, check if the first sentence looks like a role description
        if role == "Unknown":
            first_sentence_match = re.match(r'^([^\.!\?]+)[\.!\?]', description.strip())
            if first_sentence_match:
                first_sentence = first_sentence_match.group(1).strip()
                # Check if this looks like a role description (not too long, contains role-like words)
                role_words = ["speaker", "host", "guest", "presenter", "interviewer", "interviewee", 
                             "male", "female", "person", "voice", "narrator", "participant"]
                if (len(first_sentence.split()) < 10 and 
                    any(word in first_sentence.lower() for word in role_words)):
                    role = first_sentence
        
        # Extract characteristics from description
        characteristics = ""
        char_patterns = [
            r'(?:Characteristics|Profile|Voice|Description)[:\s]+([^\n\.]+)',
            r'(?:characterized by|sounds like|appears to be)[:\s]+([^\n\.]+)'
        ]
        
        for pattern in char_patterns:
            char_match = re.search(pattern, description, re.IGNORECASE)
            if char_match:
                characteristics = char_match.group(1).strip()
                break
        
        # If no explicit characteristics found, try to extract from second sentence
        if not characteristics:
            second_sentence_match = re.search(r'^[^\.!\?]+[\.!\?]\s*([^\.!\?]+)[\.!\?]', description.strip())
            if second_sentence_match:
                second_sentence = second_sentence_match.group(1).strip()
                # Check if this looks like characteristics description
                char_words = ["voice", "tone", "accent", "speaks", "speaking", "sound", 
                             "male", "female", "young", "old", "character", "quality"]
                if (len(second_sentence.split()) < 15 and 
                    any(word in second_sentence.lower() for word in char_words)):
                    characteristics = second_sentence
        
        # Extract speaking time if available
        total_time = 0
        time_match = re.search(r'(?:total|speaking|talk(?:ing)?)\s+time[:\s]+(\d+(?:\.\d+)?)', 
                              description, re.IGNORECASE)
        if time_match:
            try:
                total_time = float(time_match.group(1))
            except ValueError:
                pass
        
        # Extract segments if available
        segments = []
        
        # Try different segment patterns
        segment_patterns = [
            # Standard format: 10.5 - 20.5: Text
            r'(\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(\d+(?:\.\d+)?)\s*[:\s]+([^\n]+)',
            # Alternative timestamp format: [00:10.5 - 00:20.5] Text
            r'\[(\d+:\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(\d+:\d+(?:\.\d+)?)\]\s*([^\n]+)',
            # Just timestamp and text: 10.5s: Text
            r'(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\s*[:\s]+([^\n]+)'
        ]
        
        for pattern in segment_patterns:
            segment_matches = re.finditer(pattern, description, re.MULTILINE)
            
            for seg_match in segment_matches:
                if len(seg_match.groups()) == 3:  # Start-End-Text format
                    start_str = seg_match.group(1)
                    end_str = seg_match.group(2)
                    content = seg_match.group(3).strip()
                    
                    # Convert timestamps if needed (e.g., "01:30" to seconds)
                    start = parse_timestamp(start_str)
                    end = parse_timestamp(end_str)
                    
                    if start is not None and end is not None:
                        segments.append({
                            "start": start,
                            "end": end,
                            "text": content
                        })
                elif len(seg_match.groups()) == 2:  # Just timestamp and text
                    start_str = seg_match.group(1)
                    content = seg_match.group(2).strip()
                    
                    start = parse_timestamp(start_str)
                    if start is not None:
                        # Estimate end time (assume 5 seconds per segment if not specified)
                        segments.append({
                            "start": start,
                            "end": start + 5.0,
                            "text": content
                        })
        
        # Calculate total time from segments if not explicitly found
        if total_time == 0 and segments:
            total_time = sum((seg["end"] - seg["start"]) for seg in segments)
        
        # Create speaker data
        speakers[speaker_key] = {
            "role": role,
            "characteristics": characteristics if characteristics else "No specific characteristics identified",
            "total_time": total_time,
            "segments": segments
        }
    
    # If no speakers were found with the main pattern, try alternative patterns
    if not speakers:
        # Try to find any speaker references
        alt_patterns = [
            # Format: "Speaker 1: Details"
            r'(?:Speaker|SPEAKER)[\s_](\w+|"\w+"|[\d]+)[:\s]+([^\n]+)',
            # Format with brackets: [Speaker 1] Details
            r'\[(?:Speaker|SPEAKER)[\s_]?(\w+|"\w+"|[\d]+)\][:\s]*([^\n]+)',
            # Named speaker format: "John: Details"
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)[:\s]+([^\n]+)'
        ]
        
        for pattern in alt_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                speaker_id = match.group(1).strip('" ')
                description = match.group(2).strip()
                
                speaker_key = f"SPEAKER_{speaker_id}" if not speaker_id.startswith("SPEAKER_") else speaker_id
                
                # Only add if not already present
                if speaker_key not in speakers:
                    speakers[speaker_key] = {
                        "role": description[:50] + "..." if len(description) > 50 else description,
                        "characteristics": "No characteristics identified",
                        "total_time": 0,
                        "segments": []
                    }
    
    # If we still have no speakers, create a generic set based on any mentioned speakers
    if not speakers:
        # Just look for any mentions of "speaker" followed by numbers or identifiers
        generic_pattern = r'(?:speaker|SPEAKER)[\s_]?(\w+|"\w+"|[\d]+)'
        generic_matches = re.finditer(generic_pattern, text, re.IGNORECASE)
        
        speaker_ids = set()
        for match in generic_matches:
            speaker_ids.add(match.group(1).strip('" '))
        
        for speaker_id in speaker_ids:
            speaker_key = f"SPEAKER_{speaker_id}" if not speaker_id.startswith("SPEAKER_") else speaker_id
            speakers[speaker_key] = {
                "role": "Unknown speaker",
                "characteristics": "No characteristics identified",
                "total_time": 0,
                "segments": []
            }
    
    # If we STILL have no speakers, create at least one generic speaker
    if not speakers:
        speakers["SPEAKER_1"] = {
            "role": "Unknown speaker",
            "characteristics": "No characteristics identified",
            "total_time": 0,
            "segments": []
        }
    
    return speakers

def parse_timestamp(timestamp_str: str) -> Optional[float]:
    """Parse a timestamp string into seconds.
    
    Args:
        timestamp_str: Timestamp string (e.g., "10.5", "01:30", "01:30.5")
        
    Returns:
        Timestamp in seconds or None if parsing failed
    """
    try:
        # Simple format: "10.5"
        if ":" not in timestamp_str:
            return float(timestamp_str)
        
        # Format: "MM:SS" or "MM:SS.ms"
        if timestamp_str.count(":") == 1:
            parts = timestamp_str.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        
        # Format: "HH:MM:SS" or "HH:MM:SS.ms"
        if timestamp_str.count(":") == 2:
            parts = timestamp_str.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        
        return None
    except (ValueError, IndexError):
        return None