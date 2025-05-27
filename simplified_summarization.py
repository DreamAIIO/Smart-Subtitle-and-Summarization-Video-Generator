"""
A simplified summarization agent that handles path issues better.
"""
import os
import json
import re
import logging
from typing import Dict, Any, Optional

from smolagents import CodeAgent
from smolagents.models import Model

# Configure logging
logger = logging.getLogger("subtitle_generator")

def create_simple_summarization_agent(model: Model, summarization_tool, file_tool, quality_tool):
    """Create a simplified agent for content summarization.
    
    Args:
        model: The LLM model to use
        summarization_tool: The summarization tool to use
        file_tool: The file operations tool
        quality_tool: The quality verification tool
        
    Returns:
        CodeAgent for summarization
    """
    # Common imports for the agent
    COMMON_IMPORTS = ["json", "os", "re", "tempfile", "shutil", "uuid", "open", "subprocess", "globals"]
    
    # Create the agent with a much simpler prompt
    agent = CodeAgent(
        tools=[summarization_tool, file_tool, quality_tool],
        model=model,
        name="summary_agent",
        description="Generates high-quality summaries from transcripts, focusing on key content.",
        max_steps=3,
        planning_interval=None,  # Disable planning to reduce LLM calls
        additional_authorized_imports=COMMON_IMPORTS
    )
    
    return agent

def run_summarization(agent: CodeAgent, transcript_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the summarization agent with a simplified approach.
    
    Args:
        agent: The CodeAgent to use
        transcript_path: Path to the transcript file
        options: Additional options for summarization
        
    Returns:
        Dictionary with summarization results
    """
    options = options or {}
    
    # Create a much more detailed task for the agent
    task = f"""
    Generate a comprehensive, thoughtful summary of the transcript at {transcript_path}.
    
    IMPORTANT: This should be a TRUE SUMMARY that distills and analyzes the content, not just a repetition of phrases from the transcript.
    
    Follow these steps:
    
    1. Carefully read and analyze the entire transcript to understand the overall topic and key points
    2. Identify the main themes, concepts, and important information
    3. Create an abstractive summary that:
       - Explains the main topic in your own words
       - Highlights the core message and significance
       - Synthesizes the key points into a coherent narrative
       - Does NOT merely repeat phrases from the transcript
    4. Also extract 3-5 key timeline highlights with timestamps
    5. Save the summary in both JSON and text formats
    
    Your summary should help someone who hasn't watched the video understand what it's about and what's important.
    """
    
    # Add options to the task if provided
    if "summary_method" in options:
        method = options["summary_method"]
        task += f"""
        
        For the summarization approach, use the {method} method to identify key points.
        
        The {method} method should help you:
        - Identify the most important sentences in the content
        - Recognize central themes and concepts
        - Determine which information is essential vs. peripheral
        """
    
    if "summary_length" in options:
        length = options["summary_length"]
        task += f"""
        
        Create a focused summary of approximately {length} sentences that captures the essence of the content.
        Being concise is important, but make sure to include all critical information.
        """
    
    # Add specific instructions about writing style
    task += """
    
    WRITING GUIDELINES:
    1. Write in your own words - DO NOT copy phrases directly from the transcript
    2. Use clear, concise language that is easy to understand
    3. Focus on the meaning and significance of what was said, not just what was literally stated
    4. Organize the information logically with proper transitions
    5. Make sure the summary stands on its own as a coherent piece of writing
    
    Remember, a good summary should tell the reader what the video is ABOUT, not just repeat what was SAID.
    """
    
    try:
        logger.info("Running summarization agent with enhanced instructions")
        result = agent.run(task)

        # 1. Find the summary .txt file path (look for _summary.txt, _summary_final.txt, or summary_output.txt)
        txt_path = None
        if isinstance(result, dict):
            txt_path = result.get("text_output_path")
        if not txt_path:
            # Try to find a .txt path in the result string
            match = re.search(r'(/[^\s]+(?:_summary|_summary_final|summary_output)\.txt)', str(result))
            if match:
                txt_path = match.group(1)

        # 2. If not found, try to infer from transcript_path
        if not txt_path:
            base = os.path.splitext(transcript_path)[0]
            for suffix in ["_summary.txt", "_summary_final.txt", "summary_output.txt"]:
                candidate = base + suffix
                if os.path.exists(candidate):
                    txt_path = candidate
                    break

        # 3. Read the summary text from the .txt file
        summary_content = None
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                summary_content = f.read()
                # If the file has "Summary:\n..." and "Timeline Highlights:", only keep the summary part
                if "Summary:" in summary_content:
                    summary_content = summary_content.split("Summary:")[1]
                if "Timeline Highlights:" in summary_content:
                    summary_content = summary_content.split("Timeline Highlights:")[0]
                summary_content = summary_content.strip()

        # 4. Fallback: Try to read from JSON summary if .txt not found
        if not summary_content:
            json_path = None
            if isinstance(result, dict):
                json_path = result.get("json_output_path")
            if not json_path:
                match = re.search(r'(/[^\s]+(?:_summary|_summary_final|summary_output)\.json)', str(result))
                if match:
                    json_path = match.group(1)
            if not json_path:
                base = os.path.splitext(transcript_path)[0]
                for suffix in ["_summary.json", "_summary_final.json", "summary_output.json"]:
                    candidate = base + suffix
                    if os.path.exists(candidate):
                        json_path = candidate
                        break
            if json_path and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    summary_content = (
                        data.get("abstractive_summary") or
                        data.get("summary") or
                        data.get("summary_text") or
                        data.get("text")
                    )

        # 5. Final fallback: If still nothing, show a friendly error
        if not summary_content or summary_content.strip() == "":
            summary_content = "Summary could not be generated. Please check the transcript and try again."

        return {
            "summary_path": txt_path,
            "summary_content": summary_content,
        }

    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return {
            "error": str(e),
            "raw_error": str(e)
        }