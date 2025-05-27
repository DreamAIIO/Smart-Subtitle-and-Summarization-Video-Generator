"""
Updated Streamlit application that uses the hybrid subtitle generator system.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import time

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import the hybrid subtitle generator
from hybrid_subtitle_generator import HybridSubtitleGenerator

# Import utilities
from utils.file_handling import TempFileManager
from utils.file_path_utils import PathManager
from utils.error_handling import ProcessingError, handle_errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("subtitle_generator_app")

# Set page configuration
st.set_page_config(
    page_title="Smart Video Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "temp_file_manager" not in st.session_state:
    st.session_state.temp_file_manager = TempFileManager()

if "path_manager" not in st.session_state:
    # Use the current working directory as the base directory
    st.session_state.path_manager = PathManager(os.getcwd())

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "current_video" not in st.session_state:
    st.session_state.current_video = None

if "generator" not in st.session_state:
    # Initialize with default values - will be updated when user selects a model
    st.session_state.generator = None

if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

if "app_dir" not in st.session_state:
    # Get the directory where the streamlit app is running
    st.session_state.app_dir = os.getcwd()
    # Ensure output directories exist
    os.makedirs(os.path.join(st.session_state.app_dir, "output"), exist_ok=True)
    for subdir in ["videos", "audios", "transcripts", "subtitles", "summaries", "temp"]:
        os.makedirs(os.path.join(st.session_state.app_dir, "output", subdir), exist_ok=True)


def initialize_generator(model_id, api_key=None, model_kwargs=None):
    """Initialize the subtitle generator with the selected model."""
    try:
        # Use the app directory as the base directory for file operations
        st.session_state.generator = HybridSubtitleGenerator(
            model_id=model_id,
            api_key=api_key,
            model_kwargs=model_kwargs or {},
            base_dir=st.session_state.app_dir
        )
        return True, "Generator initialized successfully!"
    except Exception as e:
        logger.error(f"Error initializing generator: {str(e)}")
        return False, f"Error initializing generator: {str(e)}"


def main():
    """Main application function."""
    # Display the header
    st.title("🎬 Smart Video Subtitle Generator")
    st.markdown("""
    Upload a video to automatically generate subtitles, translate them, identify speakers, 
    and create content summaries - all powered by AI!
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        model_options = {
            "Gemini Pro Flash": "gemini-2.5-flash-preview-04-17",
            "Gemini Pro": "gemini-2.5-pro-preview-05-06"
        }
        selected_model = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=1  # Default to Gemini Pro
        )
        model_id = model_options[selected_model]
        
        # API key for Gemini
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", "")
        )
        api_key = gemini_api_key if gemini_api_key else None
        
        # Additional parameters for Gemini
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Controls randomness. Lower values (e.g., 0.2) make output more focused."
        )
        
        model_kwargs = {
            "temperature": temperature,
            "max_tokens": 8192  # Set a reasonable default for response length
        }
        
        # Language options
        default_language = st.radio(
            "Default Language Detection",
            ["Auto Detect", "Specify Language"],
            index=0
        )
        
        language_code = None
        if default_language == "Specify Language":
            language_options = {
                "English": "en",
                "Spanish": "es",
                "French": "fr",
                "German": "de",
                "Chinese": "zh",
                "Japanese": "ja",
                "Korean": "ko",
                "Russian": "ru",
                "Arabic": "ar",
                "Hindi": "hi"
            }
            selected_language = st.selectbox(
                "Language",
                list(language_options.keys()),
                index=0
            )
            language_code = language_options[selected_language]
        
        # Additional feature toggles
        st.subheader("Features")
        enable_speaker_recognition = st.toggle("Enable Speaker Recognition", value=False)
        
        # Add speaker recognition options when enabled
        if enable_speaker_recognition:
            st.write("Speaker Recognition Options:")
            expected_speakers = st.number_input(
                "Expected Number of Speakers", 
                min_value=1, 
                max_value=10, 
                value=2,
                help="Approximate number of speakers in the video"
            )
            
            use_semantic_analysis = st.checkbox(
                "Use Semantic Analysis", 
                value=True,
                help="Use Gemini to identify speaker names and characteristics"
            )
            
            hf_token = st.text_input(
                "HuggingFace Token for pyannote.audio (optional)",
                type="password",
                value=os.environ.get("HF_TOKEN", ""),
                help="Authentication token for pyannote.audio speaker diarization model"
            )
        
        enable_summarization = st.toggle("Generate Content Summary", value=True)
        enable_translation = st.toggle("Enable Translation", value=False)
        
        translation_language = None
        if enable_translation:
            translation_options = {
                "English": "en",
                "Spanish": "es",
                "French": "fr",
                "German": "de",
                "Chinese": "zh",
                "Japanese": "ja",
                "Korean": "ko",
                "Russian": "ru",
                "Arabic": "ar",
                "Hindi": "hi"
            }
            selected_translation = st.selectbox(
                "Translate To",
                list(translation_options.keys()),
                index=0
            )
            translation_language = translation_options[selected_translation]
        
        # Advanced options
        st.subheader("Advanced Options")
        show_advanced = st.checkbox("Show Advanced Options", value=False)
        
        advanced_options = {}
        if show_advanced:
            subtitle_format = st.selectbox(
                "Subtitle Format",
                ["SRT", "VTT"],
                index=0
            )
            advanced_options["subtitle_format"] = subtitle_format.lower()
            
            if enable_summarization:
                summarization_method = st.selectbox(
                    "Summarization Method",
                    ["lexrank", "lsa", "luhn"],
                    index=0
                )
                summary_length = st.slider(
                    "Summary Length (sentences)",
                    min_value=3,
                    max_value=10,
                    value=5
                )
                advanced_options["summary_method"] = summarization_method
                advanced_options["summary_length"] = summary_length
                
                # Add option for verbose logging during summarization
                advanced_options["verbose_summarization"] = st.checkbox(
                    "Enable Verbose Summarization Logging", 
                    value=True,
                    help="Enable detailed logging during summarization to help diagnose issues"
                )
            
            # Add speaker recognition advanced options if enabled
            if enable_speaker_recognition:
                st.write("Advanced Speaker Recognition Options:")
                min_speakers = st.number_input(
                    "Minimum Speakers", 
                    min_value=1, 
                    max_value=10, 
                    value=1,
                    help="Minimum number of speakers to identify"
                )
                max_speakers = st.number_input(
                    "Maximum Speakers", 
                    min_value=min_speakers, 
                    max_value=20, 
                    value=min(10, max(min_speakers + 2, 5)),
                    help="Maximum number of speakers to identify"
                )
                advanced_options["min_speakers"] = min_speakers
                advanced_options["max_speakers"] = max_speakers
                advanced_options["verbose_speaker_recognition"] = st.checkbox(
                    "Enable Verbose Speaker Recognition Logging", 
                    value=True,
                    help="Enable detailed logging during speaker recognition"
                )
                
            # Add subtitle embedding options
            st.subheader("Subtitle Embedding Options")
            subtitle_method = st.radio(
                "Subtitle Embedding Method",
                ["Auto (try all methods)", "Direct Burn", "Explicit Mapping", "Simple Filter"],
                index=0,
                help="Method used to add subtitles to video. Auto will try all methods in sequence."
            )
            
            if subtitle_method != "Auto (try all methods)":
                if subtitle_method == "Direct Burn":
                    advanced_options["subtitle_method"] = "burn"
                elif subtitle_method == "Explicit Mapping":
                    advanced_options["subtitle_method"] = "map"
                elif subtitle_method == "Simple Filter":
                    advanced_options["subtitle_method"] = "filter"
            
            # Add option for ffmpeg path
            use_custom_ffmpeg = st.checkbox("Use Custom FFmpeg Path", value=False)
            if use_custom_ffmpeg:
                ffmpeg_path = st.text_input("FFmpeg Binary Path", value="")
                if ffmpeg_path and os.path.exists(ffmpeg_path):
                    advanced_options["ffmpeg_path"] = ffmpeg_path
                else:
                    st.warning("Invalid FFmpeg path or path doesn't exist. Using system default.")
            
            # Add debugging options
            st.subheader("Debug Options")
            debug_mode = st.checkbox("Enable Debug Mode", value=False)
            if debug_mode:
                advanced_options["debug_mode"] = True
                advanced_options["verbose_logging"] = True
                debug_log_path = st.text_input(
                    "Debug Log Path", 
                    value=os.path.join(st.session_state.app_dir, "output", "debug.log")
                )
                advanced_options["debug_log_path"] = debug_log_path
        
        # Update generator if model or token changes
        if st.button("Update Configuration"):
            with st.spinner("Updating configuration..."):
                success, message = initialize_generator(
                    model_id=model_id,
                    api_key=api_key,
                    model_kwargs=model_kwargs
                )
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Video")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload a video file to generate subtitles"
        )
        
        if uploaded_file:
            # Save the uploaded file to a persistent location
            temp_file_path = st.session_state.temp_file_manager.save_uploaded_file(uploaded_file)
            
            if not temp_file_path:
                st.error("Failed to save uploaded file. Please try again.")
                st.stop()
                
            # Process the Streamlit path to ensure accessibility
            processed_path = st.session_state.path_manager.handle_streamlit_path(temp_file_path)
            st.session_state.current_video = processed_path
            
            # Display the uploaded video
            st.video(temp_file_path)
            
            # Process button
            process_options = {}
            if language_code:
                process_options["language"] = language_code
            
            process_options.update(advanced_options)
            
            if enable_translation:
                process_options["target_language"] = translation_language
            
            # Add speaker recognition options if enabled
            if enable_speaker_recognition:
                if 'expected_speakers' in locals():
                    process_options["expected_speakers"] = expected_speakers
                if 'min_speakers' in advanced_options and 'max_speakers' in advanced_options:
                    process_options["min_speakers"] = advanced_options["min_speakers"]
                    process_options["max_speakers"] = advanced_options["max_speakers"]
                if 'use_semantic_analysis' in locals():
                    process_options["use_semantic_analysis"] = use_semantic_analysis
                if 'hf_token' in locals() and hf_token:
                    process_options["auth_token"] = hf_token
                if 'verbose_speaker_recognition' in advanced_options:
                    process_options["verbose"] = advanced_options["verbose_speaker_recognition"]
            
            if st.button("Generate Subtitles"):
                # Make sure the generator is initialized
                if st.session_state.generator is None:
                    # Initialize with current settings
                    with st.spinner("Initializing generator..."):
                        success, message = initialize_generator(
                            model_id=model_id,
                            api_key=api_key,
                            model_kwargs=model_kwargs
                        )
                        
                        if not success:
                            st.error(message)
                            st.stop()
                
                with st.spinner("Processing video..."):
                    try:
                        # Create a status placeholder for progress updates
                        status_placeholder = st.empty()
                        status_placeholder.info("Starting video processing...")
                        
                        # Store the status placeholder in session state for updates
                        st.session_state.processing_status = status_placeholder
                        
                        # Determine operations to perform
                        operations = ["subtitles"]  # Always generate subtitles
                        
                        if enable_summarization:
                            operations.append("summarization")
                        
                        if enable_speaker_recognition:
                            operations.append("speaker_recognition")
                        
                        if enable_translation:
                            operations.append("translation")
                        
                        # Process the video
                        result = process_video(st.session_state.current_video, operations, process_options)
                        
                        # Store the result in session state
                        st.session_state.processed_files[st.session_state.current_video] = result
                        
                        status_placeholder.success("Video processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
    
    with col2:
        st.subheader("Results")
        
        # Display results if available
        if st.session_state.current_video and st.session_state.current_video in st.session_state.processed_files:
            result = st.session_state.processed_files[st.session_state.current_video]
            
            # Extract relevant paths from result
            subtitled_video_path = result.get("subtitled_video_path")
            subtitle_path = result.get("subtitle_path")
            transcript_path = result.get("transcript_path")
            translated_subtitle_path = result.get("translated_subtitle_path")
            summary_path = result.get("summary_path")
            
            # Validate and find paths if needed
            if subtitled_video_path and not os.path.exists(subtitled_video_path):
                subtitled_video_path = st.session_state.path_manager.find_nearest_match(subtitled_video_path, directory="videos")
            
            if subtitle_path and not os.path.exists(subtitle_path):
                subtitle_path = st.session_state.path_manager.find_nearest_match(subtitle_path, directory="subtitles")
                
            if transcript_path and not os.path.exists(transcript_path):
                transcript_path = st.session_state.path_manager.find_nearest_match(transcript_path, directory="transcripts")
                
            if summary_path and not os.path.exists(summary_path):
                summary_path = st.session_state.path_manager.find_nearest_match(summary_path, directory="summaries")
            
            # Display language detection result
            if "detected_language" in result:
                st.info(f"Detected Language: {result['detected_language']}")
            elif "language" in result:
                st.info(f"Detected Language: {result['language']}")
            
            # Display the subtitled video if available
            if subtitled_video_path and os.path.exists(subtitled_video_path):
                st.subheader("Video with Subtitles")
                st.video(subtitled_video_path)
                
                # Download button for the subtitled video
                with open(subtitled_video_path, "rb") as file:
                    st.download_button(
                        label="Download Video with Subtitles",
                        data=file,
                        file_name=os.path.basename(subtitled_video_path),
                        mime="video/mp4"
                    )
                
                # Optional: If we have a fallback info file, display its contents
                fallback_info_path = os.path.splitext(subtitled_video_path)[0] + "_subtitles_info.txt"
                if os.path.exists(fallback_info_path):
                    with open(fallback_info_path, "r", encoding="utf-8") as f:
                        fallback_info = f.read()
                    st.warning(f"Note about subtitles: {fallback_info}")
            
            # Display subtitle file if available
            if subtitle_path and os.path.exists(subtitle_path):
                st.subheader("Subtitles")
                
                with open(subtitle_path, "r", encoding="utf-8") as file:
                    subtitle_content = file.read()
                
                st.text_area("Subtitle Content (SRT)", subtitle_content, height=200)
                
                # Download button for subtitle file
                with open(subtitle_path, "rb") as file:
                    st.download_button(
                        label="Download Subtitle File",
                        data=file,
                        file_name=os.path.basename(subtitle_path),
                        mime="text/plain"
                    )
            
            # Display translated subtitles if available
            if translated_subtitle_path and os.path.exists(translated_subtitle_path):
                st.subheader("Translated Subtitles")
                
                with open(translated_subtitle_path, "r", encoding="utf-8") as file:
                    translated_content = file.read()
                
                st.text_area("Translated Subtitle Content", translated_content, height=200)
                
                # Download button for translated subtitle file
                with open(translated_subtitle_path, "rb") as file:
                    st.download_button(
                        label="Download Translated Subtitle File",
                        data=file,
                        file_name=os.path.basename(translated_subtitle_path),
                        mime="text/plain"
                    )
            
            # Display speaker recognition results if available
            if "speakers" in result:
                st.subheader("Speaker Recognition")
                
                speakers = result.get("speakers", {})
                
                # Create tabs for different views of speaker data
                speaker_tabs = st.tabs(["Overview", "Speaker Details", "Timeline"])
                
                with speaker_tabs[0]:  # Overview tab
                    # Create a table of speakers and their speaking time
                    speaker_data = []
                    for speaker_id, speaker_info in speakers.items():
                        # Get speaker role or default to speaker ID
                        role = speaker_info.get("role", "Unknown")
                        # Add speaker ID if role doesn't already include it
                        if speaker_id not in role:
                            display_name = f"{role} ({speaker_id})"
                        else:
                            display_name = role
                            
                        speaker_data.append({
                            "Speaker": display_name,
                            "Speaking Time (s)": round(speaker_info.get("total_time", 0), 2),
                            "Segments": len(speaker_info.get("segments", []))
                        })
                    
                    # Sort by speaking time
                    speaker_data = sorted(speaker_data, key=lambda x: x["Speaking Time (s)"], reverse=True)
                    
                    # Create a bar chart of speaking time distribution
                    if speaker_data:
                        st.subheader("Speaking Time Distribution")
                        
                        # Convert to appropriate format for chart
                        chart_data = {
                            "Speaker": [item["Speaker"] for item in speaker_data],
                            "Speaking Time (s)": [item["Speaking Time (s)"] for item in speaker_data]
                        }
                        
                        # Display as a bar chart
                        st.bar_chart(chart_data, x="Speaker", y="Speaking Time (s)")
                        
                        # Show the table
                        st.table(speaker_data)
                
                with speaker_tabs[1]:  # Speaker Details tab
                    for speaker_id, speaker_info in sorted(
                        speakers.items(), 
                        key=lambda x: x[1].get("total_time", 0), 
                        reverse=True
                    ):
                        # Create an expander for each speaker
                        with st.expander(f"{speaker_info.get('role', speaker_id)} ({round(speaker_info.get('total_time', 0), 2)}s)"):
                            # Show characteristics
                            st.write(f"**Characteristics:** {speaker_info.get('characteristics', 'None specified')}")
                            
                            # Show segments if available
                            segments = speaker_info.get("segments", [])
                            if segments:
                                st.write(f"**Speaking segments:** {len(segments)}")
                                
                                # Show first few segments as examples
                                for i, segment in enumerate(segments[:5]):
                                    start = segment.get("start", 0)
                                    end = segment.get("end", 0)
                                    text = segment.get("text", "")
                                    
                                    # Format as MM:SS
                                    start_fmt = f"{int(start/60):02d}:{int(start%60):02d}"
                                    end_fmt = f"{int(end/60):02d}:{int(end%60):02d}"
                                    
                                    st.write(f"**{start_fmt} - {end_fmt}:** {text}")
                                
                                if len(segments) > 5:
                                    st.write(f"... and {len(segments) - 5} more segments")
                
                with speaker_tabs[2]:  # Timeline tab
                    # Create a timeline view of all segments
                    all_segments = []
                    for speaker_id, speaker_info in speakers.items():
                        role = speaker_info.get("role", speaker_id)
                        for segment in speaker_info.get("segments", []):
                            all_segments.append({
                                "start": segment.get("start", 0),
                                "end": segment.get("end", 0),
                                "speaker": role if speaker_id in role else f"{role} ({speaker_id})",
                                "text": segment.get("text", "")
                            })
                    
                    # Sort by start time
                    all_segments.sort(key=lambda x: x["start"])
                    
                    # Display timeline
                    if all_segments:
                        st.write("**Timeline of speech segments:**")
                        for segment in all_segments:
                            start = segment["start"]
                            end = segment["end"]
                            speaker = segment["speaker"]
                            text = segment["text"]
                            
                            # Format as MM:SS
                            start_fmt = f"{int(start/60):02d}:{int(start%60):02d}"
                            end_fmt = f"{int(end/60):02d}:{int(end%60):02d}"
                            
                            # Display with different colors for different speakers
                            st.markdown(f"**{start_fmt} - {end_fmt}** [{speaker}]: {text}")
                
                # Download button for speaker data
                if "json_path" in result and os.path.exists(result["json_path"]):
                    with open(result["json_path"], "rb") as file:
                        st.download_button(
                            label="Download Speaker Data (JSON)",
                            data=file,
                            file_name=os.path.basename(result["json_path"]),
                            mime="application/json"
                        )
                elif "text_path" in result and os.path.exists(result["text_path"]):
                    with open(result["text_path"], "rb") as file:
                        st.download_button(
                            label="Download Speaker Analysis (Text)",
                            data=file,
                            file_name=os.path.basename(result["text_path"]),
                            mime="text/plain"
                        )
            
            # Display summary if available
            if summary_path and os.path.exists(summary_path):
                st.subheader("Content Summary")
                
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        if summary_path.endswith(".json"):
                            summary_data = json.load(f)
                            summary_content = summary_data.get("text", "")
                            if not summary_content and "abstractive_summary" in summary_data:
                                summary_content = summary_data["abstractive_summary"]
                            elif not summary_content and "extractive_summary" in summary_data:
                                summary_content = " ".join(summary_data["extractive_summary"])
                        else:
                            summary_content = f.read()
                    
                    st.text_area("Summary", summary_content, height=200)
                    
                    # If we have timeline data, display it
                    if summary_path.endswith(".json"):
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary_data = json.load(f)
                            if "timeline" in summary_data and summary_data["timeline"]:
                                st.subheader("Timeline Highlights")
                                for point in summary_data["timeline"]:
                                    st.markdown(f"**{point['timestamp']}**: {point['summary']}")
                    
                    # Download button for summary file
                    with open(summary_path, "rb") as f:
                        st.download_button(
                            label="Download Summary",
                            data=f,
                            file_name=os.path.basename(summary_path),
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"Error loading summary: {str(e)}")
                    st.info("Summary was generated but could not be displayed. You can still download it.")
            elif "summary_content" in result:
                # If we have direct summary content in the result dict
                st.subheader("Content Summary")
                st.text_area("Summary", result["summary_content"], height=200)
            
            # If there's an error message, display it
            if "error" in result:
                st.error(f"Error: {result['error']}")
            
            # If there's a warning message, display it
            if "warning" in result:
                st.warning(f"Warning: {result['warning']}")
                
            # Display partial errors if present
            for error_key in [k for k in result.keys() if k.startswith("error_")]:
                st.warning(f"{error_key.replace('error_', '').title()}: {result[error_key]}")
                
            # Add a retry button
            if "error" in result or "warning" in result or any(k.startswith("error_") for k in result.keys()):
                if st.button("Retry Processing"):
                    with st.spinner("Reprocessing video..."):
                        try:
                            # Create a status placeholder for progress updates
                            status_placeholder = st.empty()
                            status_placeholder.info("Starting video reprocessing...")
                            
                            # Store the status placeholder in session state for updates
                            st.session_state.processing_status = status_placeholder
                            
                            # Add retry flag to options
                            process_options["retry"] = True
                            
                            # Determine operations to perform
                            operations = ["subtitles"]  # Always generate subtitles
                            
                            if enable_summarization:
                                operations.append("summarization")
                            
                            if enable_speaker_recognition:
                                operations.append("speaker_recognition")
                            
                            if enable_translation:
                                operations.append("translation")
                            
                            # Process the video
                            result = process_video(st.session_state.current_video, operations, process_options)
                            
                            # Store the result in session state
                            st.session_state.processed_files[st.session_state.current_video] = result
                            
                            status_placeholder.success("Video reprocessed successfully!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error reprocessing video: {str(e)}")


@handle_errors(default_return={"error": "Processing error occurred"})
def process_video(video_path: str, operations: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
    """Process a video with the generator.
    
    Args:
        video_path: Path to the video file
        operations: List of operations to perform
        options: Processing options
        
    Returns:
        Dictionary with processing results
    """
    # Log the operation start
    logger.info(f"Starting video processing with operations: {operations}")
    
    # Check if generator is initialized
    if st.session_state.generator is None:
        # Initialize with default settings for Gemini Pro
        success, message = initialize_generator("gemini-2.5-pro-preview-05-06")
        if not success:
            return {"error": message}
    
    # Process with the generator
    try:
        # Show a progress message
        progress_placeholder = st.session_state.processing_status
        progress_placeholder.info("Processing video... This may take a few minutes.")
        
        # Run the generator
        result = st.session_state.generator.process_video(video_path, operations, options)
        
        # Update final status
        success_msg = "Video processed successfully!"
        if "error" in result or "warning" in result or any(k.startswith("error_") for k in result.keys()):
            success_msg += " (with some issues)"
        progress_placeholder.success(success_msg)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        if hasattr(st.session_state, "processing_status") and st.session_state.processing_status:
            st.session_state.processing_status.error(f"Error processing video: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    main()