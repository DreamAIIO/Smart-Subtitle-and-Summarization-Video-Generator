"""
Translation tool for translating subtitles to different languages.
"""
import os
import json
from typing import Dict, Optional, Any, List

import pysrt
from deep_translator import GoogleTranslator
from smolagents import Tool

from utils.error_handling import handle_errors, log_execution


class TranslationTool(Tool):
    """Tool for translating subtitle files to different languages."""
    
    name = "translator"
    description = """
    Translates subtitles from one language to another.
    """
    inputs = {
        "subtitle_path": {
            "type": "string",
            "description": "Path to the subtitle file to translate"
        },
        "source_language": {
            "type": "string",
            "description": "Source language code (auto for auto-detection)",
            "nullable": True
        },
        "target_language": {
            "type": "string",
            "description": "Target language code",
            "nullable": True
        },
        "options": {
            "type": "object",
            "description": "Additional translation options",
            "nullable": True
        }
    }
    output_type = "string"
    
    # Language code mapping between ISO and GoogleTranslator
    LANGUAGE_MAP = {
        "en": "english",
        "fr": "french",
        "es": "spanish",
        "de": "german",
        "it": "italian",
        "ja": "japanese",
        "ko": "korean",
        "zh": "chinese",
        "ru": "russian",
        "pt": "portuguese",
        "ar": "arabic",
        "nl": "dutch",
        "sv": "swedish",
        "fi": "finnish",
        "no": "norwegian",
        "da": "danish",
        "hi": "hindi",
        "bn": "bengali",
        "tr": "turkish",
        "pl": "polish",
        "vi": "vietnamese",
        "th": "thai",
        "auto": "auto"
    }
    
    @log_execution
    @handle_errors(default_return="Error translating subtitles")
    def forward(self, subtitle_path: str, source_language: str = "auto", target_language: str = "en", 
                options: Optional[Dict[str, Any]] = None) -> str:
        """Translate subtitles to a different language.
        
        Args:
            subtitle_path: Path to the subtitle file
            source_language: Source language code (auto for auto-detection)
            target_language: Target language code
            options: Additional translation options
            
        Returns:
            Path to the translated subtitle file
        """
        options = options or {}
        
        if not os.path.exists(subtitle_path):
            return f"Subtitle file does not exist: {subtitle_path}"
        
        # Map language codes
        source_lang = self._map_language_code(source_language)
        target_lang = self._map_language_code(target_language)
        
        # Determine file format
        file_ext = os.path.splitext(subtitle_path)[1].lower()
        
        output_path = options.get("output_path", None)
        if output_path is None:
            # Generate output path if not provided
            dirname = os.path.dirname(subtitle_path)
            basename = os.path.splitext(os.path.basename(subtitle_path))[0]
            output_path = os.path.join(dirname, f"{basename}_{target_language}{file_ext}")
        
        # Process based on file format
        if file_ext == '.srt':
            return self._translate_srt(subtitle_path, source_lang, target_lang, output_path, options)
        elif file_ext == '.json':
            return self._translate_json(subtitle_path, source_lang, target_lang, output_path, options)
        else:
            raise ValueError(f"Unsupported subtitle format: {file_ext}")
    
    @handle_errors()
    def _translate_srt(self, subtitle_path: str, source_lang: str, target_lang: str, 
                        output_path: str, options: Dict[str, Any]) -> str:
        """Translate SRT subtitle file.
        
        Args:
            subtitle_path: Path to the SRT file
            source_lang: Source language name
            target_lang: Target language name
            output_path: Path to save the translated file
            options: Additional translation options
            
        Returns:
            Path to the translated subtitle file
        """
        # Load subtitles
        subtitles = pysrt.open(subtitle_path)
        
        # Initialize translator
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        # Batch size for translation to avoid rate limits
        batch_size = options.get("batch_size", 10)
        preserve_formatting = options.get("preserve_formatting", True)
        
        # Process subtitles in batches
        for i in range(0, len(subtitles), batch_size):
            batch = subtitles[i:i+batch_size]
            
            # Extract text for translation
            texts = [item.text for item in batch]
            
            try:
                # Translate texts
                if preserve_formatting:
                    # Translate each text individually to preserve formatting
                    translated_texts = [translator.translate(text) for text in texts]
                else:
                    # Translate all texts at once for better context
                    combined_text = "\n\n".join(texts)
                    translated_combined = translator.translate(combined_text)
                    translated_texts = translated_combined.split("\n\n")
                    
                    # Ensure we have the right number of translations
                    if len(translated_texts) != len(texts):
                        # Fallback to individual translation
                        translated_texts = [translator.translate(text) for text in texts]
                
                # Update subtitles with translations
                for j, item in enumerate(batch):
                    if j < len(translated_texts):
                        item.text = translated_texts[j]
                    else:
                        # If we don't have a translation, keep the original text
                        item.text = texts[j]
                
            except Exception as e:
                # If translation fails for a batch, keep original text
                for j, item in enumerate(batch):
                    item.text = texts[j]
                continue
        
        # Save translated subtitles
        subtitles.save(output_path, encoding='utf-8')
        
        return output_path
    
    @handle_errors()
    def _translate_json(self, json_path: str, source_lang: str, target_lang: str, 
                         output_path: str, options: Dict[str, Any]) -> str:
        """Translate transcript JSON file.
        
        Args:
            json_path: Path to the JSON file
            source_lang: Source language name
            target_lang: Target language name
            output_path: Path to save the translated file
            options: Additional translation options
            
        Returns:
            Path to the translated JSON file
        """
        # Load JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # Initialize translator
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        
        # Extract segments
        segments = transcript.get("segments", [])
        
        # Batch size for translation to avoid rate limits
        batch_size = options.get("batch_size", 10)
        
        # Process segments in batches
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            
            # Extract text for translation
            texts = [segment["text"] for segment in batch]
            
            # Translate texts
            translated_texts = [translator.translate(text) for text in texts]
            
            # Update segments with translations
            for j, segment in enumerate(batch):
                if j < len(translated_texts):
                    segment["text"] = translated_texts[j]
        
        # Save translated transcript
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _map_language_code(self, code: str) -> str:
        """Map ISO language code to the format expected by the translator.
        
        Args:
            code: ISO language code (e.g., 'en', 'fr')
            
        Returns:
            Language name for the translator
        """
        # Handle common codes
        normalized_code = code.lower().split('-')[0]  # Convert to lowercase and strip country code
        
        if normalized_code in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[normalized_code]
        
        # For unsupported languages, return as is and let the translator handle it
        return normalized_code