"""
Gemini model implementation for the subtitle generator application.
"""
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from smolagents.models import Model, ChatMessage

# Configure logging
logger = logging.getLogger("subtitle_generator")

class GeminiModel(Model):
    """Model implementation that uses Google's Gemini models."""
    
    def __init__(
        self, 
        model_id: str = "gemini-2.5-pro-preview-05-06",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Gemini model.
        
        Args:
            model_id: Gemini model ID to use
            api_key: API key for Google Generative AI
            **kwargs: Additional parameters for the model
        """
        self.model_id = model_id
        
        # Get API key from environment variable if not provided
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY environment variable not set and no api_key provided")
        
        # Initialize the Gemini API
        genai.configure(api_key=api_key)
        
        # Store additional parameters
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.top_p = kwargs.get("top_p", 0.95)
        self.top_k = kwargs.get("top_k", 32)
        
        # Record initialization time for logging purposes
        self._init_time = time.time()
        logger.info(f"Initialized GeminiModel with model_id={model_id}")
    
    def _extract_text_content(self, content):
        """Extract text content from various formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
            return text_content
        return str(content)
    
    def _format_prompt_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into a single prompt string for Gemini."""
        prompt = ""
        
        # Process each message
        for message in messages:
            role = message.get("role", "user")
            content = self._extract_text_content(message.get("content", ""))
            
            # Skip empty messages
            if not content.strip():
                continue
                
            # Format based on role
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            else:
                prompt += f"{role.capitalize()}: {content}\n\n"
        
        return prompt.strip()
    
    def get_tool_call(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get tool call from Gemini.
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions
            
        Returns:
            Tool call dictionary
        """
        try:
            # Format messages into a prompt
            prompt = self._format_prompt_from_messages(messages)
            
            # Configure the model
            model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "top_k": self.top_k
                }
            )
            
            # Create formatted function declarations
            gemini_tools = []
            for tool in tools:
                gemini_tools.append({
                    "function_declarations": [{
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }]
                })
            
            # Generate content with tools
            response = model.generate_content(prompt, tools=gemini_tools)
            
            # Check for function calls in the response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                function_call = part.function_call
                                return {
                                    "name": function_call.name,
                                    "arguments": json.loads(function_call.args)
                                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting tool call from Gemini: {str(e)}")
            return None
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> ChatMessage:
        """Call the Gemini model to generate a response.
        
        Args:
            messages: List of message dictionaries
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional arguments for the model
            
        Returns:
            ChatMessage with the response
        """
        try:
            # Format messages into a prompt
            prompt = self._format_prompt_from_messages(messages)
            
            # Log the request
            logger.info(f"Sending request to Gemini model: {self.model_id}")
            logger.info(f"Sending prompt: {prompt[:100]}...")  # Log part of the prompt
            
            # Configure the model
            model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "top_k": kwargs.get("top_k", self.top_k),
                    "stop_sequences": stop_sequences
                }
            )
            
            # Generate content directly (not using chat format)
            response = model.generate_content(prompt)
            logger.info("Successfully received response from Gemini")
            
            # Extract the text from the response
            if hasattr(response, 'text'):
                text = response.text
            else:
                text = ""
            
            # Return the response as a ChatMessage (with role="assistant" for smolagents)
            return ChatMessage(role="assistant", content=text)
            
        except Exception as e:
            logger.error(f"Error calling Gemini model: {str(e)}")
            # Return a minimal response in case of error (with role="assistant" for smolagents)
            return ChatMessage(role="assistant", content=f"Error generating response: {str(e)}")