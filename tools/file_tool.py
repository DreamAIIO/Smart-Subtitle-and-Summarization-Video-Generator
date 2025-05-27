# Create a new file called tools/file_tool.py
"""
File operations tool for the subtitle generator application.
"""
import os
import json
from typing import Dict, Optional, Any

from smolagents import Tool
from utils.error_handling import handle_errors, log_execution

class FileOperationsTool(Tool):
    """Tool for file operations."""
    
    name = "file_operator"
    description = """
    Performs file operations like reading and writing files.
    """
    inputs = {
        "operation": {
            "type": "string",
            "description": "The operation to perform (read, write)"
        },
        "file_path": {
            "type": "string",
            "description": "Path to the file"
        },
        "content": {
            "type": "string",
            "description": "Content to write (for write operation)",
            "nullable": True
        },
        "options": {
            "type": "object",
            "description": "Additional options",
            "nullable": True
        }
    }
    output_type = "string"
    
    @log_execution
    @handle_errors(default_return="Error performing file operation")
    def forward(self, operation: str, file_path: str, content: Optional[str] = None, 
                options: Optional[Dict[str, Any]] = None) -> str:
        """Perform a file operation.
        
        Args:
            operation: Operation to perform (read, write)
            file_path: Path to the file
            content: Content to write (for write operation)
            options: Additional options
            
        Returns:
            Result of the operation
        """
        options = options or {}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if operation == "read":
            encoding = options.get("encoding", "utf-8")
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        elif operation == "write":
            if content is None:
                return "Error: No content provided for write operation"
            
            encoding = options.get("encoding", "utf-8")
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        else:
            return f"Unsupported operation: {operation}"