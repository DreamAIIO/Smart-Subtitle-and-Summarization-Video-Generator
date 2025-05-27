import os

def get_output_dir():
    """Get the permanent output directory path."""
    output_dir = os.path.join(os.getcwd(), "output", "videos")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_output_path(input_path: str, suffix: str, extension: str = None) -> str:
    """Generate an output path in the permanent directory.
    
    Args:
        input_path: Original input file path
        suffix: Suffix to add to the filename (e.g., '_subtitled', '_audio')
        extension: Optional new extension (e.g., 'mp4', 'srt'). If None, keeps original extension
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]
    if extension is None:
        extension = os.path.splitext(input_path)[1][1:]
    return os.path.join(get_output_dir(), f"{basename}{suffix}.{extension}") 