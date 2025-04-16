def get_custom_metadata(info, audio):
    """
    Maps clean audio files to their corresponding degraded versions.
    
    Args:
        info (dict): Dictionary containing metadata about the audio file
        audio (np.ndarray): The audio data (not used in this case)
        
    Returns:
        dict: Dictionary containing the path to the degraded audio folder
    """
    # Get the path to the clean audio file from info
    clean_path = info['path']
    
    # Convert clean path to degraded path
    # Assuming degraded files are in a parallel directory structure
    # For example, if clean files are in /data/clean/...
    # then degraded files would be in /data/degraded/...
    import os
    clean_dir = os.path.dirname(clean_path)
    parent_dir = os.path.dirname(clean_dir)
    degraded_dir = os.path.join(parent_dir, 'degraded')
    
    # Verify the degraded file exists
    if not os.path.exists(degraded_dir):
        raise FileNotFoundError(f"Degraded audio folder not found: {degraded_dir}")
    
    return {
        "degraded_audio_path": degraded_dir
    } 