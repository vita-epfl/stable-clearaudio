import torchaudio
import torch
import logging


def get_custom_metadata(info, audio):
    """
    Maps clean audio files to their corresponding degraded versions.
    
    Args:
        info (dict): Dictionary containing metadata about the audio file
        audio (np.ndarray): The clean audio data
        
    Returns:
        dict: Dictionary containing the degraded audio data
    """
    # Get the path to the clean audio file from info
    clean_path = info['path']
    
    t_start, t_end = info["timestamps"]
    total_length = info["total_length"]
    # Convert clean path to degraded path
    # Assuming degraded files are in a parallel directory structure
    # For example, if clean files are in /data/clean/...
    # then degraded files would be in /data/degraded/...
    import os
    clean_dir = os.path.dirname(clean_path)
    parent_dir = os.path.dirname(clean_dir)
    degraded_dir = os.path.join(parent_dir, 'degraded')
    filename = os.path.basename(clean_path)
    degraded_path = os.path.join(degraded_dir, filename)
    
    # Verify the degraded file exists
    if not os.path.exists(degraded_path):
        raise FileNotFoundError(f"Degraded audio file not found: {degraded_path}")
    
    degraded_audio, sr = torchaudio.load(degraded_path)
    
    # First apply the timestamp slicing
    degraded_audio = degraded_audio[:, round(t_start*total_length):round(t_end*total_length)]
    
    # Now ensure degraded audio matches clean audio size
    target_length = audio.shape[-1]  # Get length from clean audio
    current_length = degraded_audio.shape[-1]
    
    if current_length < target_length:
        # Pad with zeros if degraded audio is shorter
        pad_length = target_length - current_length
        degraded_audio = torch.nn.functional.pad(degraded_audio, (0, pad_length))
    elif current_length > target_length:
        # Truncate if degraded audio is longer
        degraded_audio = degraded_audio[:, :target_length]
    
    # Return the degraded audio data
    return {
        "degraded_audio": degraded_audio
    } 