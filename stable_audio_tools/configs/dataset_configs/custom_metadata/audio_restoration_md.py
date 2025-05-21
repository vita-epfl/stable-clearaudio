import torchaudio
import torch
import logging
import os
import random
from pathlib import Path
from stable_audio_tools.transforms import signal
from omegaconf import OmegaConf

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

def get_custom_metadata(info, audio, custom_metadata_args=None):
    build_degraded = custom_metadata_args.get("build_degraded", False)
    if build_degraded:
        return get_metadata_on_the_fly(info, audio, custom_metadata_args)
    elif not build_degraded:
        return get_metadata_from_local(info, audio)
    else:
        raise ValueError("Invalid build_degraded value. Must be True or False")

def get_metadata_on_the_fly(info, audio, args):
    """
    Maps clean audio files to their corresponding degraded versions using random mode.
    
    Args:
        info (dict): Dictionary containing metadata about the audio file
            info["total_length"] = audio.shape[-1]
            info["path"] = audio_filename
            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask
            info["sample_rate"] = self.sr
        audio (np.ndarray): The clean audio data
        args (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing the degraded audio data
    """
    
    audio = signal.apply_config_to_audio(
        info, 
        audio, 
        args["low_quality_effects_dir"],
    )

    # Return the degraded audio data
    return {
        "degraded_audio": audio
    } 


def get_metadata_from_local(info, audio):
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