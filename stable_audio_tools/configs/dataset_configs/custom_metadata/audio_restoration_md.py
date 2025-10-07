import torchaudio
import torch
import logging
import os
import random
from pathlib import Path
from stable_audio_tools.transforms import signal

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

def get_custom_metadata(info, audio, custom_metadata_args=None):
    build_degraded = custom_metadata_args.get("build_degraded", False)
    build_degraded_args = custom_metadata_args.get("build_degraded_args", False)
    if build_degraded:
        return get_metadata_on_the_fly(info, audio, build_degraded_args)
    elif not build_degraded:
        return get_metadata_from_local(info, audio)
    else:
        raise ValueError("Invalid build_degraded value. Must be True or False")

def get_metadata_on_the_fly(info, audio, build_degraded_args):
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

    degradation_preset_names = build_degraded_args.get("degradation_presets", None)
    low_quality_effects_dir = build_degraded_args.get("low_quality_effects_dir", None)

    # Get the absolute path of the low quality effects directory
    if low_quality_effects_dir.startswith("/") and not os.path.exists(low_quality_effects_dir):
        # Obtenir le chemin de base du projet
        base_path = Path(__file__).resolve().parent.parent.parent.parent  # One more parent to avoid configs duplication
        
        if low_quality_effects_dir.startswith("/stable-clearaudio/"):
            relative_path = low_quality_effects_dir[len("/stable-clearaudio/"):]
            low_quality_effects_dir = str(base_path / relative_path)
        elif low_quality_effects_dir.startswith("/stable_audio_tools/"):
            relative_path = low_quality_effects_dir[len("/stable_audio_tools/"):]
            low_quality_effects_dir = str(base_path / relative_path)  # Don't add 'stable_audio_tools' again


    preset_dir = os.path.join(low_quality_effects_dir)
    degradation_info = []

    for preset_name in degradation_preset_names:
        # Check if preset_name exists in low_quality_effects_dir
        preset_files = [os.path.splitext(f)[0] for f in os.listdir(preset_dir)]
        if preset_name not in preset_files:
            LOG.error(f"Configuration '{preset_name}' not found in {preset_dir}")
            return []
            
        # Construire le chemin final
        preset_path = os.path.join(
            preset_dir,
            preset_name + ".yaml",
        )

        result = signal.apply_config_to_audio(
            info, 
            audio, 
            preset_path
        )

        audio = result["audio"]
        degradation_info.extend(result["degradation_info"])

    # Return the degraded audio data
    return {
        "degraded_audio": audio,
        "degradation_info": degradation_info
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