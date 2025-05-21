import torchaudio
import torch
import logging
import os
from pathlib import Path
from stable_audio_tools.transforms import signal
from omegaconf import OmegaConf

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


def get_custom_metadata(info, audio, args):
    """
    Maps clean audio files to their corresponding degraded versions.
    
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
        
    Returns:
        dict: Dictionary containing the degraded audio data
    """
    
    # Default values for backward compatibility
    randomize_effects = False
    randomize_intensities = False
    effect_files = []
    
    # Get the effects mode (random or specific)
    effects_mode = args.get("effects_mode", "none")
    
    if effects_mode == "random":
        # Random mode - use random effects
        randomize_effects = True
        randomize_intensities = args.get("random_mode", {}).get("randomize_intensities", True)
        # Use a placeholder effect file that will be ignored since randomize_effects is True
        effect_files = ["random"] 
    elif effects_mode == "specific":
        # Specific mode - use specified effect files
        randomize_effects = False
        specific_config = args.get("specific_mode", {})
        effect_files = specific_config.get("effect_files", [])
        randomize_intensities = specific_config.get("randomize_intensities", False)
    else:
        # For backward compatibility
        effect_files = args.get("low_quality_effect_files", [])
        randomize_effects = args.get("randomize_effects", False)
        randomize_intensities = args.get("randomize_intensities", False)
    
    # Process the effect files if any
    if effect_files and effect_files != [] and effect_files != [""]:
        for effect_file in effect_files:
            LOG.debug(f"Applying effect configuration: {effect_file}")
            audio = signal.apply_config_to_audio(
                info, 
                audio, 
                args["low_quality_effects_dir"], 
                effect_file, 
                randomize_effects, 
                randomize_intensities
            )

    # Return the degraded audio data
    return {
        "degraded_audio": audio
    } 