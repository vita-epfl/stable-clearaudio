import torchaudio
import torch
import logging
import os
from pathlib import Path
from stable_audio_tools.transforms import signal
from omegaconf import OmegaConf

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.DEBUG)


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

    low_quality_effect_files = args["low_quality_effect_files"]

    # Check if we should test a specific effect configuration file
    if low_quality_effect_files and low_quality_effect_files != [] and low_quality_effect_files != [""]:
        for effect_file in low_quality_effect_files:
            LOG.debug(f"Testing with configuration file: {effect_file}")

            audio = signal.apply_config_to_audio(info, audio, args["low_quality_effects_dir"], effect_file)

    # Return the degraded audio data
    return {
        "degraded_audio": audio
    } 