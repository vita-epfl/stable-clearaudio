import torch
import logging

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

def get_custom_metadata(info, audio, custom_metadata_args=None):
    """
    Returns a silent degraded audio tensor for cold diffusion models.
    This avoids loading and processing the actual degraded audio file.
    
    Args:
        info (dict): Dictionary containing metadata about the audio file.
        audio (torch.Tensor): The clean audio data.
        custom_metadata_args (dict): Optional custom metadata arguments (not used).
        
    Returns:
        dict: Dictionary containing the silent degraded audio data.
    """
    LOG.debug(f"Generating silent degraded audio for cold diffusion. Clean audio shape: {audio.shape}")
    
    # Create a tensor of zeros with the same shape as the clean audio
    degraded_audio = torch.zeros_like(audio)
    
    return {
        "degraded_audio": degraded_audio
    }
