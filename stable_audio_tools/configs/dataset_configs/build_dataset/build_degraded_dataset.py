import os
import sys
import logging
import argparse
import json
import random
import torchaudio
import torch
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

LOG = logging.getLogger(__name__)

# Ajout du chemin racine du projet pour pouvoir importer les modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(project_root)
LOG.info(f"Added to path: {project_root}")

# Importation des modules spécifiques au projet
from stable_audio_tools.transforms import signal
from stable_audio_tools.configs.dataset_configs.custom_metadata.audio_restoration_md import get_metadata_on_the_fly


@dataclass
class BuildDatasetConfig:
    """Configuration for building degraded audio dataset"""
    dataset_path: str
    degraded_output_dir: str
    clean_output_dir: str
    sox_noises_dir: str
    low_quality_effects_dir: str
    build_clean_dataset: bool
    duration: float
    num_files: int
    degradation_presets: List[str] = field(default_factory=list)
    effects_mode: str = "specific"
    noise_gain: float = 1.0
    external_sounds: List[str] = field(default_factory=list)
    start_freq: int = 1000
    end_freq: Optional[int] = None
    

def load_config(config_path: str) -> BuildDatasetConfig:
    """Load configuration from JSON file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        BuildDatasetConfig: Configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return BuildDatasetConfig(**config_dict)


def get_audio_files(dataset_path: str, num_files: int = -1) -> List[Path]:
    """Get list of audio files from the dataset directory
    
    Args:
        dataset_path: Path to the dataset directory
        num_files: Number of files to process, -1 for all files
        
    Returns:
        List of paths to audio files
    """
    dataset_path = Path(dataset_path)
    
    if dataset_path.is_file() and dataset_path.suffix.lower() in [".wav", ".mp3", ".flac"]:
        # If dataset_path is a single audio file
        return [dataset_path]
    
    # If dataset_path is a directory, find all audio files
    audio_files = []
    for ext in [".wav", ".mp3", ".flac"]:
        audio_files.extend(list(dataset_path.glob(f"**/*{ext}")))
    
    LOG.info(f"Found {len(audio_files)} audio files in {dataset_path}")
    
    # Sort files to ensure reproducibility
    audio_files.sort()
    
    # Limit the number of files if specified
    if num_files > 0 and num_files < len(audio_files):
        audio_files = audio_files[:num_files]
        LOG.info(f"Limited to {len(audio_files)} audio files")
    
    return audio_files


def process_audio_file(file_path: Path, config: BuildDatasetConfig) -> Tuple[torch.Tensor, int]:
    """Load and trim audio file
    
    Args:
        file_path: Path to the audio file
        config: Configuration object
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    try:
        # Load audio file
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Calculate duration in samples
        duration_samples = int(config.duration * sr)
        
        # Trim audio if longer than specified duration
        if audio.shape[1] > duration_samples:
            # Random starting point if we want to extract a segment
            max_start = audio.shape[1] - duration_samples
            start = random.randint(0, max_start)
            audio = audio[:, start:start+duration_samples]
        
        return audio, sr
    
    except Exception as e:
        LOG.error(f"Error processing file {file_path}: {e}")
        return None, None


def build_degraded_dataset(config: BuildDatasetConfig):
    """Build dataset with degraded audio
    
    Args:
        config: Configuration object
    """
    # Create output directories if they don't exist
    Path(config.degraded_output_dir).mkdir(parents=True, exist_ok=True)
    if config.build_clean_dataset:
        Path(config.clean_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of audio files to process
    audio_files = get_audio_files(config.dataset_path, config.num_files)
    
    # Process each audio file
    for file_idx, file_path in enumerate(audio_files):
        LOG.info(f"Processing file {file_idx+1}/{len(audio_files)}: {file_path.name}")
        
        # Load and trim audio
        audio, sr = process_audio_file(file_path, config)
        if audio is None:
            continue
        
        # Prepare info dictionary for metadata function
        info = {
            "path": str(file_path),
            "total_length": audio.shape[1],
            "timestamps": (0, audio.shape[1]),
            "seconds_start": 0,
            "seconds_total": audio.shape[1] / sr,
            "padding_mask": None,
            "sample_rate": sr
        }
        
        # Prepare build_degraded_args dictionary
        build_degraded_args = {
            "degradation_presets": config.degradation_presets,
            "low_quality_effects_dir": config.low_quality_effects_dir,
            "effects_mode": config.effects_mode,
            "noise_gain": config.noise_gain,
            "external_sounds": config.external_sounds,
            "start_freq": config.start_freq,
            "end_freq": config.end_freq
        }

        LOG.debug(f"Building degraded audio with {build_degraded_args}")
        
        # Get degraded audio using get_metadata_on_the_fly
        degraded_result = get_metadata_on_the_fly(info, audio, build_degraded_args)
        
        if not degraded_result or "degraded_audio" not in degraded_result:
            LOG.error(f"Failed to create degraded version for {file_path.name}")
            continue
        
        degraded_audio = degraded_result["degraded_audio"]
        
        # Generate output filenames
        base_filename = file_path.stem
        output_filename = f"{base_filename}_degraded.wav"
        
        # Save degraded audio
        degraded_path = Path(config.degraded_output_dir) / output_filename
        torchaudio.save(degraded_path, degraded_audio, sr)
        LOG.info(f"Saved degraded audio to {degraded_path}")
        
        # Save clean audio if requested
        if config.build_clean_dataset:
            clean_path = Path(config.clean_output_dir) / f"{base_filename}_clean.wav"
            torchaudio.save(clean_path, audio, sr)
            LOG.info(f"Saved clean audio to {clean_path}")


def main():
    """Main function for building degraded audio dataset."""
    parser = argparse.ArgumentParser(description="Build dataset for audio restoration")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # If a configuration file is provided, load it
    if args.config:
        cfg_file = args.config
    else:
        # Use the default JSON configuration file in the same directory
        script_dir = Path(__file__).parent
        cfg_file = script_dir / "build_degraded_config_workstation.json"
    
    # Ensure the config file exists
    if not Path(cfg_file).exists():
        LOG.error(f"Configuration file not found: {cfg_file}")
        return
    
    # Load configuration
    LOG.info(f"Loading configuration from {cfg_file}")
    config = load_config(cfg_file)
    
    # Build the dataset
    build_degraded_dataset(config)


if __name__ == "__main__":
    main()
