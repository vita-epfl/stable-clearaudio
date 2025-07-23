import os
import sys
import logging
import argparse
import json
import random
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

# Disable matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
    noise_gain: float = 1.0
    generate_visualizations: bool = True
    

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

    # print nb of files in dataset_path
    LOG.info("Number of files in %s: %d" % (dataset_path, len(list(dataset_path.glob("**/*")))))
    
    if dataset_path.is_file() and dataset_path.suffix.lower() in [".wav", ".mp3", ".flac"]:
        # If dataset_path is a single audio file
        return [dataset_path]
    
    # If dataset_path is a directory, find all audio files
    audio_files = []
    for ext in [".wav", ".mp3", ".flac"]:
        audio_files.extend(list(dataset_path.glob(f"**/*{ext}")))
    
    LOG.info(f"Found {len(audio_files)} audio files in {dataset_path}")
    
    # Limit the number of files if specified
    if num_files > 0 and num_files < len(audio_files):
        # Get random files from the folder
        audio_files = random.sample(audio_files, num_files)
        LOG.info(f"Limited to {len(audio_files)} audio files (randomly selected)")

    # Sort files to ensure reproducibility
    audio_files.sort()
    
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
        output_filename = f"{base_filename}_degraded_{'_'.join(config.degradation_presets)}.wav"
        
        # Save degraded audio
        degraded_path = Path(config.degraded_output_dir) / output_filename
        torchaudio.save(degraded_path, degraded_audio, sr)
        LOG.debug(f"Saved degraded audio to {degraded_path}")

        if config.generate_visualizations:
            # Generate frequency visualization for degraded audio
            degraded_viz_path = degraded_path.parent / f"{base_filename}_degraded_freq_analysis.png"
            generate_frequency_visualization(degraded_audio, sr, degraded_viz_path)
            LOG.debug(f"Saved degraded frequency visualization to {degraded_viz_path}")
        
        # Save clean audio if requested
        if config.build_clean_dataset:
            clean_path = Path(config.clean_output_dir) / f"{base_filename}_clean.wav"
            torchaudio.save(clean_path, audio, sr)
            LOG.debug(f"Saved clean audio to {clean_path}")

            if config.generate_visualizations:
                # Generate frequency visualization for clean audio
                clean_viz_path = clean_path.parent / f"{base_filename}_freq_analysis.png"
                generate_frequency_visualization(audio, sr, clean_viz_path)
                LOG.debug(f"Saved clean frequency visualization to {clean_viz_path}")
                
                # Generate comparative frequency visualization between clean and degraded audio
                comparison_path = degraded_path.parent / f"{base_filename}_freq_comparison.png"
                generate_comparative_visualization(audio, degraded_audio, sr, comparison_path)
                LOG.debug(f"Saved comparative frequency visualization to {comparison_path}")


def generate_frequency_visualization(audio: torch.Tensor, sr: int, output_path: Path):
    """Generate and save a frequency visualization of the audio
    
    Args:
        audio: Audio tensor (1, n_samples)
        sr: Sample rate
        output_path: Path to save the visualization
    """
    try:
        # Convert to numpy array and flatten if needed
        audio_np = audio.numpy().flatten()
        
        plt.figure(figsize=(10, 8))
        
        # Create subplot layout
        plt.subplot(2, 1, 1)
        
        # Create spectrum
        D = np.abs(np.fft.rfft(audio_np))
        # Convert to dB scale
        D_db = 20 * np.log10(D + 1e-10)
        
        # Frequency axis
        freqs = np.linspace(0, sr/2, len(D))
        
        # Plot spectrum
        plt.plot(freqs, D_db)
        plt.title('Frequency Spectrum Analysis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        
        # Add more frequency ticks for the linear scale with better spacing
        linear_ticks = [0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 22000]
        # Filter out frequencies above Nyquist
        linear_ticks = [f for f in linear_ticks if f < sr/2]
        plt.xticks(linear_ticks, [str(f) for f in linear_ticks], fontsize=8)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
        
        # Add log-scale version for better visualization of lower frequencies
        plt.subplot(2, 1, 2)
        plt.semilogx(freqs, D_db)
        plt.title('Frequency Spectrum (Log Scale)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([20, sr/2])  # Audible range starts around 20Hz
        
        # Add vertical lines at standard octave frequencies for reference
        octave_freqs = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in octave_freqs:
            if freq < sr/2:  # Only show frequencies below Nyquist
                plt.axvline(x=freq, color='r', linestyle='--', alpha=0.3)
                plt.annotate(f"{freq}Hz", (freq, np.min(D_db)), rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
    except Exception as e:
        LOG.error(f"Error generating frequency visualization: {e}")


def generate_comparative_visualization(clean_audio: torch.Tensor, degraded_audio: torch.Tensor, sr: int, output_path: Path):
    """Generate and save a comparative frequency visualization of clean and degraded audio
    
    Args:
        clean_audio: Clean audio tensor (1, n_samples)
        degraded_audio: Degraded audio tensor (1, n_samples)
        sr: Sample rate
        output_path: Path to save the visualization
    """
    try:
        # Convert to numpy arrays and flatten if needed
        clean_np = clean_audio.numpy().flatten()
        degraded_np = degraded_audio.numpy().flatten()
        
        # Create figure with 2 rows, 2 columns layout
        plt.figure(figsize=(15, 10))
        
        # Calculate FFT for both signals
        clean_D = np.abs(np.fft.rfft(clean_np))
        degraded_D = np.abs(np.fft.rfft(degraded_np))
        
        # Convert to dB scale
        clean_db = 20 * np.log10(clean_D + 1e-10)
        degraded_db = 20 * np.log10(degraded_D + 1e-10)
        
        # Frequency axis
        freqs = np.linspace(0, sr/2, len(clean_db))
        
        # Plot linear scale
        plt.subplot(2, 2, 1)
        plt.plot(freqs, clean_db, label='Clean', alpha=0.7, color='green')
        plt.title('Clean Audio - Linear Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        
        # Add more frequency ticks for the linear scale with better spacing
        linear_ticks = [0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 22000]
        # Filter out frequencies above Nyquist
        linear_ticks = [f for f in linear_ticks if f < sr/2]
        plt.xticks(linear_ticks, [str(f) for f in linear_ticks], fontsize=8)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
        
        plt.subplot(2, 2, 2)
        plt.plot(freqs, degraded_db, label='Degraded', alpha=0.7, color='red')
        plt.title('Degraded Audio - Linear Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        
        # Add more frequency ticks for the linear scale with better spacing
        linear_ticks = [0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 22000]
        # Filter out frequencies above Nyquist
        linear_ticks = [f for f in linear_ticks if f < sr/2]
        plt.xticks(linear_ticks, [str(f) for f in linear_ticks], fontsize=8)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
        
        # Plot log scale
        plt.subplot(2, 2, 3)
        plt.semilogx(freqs, clean_db, label='Clean', alpha=0.7, color='green')
        plt.title('Clean Audio - Log Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([20, sr/2])  # Audible range
        
        # Add vertical lines at standard octave frequencies for reference
        octave_freqs = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in octave_freqs:
            if freq < sr/2:  # Only show frequencies below Nyquist
                plt.axvline(x=freq, color='black', linestyle='--', alpha=0.2)
        
        plt.subplot(2, 2, 4)
        plt.semilogx(freqs, degraded_db, label='Degraded', alpha=0.7, color='red')
        plt.title('Degraded Audio - Log Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.xlim([20, sr/2])  # Audible range
        
        # Add octave frequency reference lines
        for freq in octave_freqs:
            if freq < sr/2:  # Only show frequencies below Nyquist
                plt.axvline(x=freq, color='black', linestyle='--', alpha=0.2)
                plt.annotate(f"{freq}Hz", (freq, np.min(degraded_db)), rotation=45, fontsize=8)
        
        plt.tight_layout()
        
        # Add an extra subplot that shows the difference between clean and degraded
        plt.figure(figsize=(15, 5))
        
        # Calculate difference
        diff_db = degraded_db - clean_db
        
        # Linear scale difference
        plt.subplot(1, 2, 1)
        plt.plot(freqs, diff_db, color='purple')
        plt.title('Frequency Difference (Degraded - Clean)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude Difference (dB)')
        plt.grid(True)
        
        # Add more frequency ticks for the linear scale with better spacing
        linear_ticks = [0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 22000]
        # Filter out frequencies above Nyquist
        linear_ticks = [f for f in linear_ticks if f < sr/2]
        plt.xticks(linear_ticks, [str(f) for f in linear_ticks], fontsize=8)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
        
        # Log scale difference
        plt.subplot(1, 2, 2)
        plt.semilogx(freqs, diff_db, color='purple')
        plt.title('Frequency Difference - Log Scale')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude Difference (dB)')
        plt.grid(True)
        plt.xlim([20, sr/2])
        
        # Add a horizontal line at y=0 to show where no difference occurs
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add octave reference lines
        for freq in octave_freqs:
            if freq < sr/2:
                plt.axvline(x=freq, color='black', linestyle='--', alpha=0.2)
                plt.annotate(f"{freq}Hz", (freq, np.min(diff_db)), rotation=45, fontsize=8)
        
        plt.tight_layout()
        
        # Save all figures to a single file
        plt.savefig(output_path, dpi=150)
        plt.close('all')
        
    except Exception as e:
        LOG.error(f"Error generating comparative frequency visualization: {e}")



def main():
    """Main function for building degraded audio dataset."""
    parser = argparse.ArgumentParser(description="Build dataset for audio restoration")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # If a configuration file is provided, load it
    if args.config:
        cfg_file = Path(args.config)
    else:
        # Use the default JSON configuration file in the same directory
        cfg_file = Path(__file__).parent / "build_degraded_config_workstation.json"
    
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
