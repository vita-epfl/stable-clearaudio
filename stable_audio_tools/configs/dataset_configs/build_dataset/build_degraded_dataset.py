import torch
import torchaudio
import yaml
import os
import sys
import logging
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm

# Add the parent directory to the Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearaudio.transforms import signal
from clearaudio.datasets.audio import AudioClip, load_audio_clip

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

LOG = logging.getLogger(__name__)


def load_yaml_config(file_path):
    """
    Loads a YAML configuration file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def find_audio_files(dataset_path, max_files=10):
    """
    Finds audio files in the dataset.
    """
    LOG.info(f"Finding audio files in {dataset_path}")
    dataset_path = Path(dataset_path)

    # Vérifier si le chemin existe
    if not dataset_path.is_dir():
        LOG.error(f"Dataset path does not exist or is not a directory: {dataset_path}")
        raise FileNotFoundError(
            f"Dataset path does not exist or is not a directory: {dataset_path}"
        )

    LOG.debug("test")

    # Si max_files est None, on veut tous les fichiers
    if max_files is None:
        LOG.debug("Using glob method to find audio files")
        audio_files = list(dataset_path.glob("**/*.wav"))
        LOG.info(f"Found {len(audio_files)} WAV files")
        return audio_files

    LOG.debug("test2")

    # Utiliser os.walk pour trouver les fichiers sans construire toute la liste
    if max_files == 1:
        # Optimisation pour le cas où on ne veut qu'un seul fichier
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav") or file.endswith(".mp3"):
                    file_path = Path(os.path.join(root, file))
                    LOG.info(f"Found audio file: {file_path}")
                    return [file_path]

    LOG.debug("test3")

    # Sinon, utiliser l'approche originale mais limiter la recherche
    audio_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(Path(os.path.join(root, file)))
                if len(audio_files) >= max_files:
                    LOG.info(f"Found {len(audio_files)} WAV files")
                    return audio_files

    LOG.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {dataset_path}")

    return audio_files


def extract_audio_segment(dataset_path, duration=4.0):
    """
    Extracts an audio segment of the specified duration.

    Args:
        audio_path (str): Path to the audio file or directory containing audio files
        duration (float): Duration in seconds of the segment to extract. If -1, uses the entire audio file.

    Returns:
        AudioClip: The extracted audio segment
    """
    try:
        # Check if dataset_path is a file or directory
        if os.path.isfile(dataset_path):
            audio_path = dataset_path
        else:
            LOG.error(f"Invalid dataset path: {dataset_path}")

        audio_clip = load_audio_clip(str(audio_path), mono=True)

        # If duration is -1, return the entire audio file
        if duration == -1:
            LOG.debug(
                f"Using the entire audio file ({audio_clip.waveform.shape[1] / audio_clip.sample_rate:.2f} seconds)"
            )
            return audio_clip

        LOG.debug(
            f"Extracting audio segment of duration {duration} from {audio_clip.name}"
        )

        # Calculate the number of frames for the requested duration
        num_frames = int(duration * audio_clip.sample_rate)

        # Make sure the file is long enough
        if audio_clip.waveform.shape[1] <= num_frames:
            LOG.warning(
                "The audio file is shorter than the requested duration. Using the entire file."
            )
            return audio_clip

        # Take a random segment
        max_offset = audio_clip.waveform.shape[1] - num_frames
        offset = np.random.randint(0, max_offset)

        LOG.debug(f"Selected offset: {offset}")

        # Extract the segment
        segment = audio_clip.waveform[:, offset : offset + num_frames]

        LOG.debug(
            f"Extracted segment of duration {num_frames / audio_clip.sample_rate:.2f} seconds"
        )

        return AudioClip(
            segment,
            audio_clip.sample_rate,
            segment.shape[0],
            audio_clip.path,
            f"{Path(audio_clip.path).stem}_segment",
            0,
            segment.shape[1],
        )
    except Exception as e:
        LOG.error(f"Error when extracting the audio segment: {e}")
        raise


def apply_effects_with_transform(cfg, effect_config_name, audio_clip):
    """
    Applies audio effects to an audio clip using SoxEffectTransform.
    """
    try:
        LOG.debug(f"Applying effects with configuration: {effect_config_name}")
        
        # Extract the configuration for this specific effect
        if not hasattr(cfg.dataset.low_quality_effect, effect_config_name):
            LOG.error(f"Configuration '{effect_config_name}' not found in dataset.low_quality_effect")
            return None
            
        effect_cfg = cfg.dataset.low_quality_effect[effect_config_name]
        LOG.debug(f"Effect configuration: {effect_cfg}")
        
        # Create the config structure that matches SoxEffectTransform.from_config expectation
        config_dict = {"dataset": {"low_quality_effect": {effect_config_name: effect_cfg}}}
        config = OmegaConf.create(config_dict)
        
        # Get transformations from the configuration
        transforms = signal.SoxEffectTransform.from_config(config, effect_config_name)

        if not transforms:
            LOG.error(f"No transformation found for configuration {effect_config_name}")
            return None

        processed_clips = []

        # Apply each transformation
        for transform in transforms:
            LOG.debug(f"Applying transformation: {transform.name}")
            LOG.debug(f"Effects to apply: {transform.effects}")

            # Apply the transformation
            processed_wave, processed_sr = transform.apply_tensor(
                audio_clip.waveform, audio_clip.sample_rate
            )

            # Apply gain if specified in the configuration for this effect and stored in the transform object
            gain_value = transform.original_gain # Get gain from the transform object
            
            #if hasattr(effect_cfg, 'gain'): # Check if gain attribute exists
            #    gain_value = float(effect_cfg.gain)
            #elif isinstance(effect_cfg, dict) and 'gain' in effect_cfg: # Check if gain key exists (for raw dict access)
            #    gain_value = float(effect_cfg['gain'])
            
            if gain_value is not None:
                LOG.debug(f"Applying gain {gain_value} from config '{effect_config_name}' (via transform.original_gain) after SoX effects")
                # Normalize first to prevent clipping
                if processed_wave.abs().max() > 0:
                    processed_wave = processed_wave / processed_wave.abs().max()
                # Apply the gain
                processed_wave = processed_wave * gain_value
                LOG.debug(f"Gain applied. New max amplitude: {processed_wave.abs().max().item()}")

            # Create a new audio clip
            processed_clip = AudioClip(
                processed_wave,
                processed_sr,
                processed_wave.shape[0],
                "",
                f"{audio_clip.name}_{transform.name}",
                0,
                processed_wave.shape[1],
            )

            processed_clips.append(processed_clip)

        return processed_clips
    except Exception as e:
        LOG.error(f"Error when applying effects: {e}")
        raise


def apply_external_sound_to_audio(
    audio_clip, external_sound_config_path
):
    """
    Adds an external sound to an audio clip using a configuration file.

    Args:
        audio_clip (AudioClip): The audio clip to which to add the external sound
        external_sound_config_path (str): Path to the external sound configuration file
        output_dir (str, optional): Directory where to save the result

    Returns:
        AudioClip: The audio clip with the external sound added
    """
    LOG.debug(
        f"Adding an external sound from the configuration: {external_sound_config_path}"
    )

    # Load the external sound configuration
    cfg = OmegaConf.load(external_sound_config_path)

    # Create a configuration dictionary compatible with ExternalSoundTransform
    low_quality_effect_files = Path(external_sound_config_path).stem
    LOG.debug(f"External sound configuration: {low_quality_effect_files}")

    # Create the config structure that matches the new ExternalSoundTransform.from_config expectation
    config_dict = {"dataset": {"low_quality_effect": {low_quality_effect_files: cfg}}}
    config = OmegaConf.create(config_dict)

    # Create the external sound transformation
    transforms = signal.ExternalSoundTransform.from_config(
        config, low_quality_effect_files
    )

    if not transforms:
        LOG.error(
            f"No external sound transformation found in the configuration {external_sound_config_path}"
        )
        return audio_clip

    processed_clips = []

    # Apply each transformation
    for transform in transforms:
        LOG.debug(f"Applying external sound: {transform.sound_name}")

        # Apply the transformation
        processed_wave, processed_sr = transform.apply_external_sound(
            audio_clip.waveform, audio_clip.sample_rate
        )

        # Create a new audio clip
        processed_clip = AudioClip(
            processed_wave,
            processed_sr,
            processed_wave.shape[0],
            "",
            f"{audio_clip.name}_with_{transform.sound_name}",
            0,
            processed_wave.shape[1],
        )

        processed_clips.append(processed_clip)

    output_audio = processed_clips[0] if processed_clips else audio_clip

    LOG.debug(f"Output audio: {output_audio.name}")

    return output_audio


def apply_noise_to_audio(
    audio_clip,
    noise_type="pink",
    duration=4.0,
    sample_rate=16000,
    gain=1.0,
    output_dir=None,
    start_freq=1000,
    end_freq=None,
):
    """
    Generates a noise audio file, applies it to the input audio clip, and returns the result.

    Args:
        audio_clip (AudioClip): The audio clip to which the noise will be added
        noise_type (str): Type of noise to generate ('pink', 'white', 'brown', 'sine')
        duration (float): Duration of the noise in seconds
        sample_rate (int): Sample rate
        gain (float): Gain to apply to the signal (1.0 = no change, <1.0 = attenuation, >1.0 = amplification)
        output_dir (str, optional): Directory where to save the file
        start_freq (int, optional): Starting frequency for the sine wave (in Hz)
        end_freq (int, optional): Ending frequency for the sine wave (in Hz). If None, a constant frequency is used.

    Returns:
        AudioClip: The audio clip with noise added
    """
    LOG.debug(f"Generating noise of type {noise_type}")

    # Calculate the number of samples
    num_samples = int(duration * sample_rate)

    # Generate the noise
    if noise_type == "white":
        # White noise (uniform distribution)
        waveform = torch.randn(1, num_samples)
        LOG.debug("Generating white noise (normal distribution)")
    elif noise_type == "pink":
        # Pink noise (simple approximation)
        # Pink noise has a power spectral density inversely proportional to frequency
        LOG.debug("Generating pink noise (1/f)")
        white_noise = torch.randn(1, num_samples)

        # Convert to frequency domain
        spec = torch.fft.rfft(white_noise, dim=1)

        # Apply 1/f filter
        freqs = torch.fft.rfftfreq(num_samples, 1 / sample_rate)
        freqs[0] = 1.0  # Avoid division by zero

        # 1/sqrt(f) filter for power spectral density
        filter_shape = 1.0 / torch.sqrt(freqs)

        # Apply the filter
        spec_filtered = spec * filter_shape.unsqueeze(0)

        # Return to time domain
        waveform = torch.fft.irfft(spec_filtered, n=num_samples, dim=1)

        # Verify that the generated noise is correct
        LOG.debug(
            f"Generated waveform: min={waveform.min().item()}, max={waveform.max().item()}, mean={waveform.mean().item()}"
        )
    elif noise_type == "brown":
        # Brown noise (Brownian motion)
        LOG.debug("Generating brown noise (Brownian motion)")
        white_noise = torch.randn(1, num_samples)
        waveform = torch.cumsum(white_noise, dim=1) / sample_rate
    elif noise_type == "sine":
        if output_dir == "":
            output_dir = "tests/test_sox_output/external_sound"
        # Generating a sine wave
        if end_freq is None:
            end_freq = start_freq

        LOG.debug(f"Generating sine wave from {start_freq}Hz to {end_freq}Hz")

        # Create a time vector
        t = torch.linspace(0, duration, num_samples)

        # Calculate instantaneous frequency at each time point (linear interpolation)
        if start_freq != end_freq:
            freq = start_freq + (end_freq - start_freq) * t / duration
            # Calculate instantaneous phase (integral of frequency)
            phase = 2 * torch.pi * torch.cumsum(freq, dim=0) / sample_rate
        else:
            # Constant frequency
            phase = 2 * torch.pi * start_freq * t

        # Generate the sine wave
        waveform = torch.sin(phase).unsqueeze(0)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize amplitude
    waveform = waveform / waveform.abs().max()

    # Apply gain
    if gain != 1.0:
        LOG.debug(f"Applying gain of {gain}")
        waveform = waveform * gain

        # Ensure signal stays within [-1, 1] limits
        if waveform.abs().max() > 1.0:
            LOG.warning(
                "Gain caused amplitude to exceed maximum. Normalization applied."
            )
            waveform = waveform / waveform.abs().max()

    # Add the noise to the audio clip
    if audio_clip.waveform.shape[1] < waveform.shape[1]:
        LOG.warning("Audio clip is shorter than the noise. Truncating noise.")
        waveform = waveform[:, : audio_clip.waveform.shape[1]]
    elif audio_clip.waveform.shape[1] > waveform.shape[1]:
        LOG.warning("Noise is shorter than the audio clip. Padding with zeros.")
        padded_waveform = torch.zeros_like(audio_clip.waveform)
        padded_waveform[:, : waveform.shape[1]] = waveform
        waveform = padded_waveform

    # Mix the audio and noise
    mixed_waveform = audio_clip.waveform + waveform

    # Normalize if needed to prevent clipping
    if mixed_waveform.abs().max() > 1.0:
        LOG.warning("Mixing caused amplitude to exceed maximum. Normalization applied.")
        mixed_waveform = mixed_waveform / mixed_waveform.abs().max()

    # Update the audio clip with the mixed waveform
    audio_clip.waveform = mixed_waveform
    audio_clip.name = f"{audio_clip.name}_{noise_type}_noise"

    # Save the mixed audio if output directory is specified
    if output_dir:
        output_path = os.path.join(output_dir, f"{audio_clip.name}.wav")
        torchaudio.save(output_path, audio_clip.waveform, audio_clip.sample_rate)
        LOG.debug(f"Mixed audio saved to {output_path}")

    return audio_clip


def apply_config_to_audio(audio_clip, low_quality_effects_dir, effects_file=None):
    """
    Retrieves an audio from the dataset, applies the effects from the configuration, and adds the specified external audio.

    Args:
        audio_clip (AudioClip): The audio clip to which the effects will be applied
        output_dir (str, optional): Directory where to save the file
        effects_config_path (str): Path to the effects configuration file
        duration (float, optional): Duration in seconds to extract from the audio file

    Returns:
        AudioClip: The processed audio clip
    """
    # path is "/home/alefevre/programs/clearaudio/clearaudio/conf/dataset/low_quality_effect" + effects_file + .yaml
    effects_config_path = os.path.join(
        low_quality_effects_dir,
        effects_file + ".yaml",
    )
    LOG.debug(f"Applying effects from {effects_config_path} to audio {audio_clip.name}")

    # Check if the configuration file exists
    if not effects_config_path or not Path(effects_config_path).exists():
        LOG.error(f"Configuration file not found: {effects_config_path}")
        return audio_clip

    # Load the configuration
    try:
        config_raw = load_yaml_config(effects_config_path)
        LOG.debug(f"Configuration loaded: {effects_config_path}")
        
        # Create a configuration dictionary compatible with SoxEffectTransform
        low_quality_effect_files = Path(effects_config_path).stem
        LOG.debug(f"Effect configuration: {low_quality_effect_files}")

        # Create the config structure that matches SoxEffectTransform.from_config expectation
        config_dict = {"dataset": {"low_quality_effect": {low_quality_effect_files: config_raw}}}
        config = OmegaConf.create(config_dict)
        
        # Apply SoX effects using SoxEffectTransform
        transforms = signal.SoxEffectTransform.from_config(config, low_quality_effect_files)
        
        if transforms:
            # Apply each transformation
            for transform in transforms:
                LOG.debug(f"Applying SoX transformation: {transform.name}")
                LOG.debug(f"Effects to apply: {transform.effects}")

                # Apply the transformation
                processed_wave, processed_sr = transform.apply_tensor(
                    audio_clip.waveform, audio_clip.sample_rate
                )
                
                # Appliquer le gain s'il est spécifié dans l'effet
                if "effects" in config_raw:
                    for effect in config_raw["effects"]:
                        if "gain" in effect:
                            gain_value = float(effect["gain"])
                            LOG.debug(f"Applying gain after SOX effects: {gain_value}")
                            # Normaliser d'abord la waveform
                            if processed_wave.abs().max() > 0:
                                processed_wave = processed_wave / processed_wave.abs().max()
                            # Puis appliquer le gain
                            processed_wave = processed_wave * gain_value
                            LOG.debug(f"Applied gain. New max amplitude: {processed_wave.abs().max().item()}")
                
                # Update the audio clip
                audio_clip.waveform = processed_wave
                audio_clip.sample_rate = processed_sr
                audio_clip.num_samples = processed_wave.shape[1]
                
                LOG.debug(f"SoX effects applied successfully. New shape: {processed_wave.shape}")
        else:
            LOG.warning(f"No SoX effect transformations found in configuration {effects_config_path}")
            
        # Process external sounds if any
        if "external_sounds" in config_raw:
            LOG.debug("External sounds found in configuration. Adding...")

            # Create external sound transformations
            transforms = signal.ExternalSoundTransform.from_config(
                config, effects_file
            )

            if transforms:
                # Apply each transformation
                for transform in transforms:
                    LOG.debug(f"Applying external sound: {transform.sound_name}")

                    # Apply the transformation
                    processed_wave, processed_sr = transform.apply_external_sound(
                        audio_clip.waveform, audio_clip.sample_rate
                    )

                    # Update the audio clip
                    audio_clip.waveform = processed_wave
                    audio_clip.sample_rate = processed_sr
                    audio_clip.num_samples = processed_wave.shape[1]

                    LOG.debug(f"External sound added successfully: {transform.sound_name}")
            else:
                LOG.error(
                    f"No external sound transformation found in configuration {effects_config_path}"
                )
    except Exception as e:
        LOG.error(f"Error when applying configuration: {str(e)}")
        import traceback
        LOG.error(traceback.format_exc())
        return audio_clip

    return audio_clip


def main():
    """
    Main function for testing SoX effects.
    """
    LOG.info("Starting build_degraded_dataset...")
    # Use the TestConfig directly instead of relying on Hydra's test group
    test_config = BuildDatasetConfig()

    # Get configuration parameters
    dataset_path = test_config.dataset_path
    degraded_output_dir = test_config.degraded_output_dir
    clean_output_dir = test_config.clean_output_dir
    noise_output_dir = test_config.sox_noises_dir
    low_quality_effects_dir = test_config.low_quality_effects_dir

    build_clean_dataset = test_config.build_clean_dataset

    duration = test_config.duration

    num_files = test_config.num_files
    low_quality_effect_files = test_config.low_quality_effect_files

    # Parameters for noise generation
    noises_to_apply = test_config.noises_to_apply
    noise_gain = test_config.noise_gain

    # List of external sound configuration files to apply
    external_sounds = test_config.external_sounds

    start_freq = test_config.start_freq
    end_freq = test_config.end_freq

    # Verify that the dataset is specified
    if not dataset_path or dataset_path == "":
        LOG.error("Dataset path is not specified")
        return

    # If num_files is -1, process all files in the dataset
    max_files = None if num_files == -1 else num_files
    audio_files = find_audio_files(dataset_path, max_files=max_files)

    if not audio_files:
        LOG.error(f"No audio files found in {dataset_path}")
        return

    if degraded_output_dir == "":
        degraded_output_dir = "tests/test_sox_output/external_sound"

    # Create output directory if it doesn't exist
    if not Path(degraded_output_dir).exists():
        os.makedirs(degraded_output_dir, exist_ok=True)
        LOG.info(f"Output directory created: {degraded_output_dir}")

    if build_clean_dataset:
        if clean_output_dir == "":
            clean_output_dir = "tests/test_sox_output/clean"

        # Create clean output directory if it doesn't exist
        if not Path(clean_output_dir).exists():
            os.makedirs(clean_output_dir, exist_ok=True)
            LOG.info(f"Output directory created: {clean_output_dir}")

    if noise_output_dir == "":
        noise_output_dir = "tests/test_sox_output/noise"

    # Create output directory if it doesn't exist
    if not Path(noise_output_dir).exists():
        os.makedirs(noise_output_dir, exist_ok=True)
        LOG.info(f"Output directory created: {noise_output_dir}")

    LOG.info("Initializing processing...")

    # Use tqdm to display a progress bar
    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        LOG.debug(f"Processing file: {audio_file}")
        # Extract an audio segment
        audio_clip = extract_audio_segment(audio_file, duration=duration)
        # we copy the audio clip to use it for the clean dataset
        clean_audio_clip = AudioClip(
            audio_clip.waveform.clone(),
            audio_clip.sample_rate,
            audio_clip.num_channels,
            audio_clip.path,
            audio_clip.name,
            audio_clip.frame_offset,
            audio_clip.num_frames
        )

        # Generate noise if requested
        if noises_to_apply and noises_to_apply != [] and noises_to_apply != [""]:
            LOG.debug(f"Generating noise of type: {', '.join(noises_to_apply)}")
            # Convert to list if it's a single string
            if isinstance(noises_to_apply, str):
                noises_to_apply = [noises_to_apply]

            noise_output_dir = os.path.join(noise_output_dir, "noise")

            # Create output directory if it doesn't exist
            if not Path(noise_output_dir).exists():
                os.makedirs(noise_output_dir, exist_ok=True)

            LOG.debug(f"Generating {len(noises_to_apply)} type(s) of noise")

            # Apply each type of noise
            for noise_type in tqdm(
                noises_to_apply, desc="Generating noises", unit="noise"
            ):
                if noise_type:  # Skip empty noise types
                    LOG.debug(f"Generating noise of type: {noise_type}")
                    audio_clip = apply_noise_to_audio(
                        audio_clip,
                        noise_type=noise_type,
                        duration=duration,
                        gain=noise_gain,
                        output_dir=noise_output_dir,
                        start_freq=start_freq,
                        end_freq=end_freq,
                    )

            LOG.debug(
                f"Noise generation completed. Generated noises: {', '.join(noises_to_apply)}"
            )

        # Check if we should apply external sound effect to an audio file
        if external_sounds and external_sounds != [] and external_sounds != [""]:
            for external_sound in external_sounds:
                LOG.debug(f"Applying sound from {external_sound} to a dataset audio")

                # Apply sound to the base audio
                audio_clip = apply_external_sound_to_audio(
                    audio_clip, external_sound
                )

        # Check if we should test a specific effect configuration file
        if low_quality_effect_files and low_quality_effect_files != [] and low_quality_effect_files != [""]:
            for effect_file in low_quality_effect_files:
                LOG.debug(f"Testing with configuration file: {effect_file}")

                audio_clip = apply_config_to_audio(audio_clip, low_quality_effects_dir, effect_file)

        # Ensure the path is absolute
        if not Path(degraded_output_dir).is_absolute():
            degraded_output_dir = os.path.abspath(degraded_output_dir)
            LOG.debug(f"Converting to absolute path: {degraded_output_dir}")

        # Create directory if it doesn't exist
        os.makedirs(degraded_output_dir, exist_ok=True)
        LOG.debug(f"Directory created or already exists: {degraded_output_dir}")

        # Full path to output file
        deagraded_output_path = Path(degraded_output_dir) / f"{audio_clip.name}.wav"
        LOG.debug(f"Output file path: {deagraded_output_path}")

        # Save the audio file
        torchaudio.save(str(deagraded_output_path), audio_clip.waveform, audio_clip.sample_rate)
        LOG.debug(f"Audio saved to: {deagraded_output_path}")

        # Verify that the directory exists
        if not Path(degraded_output_dir).exists():
            LOG.error(f"Directory doesn't exist after creation: {degraded_output_dir}")
        else:
            LOG.debug(f"Verification successful: directory exists: {degraded_output_dir}")

        # Verify that the file was created
        if not Path(deagraded_output_path).exists():
            LOG.error(f"File doesn't exist after creation: {deagraded_output_path}")
        else:
            LOG.debug(f"Verification successful: file exists: {deagraded_output_path}")

        if build_clean_dataset:
            if not Path(clean_output_dir).exists():
                os.makedirs(clean_output_dir, exist_ok=True)
                LOG.debug(f"Directory created or already exists: {clean_output_dir}")

            # Full path to output file
            clean_output_path = Path(clean_output_dir) / f"{audio_clip.name}.wav"
            LOG.debug(f"Output file path: {clean_output_path}")

            # Save the audio file
            torchaudio.save(str(clean_output_path), clean_audio_clip.waveform, clean_audio_clip.sample_rate)
            LOG.debug(f"Audio saved to: {clean_output_path}")

            # Verify that the file was created
            if not Path(clean_output_path).exists():
                LOG.error(f"File doesn't exist after creation: {clean_output_path}")
            else:
                LOG.debug(f"Verification successful: file exists: {clean_output_path}") 

    LOG.debug("Test completed.")


if __name__ == "__main__":
    # Add command line arguments for new features
    from dataclasses import dataclass, field
    from typing import Optional, List

    @dataclass
    class BuildDatasetConfig:
        dataset_path: str = "/mnt/vita/scratch/datasets/maestro-v3.0.0/maestro_full"
        degraded_output_dir: str = "/mnt/vita/scratch/datasets/maestro_short/degraded" # "/home/alefevre/datasets/maestro-v3.0.0/degraded"
        clean_output_dir: str = "/mnt/vita/scratch/datasets/maestro_short/clean" # "/home/alefevre/datasets/maestro-v3.0.0/clean"
        sox_noises_dir: str = "/mnt/vita/scratch/datasets/audio_effects/sox_noises"
        low_quality_effects_dir: str = "/mnt/vita/scratch/vita-staff/users/alefevre/programs/clearaudio/clearaudio/conf/dataset/low_quality_effect"


        build_clean_dataset: bool = True

        duration: float = 10  # Duration in seconds. If -1, uses the entire audio file

        num_files: int = -1  # Use -1 to process all files in the dataset
        low_quality_effect_files: List[str] = field(default_factory=lambda: ["strong_mp3_compression"])

        # List of noise types to generate ('pink', 'white', 'brown', 'sine')
        noises_to_apply: List[str] = field(default_factory=lambda: [""])
        noise_gain: float = 1.0

        # List of external sound configuration files to apply
        external_sounds: List[str] = field(default_factory=list)

        start_freq: int = 1000
        end_freq: Optional[int] = None

    main()

# cd programs/clearaudio; source venv/bin/activate; python tests/test_sox_effects.py
