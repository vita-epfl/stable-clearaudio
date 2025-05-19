import torch
import torchaudio
import os
import sys
import logging

# Add the parent directory to the Python search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearaudio.datasets.audio import AudioClip

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

LOG = logging.getLogger(__name__)


def generate_noise_audio(
    noise_type="pink_noise",
    duration=4.0,
    sample_rate=16000,
    gain=1.0,
    filename=None,
    start_freq=1000,
    end_freq=2000,
):
    """
    Generates an audio noise file and saves it locally.

    Args:
        noise_type (str): Type of noise to generate ('pink_noise', 'white_noise', 'brown_noise', 'sine')
        duration (float): Duration of the noise in seconds
        sample_rate (int): Sample rate
        gain (float): Gain to apply to the signal (1.0 = no change, <1.0 = attenuation, >1.0 = amplification)
        output_dir (str, optional): Directory where to save the file
        filename (str, optional): Output filename (without extension)
        start_freq (int, optional): Starting frequency for sinusoidal sound (in Hz)
        end_freq (int, optional): Ending frequency for sinusoidal sound (in Hz). If None, a constant frequency is used.

    Returns:
        AudioClip: The generated audio clip
        str: Path to the saved file (if output_dir is specified)
    """
    LOG.info(f"Generating noise of type {noise_type}")

    # Calculate number of samples
    num_samples = int(duration * sample_rate)
    
    filename = f"{noise_type}_noise_{int(duration)}s"

    # Generate noise
    if noise_type == "white_noise":
        # White noise (uniform distribution)
        waveform = torch.randn(1, num_samples)

    elif noise_type == "pink_noise":
        # Pink noise (simple approximation)
        # Pink noise has a power spectral density inversely proportional to frequency
        LOG.info("Generating pink noise (1/f)")
        white_noise = torch.randn(1, num_samples)

        # Convert to frequency domain
        spec = torch.fft.rfft(white_noise, dim=1)

        # Apply 1/f filter
        freqs = torch.fft.rfftfreq(num_samples, 1 / sample_rate)
        freqs[0] = 1.0  # Avoid division by zero

        # 1/sqrt(f) filter for power spectral density
        filter_shape = 1.0 / torch.sqrt(freqs)

        # Apply filter
        spec_filtered = spec * filter_shape.unsqueeze(0)

        # Back to time domain
        waveform = torch.fft.irfft(spec_filtered, n=num_samples, dim=1)

    elif noise_type == "brown_noise":
        # Brown noise (Brownian motion)
        LOG.info("Generating brown noise (Brownian motion)")
        white_noise = torch.randn(1, num_samples)
        waveform = torch.cumsum(white_noise, dim=1) / sample_rate

    elif noise_type == "sine":
        LOG.info(f"Generating a sine wave from {start_freq}Hz to {end_freq}Hz")

        # Create a time vector
        t = torch.linspace(0, duration, num_samples)

        LOG.debug(f"What we know: {start_freq}Hz, {end_freq}Hz, {duration}s, {num_samples} samples, {sample_rate}Hz")

        # Calculate instantaneous frequency at each time point (linear interpolation)
        freq = start_freq + (end_freq - start_freq) * t / duration
        # Calculate instantaneous phase (integral of frequency)
        phase = 2 * torch.pi * torch.cumsum(freq, dim=0) / sample_rate

        # Generate sine wave
        waveform = torch.sin(phase).unsqueeze(0)

        filename = f"sine_{start_freq}Hz_to_{end_freq}Hz_{int(duration)}s"

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize amplitude
    waveform = waveform / waveform.abs().max()

    # Apply gain
    if gain != 1.0:
        LOG.info(f"Applying gain of {gain}")
        waveform = waveform * gain

        # Ensure signal remains within limits [-1, 1]
        if waveform.abs().max() > 1.0:
            LOG.warning(
                "Gain caused amplitude to exceed maximum. Normalization applied."
            )
            waveform = waveform / waveform.abs().max()

    audio_clip = AudioClip(
        waveform, sample_rate, waveform.shape[0], "", filename, 0, waveform.shape[1]
    )

    LOG.debug(f"Generated audio clip: {audio_clip}")

    return audio_clip


def save_audio_clip(clip, output_dir):
    if output_dir:
        # Convert to Path for better path handling
        try:
            # Ensure the path is absolute
            if not os.path.isabs(output_dir):
                output_dir = os.path.abspath(output_dir)
                LOG.info(f"Converting to absolute path: {output_dir}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            LOG.info(f"Directory created or existing: {os.path.dirname(output_dir)}")

            # Check that directory exists
            if not os.path.exists(os.path.dirname(output_dir)):
                LOG.error(
                    f"Directory doesn't exist after creation: {os.path.dirname(output_dir)}"
                )
            else:
                LOG.info(
                    f"Successful verification: directory exists: {os.path.dirname(output_dir)}"
                )

            # Full path to output file
            LOG.info(f"Saving audio clip to: {output_dir}/{clip.name}.wav")

            # Save audio file
            torchaudio.save(
                output_dir + "/" + clip.name + ".wav", clip.waveform, clip.sample_rate
            )
            LOG.info(f"Noise saved in: {output_dir}")

            # Verify file was created
            if os.path.exists(output_dir):
                LOG.info(f"Successful verification: file exists: {output_dir}")
            else:
                LOG.error(f"File doesn't exist after creation: {output_dir}")
        except Exception as e:
            LOG.error(
                f"Error when creating directory or saving file: {str(e)}"
            )
            import traceback

            LOG.error(traceback.format_exc())


def main():
    noise_config = GenerateNoiseConfig()

    noise_clip = generate_noise_audio(
        noise_config.type,
        noise_config.duration,
        noise_config.sample_rate,
        noise_config.gain,
        noise_config.start_freq,
        noise_config.end_freq,
    )

    # filename = f"{noise_config.type}_noise_{noise_config.duration}s.wav"
    save_audio_clip(noise_clip, noise_config.output_dir)


if __name__ == "__main__":
    # Add command line arguments for new features
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class GenerateNoiseConfig:
        output_dir: str = "/mnt/vita/scratch/datasets/audio_effects/sox_noises"
        type: str = "sine"
        duration: float = 10.0
        sample_rate: int = 16000
        gain: float = 1.0
        start_freq: int = 1000
        end_freq: Optional[int] = 2000

    main()
