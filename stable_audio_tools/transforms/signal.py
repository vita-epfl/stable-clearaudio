from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from pathlib import Path
import numpy as np

import torchaudio
import torchaudio.sox_effects as sox
import torchaudio.transforms as T
from torch import Tensor, nn
import torch
import yaml

LOG = logging.getLogger(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    force=True)


def resample_signal(waveform: Tensor, sample_rate: int, resample_rate: int) -> Tuple[Tensor, int]:
    if sample_rate == resample_rate:
        return waveform, sample_rate
    
    ### kaiser_best
    resampler = T.Resample(
        sample_rate,
        resample_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    resampled_waveform = resampler.forward(waveform)
    return resampled_waveform, resample_rate

def signal_to_mono(waveform: Tensor) -> Tensor:
    if waveform is None:
        return None
    if waveform.shape[0] != 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def load_yaml_config(file_path):
    """
    Loads a YAML configuration file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def apply_config_to_audio(info, audio, preset_path):
        """
        Retrieves an audio from the dataset, applies the effects from the configuration, and adds the specified external audio.

        Args:
            audio (Tensor): The audio to which the effects will be applied
            output_dir (str, optional): Directory where to save the file
            effects_config_path (str): Path to the effects configuration file
            duration (float, optional): Duration in seconds to extract from the audio file

        Returns:
            Tensor: The processed audio
        """

        # Check if the configuration file exists
        if not preset_path or not Path(preset_path).exists():
            LOG.error(f"Configuration file not found: {preset_path}")
            return audio

        preset_cfg = load_yaml_config(preset_path)
        LOG.debug(f"Configuration loaded: {preset_path}")
        
        transform = SoxEffectTransform.get_effects_transform(preset_cfg)
        transforms = [transform]
        
        if transforms:
            for transform in transforms:
                # Apply the transformation
                LOG.debug(f"Applying SoX transformation: {transform.name}")
                LOG.debug(f"Effects to apply: {transform.effects}")

                # Apply the transformation
                processed_wave, processed_sr = transform.apply_tensor(
                    audio, info["sample_rate"]
                )
                
                # Appliquer le gain s'il est spécifié dans l'effet
                if "effects" in preset_cfg: # TODO Check if redundant with lines 361
                    for effect in preset_cfg["effects"]:
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
                audio = processed_wave
                info["sample_rate"] = processed_sr
                info["num_samples"] = processed_wave.shape[1]
                
                LOG.debug(f"SoX effects applied successfully. New shape: {processed_wave.shape}")
        else:
            LOG.warning(f"No SoX effect transformations found in configuration {preset_path}")
            
        # Process external sounds if any
        if "external_sounds" in preset_cfg:
            external_sounds_cfg = preset_cfg["external_sounds"]
            LOG.debug("External sounds found in configuration. Adding...")
            transforms = ExternalSoundTransform.get_external_sounds_transforms(external_sounds_cfg)

            if transforms:
                # Apply each transformation
                for transform in transforms:
                    LOG.debug(f"Applying external sound: {transform.sound_name}")

                    # Apply the transformation
                    processed_wave, processed_sr = transform.apply_external_sound(
                        audio, info["sample_rate"]
                    )

                    # Update the audio clip
                    audio = processed_wave
                    info["sample_rate"] = processed_sr

                    LOG.debug(f"External sound added successfully: {transform.sound_name}")
            else:
                LOG.error(
                    f"No external sound transformation found in configuration {preset_path}"
                )
        # except Exception as e:
        #     LOG.error(f"Error when applying configuration: {str(e)}")
        #     import traceback
        #     LOG.error(traceback.format_exc())
        #     return audio

        return audio
                
class ColdDiffusionSoxTransform(nn.Module):
    """
    A transform that applies a chain of SoX effects to an audio tensor,
    with parameters dynamically interpolated based on a timestep 't'.
    This is designed for Cold Diffusion training.

    The preset YAML should be structured as follows:
    effects:
      - name: lowpass
        args: ["-1", [10000, 500]] # Interpolates the cutoff frequency
      - name: rate
        args: [[48000, 8000]] # Interpolates the sample rate
    """
    def __init__(self, name, effects_template, sample_rate, random_config=None):
        super().__init__()
        self.name = name
        self.effects_template = effects_template
        self.sample_rate = sample_rate
        self.random_config = random_config or {}
        LOG.debug(f"[ColdDiffusion] Initialized transform '{name}' with {len(effects_template)} effects at {sample_rate}Hz")
        if len(effects_template) > 0:
            LOG.debug(f"[ColdDiffusion] Effects template: {effects_template}")

    @classmethod
    def from_sox_transform(cls, sox_transform, name, sample_rate):
        """
        Creates a ColdDiffusionSoxTransform from an existing SoxEffectTransform.
        This allows reusing the same transforms created by get_effects_transform.
        
        Args:
            sox_transform: An existing SoxEffectTransform instance
            name: Name for the new transform
            sample_rate: Sample rate to use
            
        Returns:
            A ColdDiffusionSoxTransform that wraps the effects from the SoxEffectTransform
        """
        LOG.debug(f"[ColdDiffusion] Creating from SoxEffectTransform: {name}")
        
        # Convert SoxEffectTransform effects to template for ColdDiffusionSoxTransform
        effects_template = []
        
        if hasattr(sox_transform, 'effects'):
            for effect in sox_transform.effects:
                if effect and len(effect) > 0:
                    effect_dict = {
                        'name': effect[0],
                        'args': effect[1:] if len(effect) > 1 else []
                    }
                    effects_template.append(effect_dict)
                    LOG.debug(f"[ColdDiffusion] Added effect: {effect_dict}")
        
        LOG.debug(f"[ColdDiffusion] Created template with {len(effects_template)} effects")
        return cls(name, effects_template, sample_rate)
    
    @classmethod
    def from_preset(cls, preset_path, sample_rate, seed=None):
        """
        Creates a ColdDiffusionSoxTransform from a preset file.
        preset_path can be either:
          - A full path to a YAML file
          - A preset name (without extension) that exists in a effects directory

        Parameters:
            cls (type): The class to instantiate
            preset_path (str): Path to the preset file or preset name
            sample_rate (int): Sample rate to use
            seed (int, optional): Seed for random number generator

        Returns:
            A ColdDiffusionSoxTransform instance
        """
        try:
            # Check if it's already a complete path
            if os.path.exists(preset_path):
                LOG.debug(f"[ColdDiffusion] Loading preset from direct path: {preset_path}")
                preset_full_path = preset_path
            else:
                # Check if it's a preset name without extension
                if not preset_path.endswith(".yaml"):
                    LOG.debug(f"[ColdDiffusion] Adding .yaml extension to: {preset_path}")
                    preset_with_ext = preset_path + ".yaml"
                else:
                    preset_with_ext = preset_path
                
                # Case where it's just the preset name, search in effect directories
                preset_name = os.path.basename(preset_with_ext)
                
                # Search in possible directories (like done in get_metadata_on_the_fly)
                candidate_dirs = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configs/dataset_configs/low_quality_effect"),
                    "/stable_audio_tools/configs/dataset_configs/low_quality_effect",
                    "/stable-clearaudio/configs/dataset_configs/low_quality_effect"
                ]
                
                preset_full_path = None
                for base_dir in candidate_dirs:
                    # Normalize the path
                    if base_dir.startswith("/stable"):
                        # Get the project base path
                        base_path = Path(__file__).resolve().parent.parent  # Go two levels up from transforms/
                        
                        if base_dir.startswith("/stable-clearaudio/"):
                            relative_path = base_dir[len("/stable-clearaudio/"):]
                            potential_dir = str(base_path / relative_path)
                        elif base_dir.startswith("/stable_audio_tools/"):
                            relative_path = base_dir[len("/stable_audio_tools/"):]
                            potential_dir = str(base_path / relative_path)
                        else:
                            potential_dir = base_dir
                    
                    potential_path = os.path.join(potential_dir, preset_name)
                    if os.path.exists(potential_path):
                        preset_full_path = potential_path
                        LOG.debug(f"[ColdDiffusion] Found preset at: {preset_full_path}")
                        break
                
                if preset_full_path is None:
                    LOG.error(f"[ColdDiffusion] Preset file not found anywhere: {preset_path}")
                    return cls(f"invalid_preset_{os.path.basename(preset_path)}", [], sample_rate, seed=seed)
            
            LOG.debug(f"[ColdDiffusion] Loading preset from {preset_full_path}")
            preset_data = load_yaml_config(preset_full_path)
            
            effects_template = []
            if "effects" in preset_data:
                for effect in preset_data["effects"]:
                    if "name" in effect:
                        effects_template.append(effect)
                LOG.debug(f"[ColdDiffusion] Loaded {len(effects_template)} effects from preset")
            else:
                LOG.warning(f"[ColdDiffusion] No 'effects' found in preset {preset_full_path}")

            # Extract randomization config
            random_config = {}
            if preset_data.get("randomise_effects", False):
                LOG.debug("[ColdDiffusion] Randomization is enabled for this preset.")
                random_config['randomise'] = True
                random_config['min_effects'] = preset_data.get("min_effects", 1)
                random_config['max_effects'] = preset_data.get("max_effects", len(effects_template))
                
                # Calculate weights for sampling
                weights = [e.get('weight', 1.0) for e in effects_template]
                total_weight = sum(weights)
                random_config['weights'] = [w / total_weight for w in weights]

            return cls(os.path.basename(preset_path), effects_template, sample_rate, random_config=random_config)
        except Exception as e:
            LOG.error(f"[ColdDiffusion] Error loading preset {preset_path}: {str(e)}")
            import traceback
            LOG.error(traceback.format_exc())
            return cls(f"error_preset_{os.path.basename(preset_path)}", [], sample_rate)

    def _interpolate(self, value, t):
        """Linearly interpolates a value if it's a list of two numbers."""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                start, end = float(value[0]), float(value[1])
                interpolated = start + t * (end - start)
                LOG.debug(f"[ColdDiffusion] Interpolated value [{start}, {end}] at t={t:.4f} -> {interpolated:.4f}")
                return interpolated
            except (ValueError, TypeError) as e:
                LOG.debug(f"[ColdDiffusion] Failed to interpolate {value}: {str(e)}")
                return value
        else:
            LOG.debug(f"[ColdDiffusion] Using constant value {value} (not interpolatable)")
            return value

    def apply(self, audio_tensor: torch.Tensor, t: float, degradation_seed: int = None) -> torch.Tensor:
        """
        Builds the effects chain with interpolated parameters for a given timestep 't'
        and applies it to the audio tensor.
        """
        LOG.debug(f"[ColdDiffusion] Applying effects with timestep t={t:.4f} to tensor shape={audio_tensor.shape}")

        if degradation_seed is not None:
            np.random.seed(degradation_seed) # Ensure deterministic degradation
            LOG.debug(f"[ColdDiffusion] Applied seed {degradation_seed} for deterministic degradation.")
        
        if not self.effects_template:
            LOG.warning(f"[ColdDiffusion] No effects template found for {self.name}")
            return audio_tensor

        selected_effects = []
        if self.random_config.get('randomise'):
            # Select a random number of effects to apply
            min_effects = self.random_config['min_effects']
            max_effects = self.random_config['max_effects']
            
            if min_effects >= max_effects:
                num_effects_to_apply = min_effects
            else:
                num_effects_to_apply = np.random.randint(min_effects, max_effects + 1)
            
            num_effects_to_apply = min(num_effects_to_apply, len(self.effects_template))

            # Choose effects based on weights
            selected_effects = np.random.choice(
                self.effects_template,
                size=num_effects_to_apply,
                replace=False, # No duplicates
                p=self.random_config['weights']
            )
            LOG.debug(f"[ColdDiffusion] Randomly selected {len(selected_effects)} effects to apply.")
        else:
            # If not randomizing, use all effects from the template
            selected_effects = self.effects_template

        # Create a temporary SoxEffectTransform with interpolated effects
        temp_transform = SoxEffectTransform(name=f"{self.name}_t{t:.2f}")
        
        # Build the list of effects with parameter interpolation
        for effect_template in selected_effects:
            # Use the new centralized method to generate effects
            generated_effects = SoxEffectTransform._generate_effects_from_template(effect_template, t)
            for effect_parts in generated_effects:
                temp_transform.add_effect(effect_parts)
            LOG.debug(f"[ColdDiffusion] Adding interpolated effect: {generated_effects}")


        if not temp_transform.effects:
            LOG.warning(f"[ColdDiffusion] No effects generated from template for t={t:.4f}")
            return audio_tensor
        
        device = audio_tensor.device
        LOG.debug(f"[ColdDiffusion] Original audio - device: {device}, shape: {audio_tensor.shape}, min: {audio_tensor.min():.4f}, max: {audio_tensor.max():.4f}")
        
        # Handle both batched (3D) and non-batched (2D) tensors
        is_batched = audio_tensor.dim() == 3
        if not is_batched and audio_tensor.dim() != 2:
            LOG.error(f"[ColdDiffusion] Unsupported tensor shape: {audio_tensor.shape}. Expected 2D or 3D tensor.")
            return audio_tensor

        # Uniformly handle batched and non-batched by iterating
        input_tensors = audio_tensor if is_batched else audio_tensor.unsqueeze(0)
        processed_samples = []

        for i, sample_in in enumerate(input_tensors):
            try:
                LOG.debug(f"[ColdDiffusion] Applying effects to sample {i} with shape {sample_in.shape}: {temp_transform.effects}")
                
                # Move tensor to CPU for SoX processing
                audio_tensor_cpu = sample_in.to('cpu', dtype=torch.float32)
                processed_audio, out_sr = temp_transform.apply_tensor(audio_tensor_cpu, self.sample_rate)
                
                # Move back to original device
                processed_audio = processed_audio.to(device)
                LOG.debug(f"[ColdDiffusion] After SoX for sample {i} - shape: {processed_audio.shape}, sr: {out_sr}")

                # Verify and fix shape if necessary
                if processed_audio.shape != sample_in.shape:
                    LOG.warning(f"[ColdDiffusion] Shape mismatch for sample {i}: input={sample_in.shape}, output={processed_audio.shape}")
                    
                    # Adjust channel count
                    if processed_audio.shape[0] != sample_in.shape[0]:
                        LOG.debug(f"[ColdDiffusion] Fixing channel count for sample {i}: {processed_audio.shape[0]} -> {sample_in.shape[0]}")
                        if processed_audio.shape[0] == 1 and sample_in.shape[0] > 1:
                            processed_audio = processed_audio.repeat(sample_in.shape[0], 1)
                        else: # Collapse to mono and repeat
                            processed_audio = processed_audio.mean(dim=0, keepdim=True).repeat(sample_in.shape[0], 1)
                    
                    # Adjust length
                    if processed_audio.shape[1] != sample_in.shape[1]:
                        LOG.debug(f"[ColdDiffusion] Fixing audio length for sample {i}: {processed_audio.shape[1]} -> {sample_in.shape[1]}")
                        processed_audio = F.interpolate(processed_audio.unsqueeze(0), size=sample_in.shape[1], mode='linear', align_corners=False).squeeze(0)

                processed_samples.append(processed_audio)

            except Exception as e:
                LOG.error(f"[ColdDiffusion] SoX effect application failed on sample {i}: {e}")
                LOG.error(f"[ColdDiffusion] Traceback (most recent call last):\n{traceback.format_exc()}")
                # In case of error, append the original sample to avoid crashing
                processed_samples.append(sample_in)
        
        # Stack the processed samples back into a single tensor
        if not processed_samples:
            LOG.warning("[ColdDiffusion] No samples were processed.")
            return audio_tensor
            
        output_tensor = torch.stack(processed_samples)
        return output_tensor if is_batched else output_tensor.squeeze(0)


class ExternalSoundTransform(nn.Module):
    def __init__(self, sound_name: str, sound_path: List[str], gain: float = 1.0):
        super().__init__()
        self.sound_name = sound_name
        self.sound_path = sound_path
        self.gain = gain

    @staticmethod
    def get_external_sounds_transforms(external_sounds_cfg: Any) -> List[ExternalSoundTransform]:
        transforms = []
        # Handle both possible data structures: a list of sounds or a dict containing 'external_sounds' key
        if isinstance(external_sounds_cfg, list):
            sounds_list = external_sounds_cfg
        elif isinstance(external_sounds_cfg, dict) and 'external_sounds' in external_sounds_cfg:
            sounds_list = external_sounds_cfg['external_sounds']
        else:
            LOG.error(f"Unexpected external_sounds_cfg format: {type(external_sounds_cfg)}")
            return []
            
        for sound in sounds_list:
            sound_name = sound['name']
            sound_path = sound['sound_path']
            # Get gain parameter if it exists, otherwise use default value
            gain = sound.get('gain', 1.0)
            transforms.append(ExternalSoundTransform(sound_name, sound_path, gain))
        return transforms

    def apply_external_sound(self, tensor: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
        if not self.sound_path:
            LOG.error(f"Sound {self.sound_name} not found in {self.sound_name}")
            return tensor, sample_rate
        
        try:
            if not os.path.exists(self.sound_path[0]):
                LOG.error(f"File does not exist: {self.sound_path[0]}")
                return tensor, sample_rate
            
            if not os.access(self.sound_path[0], os.R_OK):
                LOG.error(f"File is not readable: {self.sound_path[0]}")
                return tensor, sample_rate
            
            LOG.debug(f"Loading external sound from: {self.sound_path[0]}")
            external_waveform, external_sr = torchaudio.load(self.sound_path[0])
            
            LOG.debug(f"External sound loaded with shape: {external_waveform.shape}, sample rate: {external_sr}")
            LOG.debug(f"Original audio shape: {tensor.shape}, sample rate: {sample_rate}")
            
            # Convert to mono if necessary
            if external_waveform.shape[0] > 1:
                external_waveform = torch.mean(external_waveform, dim=0, keepdim=True)
                LOG.debug(f"Converted external sound to mono: {external_waveform.shape}")
            
            if sample_rate != external_sr:
                LOG.warning(f"Resampling external sound: {self.sound_name} from {external_sr}Hz to {sample_rate}Hz. This may cause quality loss and take a long time.")
                external_waveform, _ = resample_signal(external_waveform, external_sr, sample_rate)
                LOG.debug(f"Resampled external sound shape: {external_waveform.shape}")
            
            if external_waveform.shape[1] < tensor.shape[1]:
                repeat_times = tensor.shape[1] // external_waveform.shape[1] + 1
                LOG.debug(f"Repeating external sound {repeat_times} times to match original audio length")
                external_waveform = external_waveform.repeat(1, repeat_times)
            
            external_waveform = external_waveform[:, :tensor.shape[1]]
            LOG.debug(f"Trimmed external sound to match original audio: {external_waveform.shape}")
            
            # Normalize both signals before mixing
            tensor_normalized = tensor / tensor.abs().max()
            external_normalized = external_waveform / external_waveform.abs().max()
            
            # Apply gain to external sound
            external_normalized = self.apply_gain_to_audio(external_normalized, self.gain)
            
            # Mix the original audio with the external sound
            # Ensure original audio is preserved by using a weighted sum
            mixed_waveform = tensor_normalized + external_normalized
            
            # Normalize to prevent clipping
            if mixed_waveform.abs().max() > 0:
                mixed_waveform = mixed_waveform / mixed_waveform.abs().max()
                LOG.debug(f"Normalized mixed waveform to prevent clipping")
            
            LOG.debug(f"Final mixed waveform shape: {mixed_waveform.shape}")
            LOG.debug(f"Applied external sound '{self.sound_name}' with gain {self.gain}")
            
            return mixed_waveform, sample_rate
        except Exception as e:
            LOG.warning(f"Error when adding external sound: {str(e)}")
            return tensor, sample_rate

    @staticmethod
    def apply_gain_to_audio(tensor: Tensor, gain: float) -> Tensor:
        """
        Applique un gain à un tenseur audio après normalisation
        
        Args:
            tensor: Le tenseur audio à transformer
            gain: La valeur de gain à appliquer (valeur multipliée)
            
        Returns:
            Le tenseur transformé
        """
        LOG.debug(f"Applying gain of {gain} to audio")
        
        # Normalize the tensor
        tensor_normalized = tensor / tensor.abs().max()
        
        # Apply gain
        tensor_with_gain = tensor_normalized * gain
        LOG.debug(f"Applied gain. Max amplitude: {tensor_with_gain.abs().max().item()}")
        
        return tensor_with_gain

class SoxEffectTransform(nn.Module):
    # http://sox.sourceforge.net/sox.html
    # https://pytorch.org/audio/stable/sox_effects.html
    # https://pysox.readthedocs.io/en/latest/api.html
    # https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html

    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
        self.effects: List[List[str]] = []
        self.original_gain: Optional[float] = None # Store gain found in config

    def reset(self):
        self.effects = []

    def __repr__(self) -> str:
        res = ''
        if self.name:
            res = f'Name: {self.name}\n'
        for e in self.effects:
            res += str(e) + '\n'
        return res
    
    @staticmethod
    def _process_effect_params(effect_name: str, effect_type: str, param_set: Dict[str, Any], is_flat: bool, t: float = 1.0) -> Tuple[str, List[str]]:
        """Helper method to process effect parameters based on effect type.
        
        Args:
            effect_name: Name of the effect being processed
            effect_type: Type of effect (equalizer, bass, etc.)
            param_set: Dictionary containing the effect parameters
            is_flat: Whether the param_set has a flat structure or nested
            
        Returns:
            Tuple containing (params_string, debug_info)
        """
        # Get parameters from either flat or nested structure
        params = param_set if is_flat else next(iter(param_set.values()))
        debug_info = []
        
        if effect_type == "equalizer":
            # Get frequency either from normal distribution or range
            if "freq_mean" in params and "freq_std" in params:
                frequency = np.random.normal(
                    params.get("freq_mean", 1000),
                    params.get("freq_std", 100)
                )
                frequency = max(20, min(20000, frequency))  # Clamp to audible range
            else:
                frequency = np.random.uniform(
                    params.get("freq_min", 100),
                    params.get("freq_max", 10000)
                )
            
            # Get gain from range (check for both db_min/db_max and gain_min/gain_max)
            db_value = np.random.uniform(
                params.get("db_min", params.get("gain_min", -10)),
                params.get("db_max", params.get("gain_max", 10))
            ) * t
            
            # Get bandwidth (Q factor)
            q_factor = np.random.uniform(
                params.get("q_min", 0.7),
                params.get("q_max", 1.4)
            )
            
            params_string = f"equalizer {frequency} {q_factor}q {db_value}"
            debug_info = [f"frequency={frequency:.2f}Hz", f"q={q_factor:.2f}", f"gain={db_value:.2f}dB"]
            
        elif effect_type == "bass":
            # Process bass effect
            if is_flat:
                # Flat dictionary structure with keys like freq_min, gain_min, etc.
                # Determine center frequency (10-800 Hz)
                if "freq_mean" in param_set and "freq_std" in param_set:
                    center_freq = np.random.normal(
                        param_set.get("freq_mean"),
                        param_set.get("freq_std")
                    )
                else:
                    center_freq = np.random.uniform(
                        param_set.get("freq_min", 50),
                        param_set.get("freq_max", 300)
                    )
                
                # Ensure frequency is within acceptable range for bass
                # Sox accepts frequencies between 10 and 1000 Hz, but we limit to 800 for safety
                center_freq = max(10, min(800, center_freq))
                
                # Determine gain
                if "db_min" in param_set and "db_max" in param_set:
                    gain = np.random.uniform(
                        param_set.get("db_min"),
                        param_set.get("db_max")
                    ) * t
                else:
                    gain = np.random.uniform(
                        param_set.get("gain_min", -20),
                        param_set.get("gain_max", 20)
                    ) * t
                
                # Q factor (width)
                q_factor = np.random.uniform(
                    param_set.get("q_min", 0.7),
                    param_set.get("q_max", 2.0)
                )
            else:
                # Nested dictionary structure
                set_name, set_params = next(iter(param_set.items()))
                
                # Determine center frequency (10-800 Hz)
                center_freq = np.random.uniform(
                    set_params.get("freq_min", 50),
                    set_params.get("freq_max", 300)
                )
                # Ensure frequency is within acceptable range for bass
                center_freq = max(10, min(800, center_freq))
                
                # Determine gain
                gain = np.random.uniform(
                    set_params.get("gain_min", -20),
                    set_params.get("gain_max", 20)
                ) * t
                
                # Q factor (width)
                q_factor = np.random.uniform(
                    set_params.get("q_min", 0.7),
                    set_params.get("q_max", 2.0)
                )
            
            # Correct order for Sox: gain frequency width_q
            # Note: In Sox, for bass effect, the width parameter must be followed by 'q' to specify Q factor
            params_string = f"bass {gain} {center_freq} {q_factor}q"
            debug_info = [f"gain={gain:.1f}dB", f"freq={center_freq:.1f}Hz", f"Q={q_factor:.2f}"]
            
        elif effect_type == "treble" or effect_type == "highpass" or effect_type == "lowpass":
            if not isinstance(params, dict):
                LOG.warning(f"Invalid parameter type for {effect_type} effect: {type(params)}, expected dict")
                return "", []
            
            # For highpass/lowpass, sox expects frequency, but original code used gain. Replicating old logic for compatibility.
            if "gain_min" in params and "gain_max" in params:
                gain_min = params.get("gain_min", -10)
                gain_max = params.get("gain_max", 10)
                gain = np.random.uniform(gain_min, gain_max)
                gain = max(-20, min(20, gain)) * t
                params_string = f"{effect_type} {gain}"
                debug_info = [f"gain={gain:.2f}dB"]
            # Fallback for highpass/lowpass to use frequency if gain is not specified
            elif effect_type in ["highpass", "lowpass"] and "freq_min" in params and "freq_max" in params:
                freq = np.random.uniform(params["freq_min"], params["freq_max"])
                params_string = f"{effect_type} {freq}"
                debug_info = [f"freq={freq:.2f}Hz"]
            else:
                LOG.warning(f"Missing suitable parameters for {effect_type} effect")

        elif effect_type == "overdrive":
            if isinstance(params, dict):
                gain = np.random.uniform(params.get("gain_min", 0), params.get("gain_max", 20)) * t
                colour = np.random.uniform(params.get("colour_min", 0), params.get("colour_max", 20))
                params_string = f"overdrive {gain} {colour}"
                debug_info = [f"gain={gain:.2f}dB", f"colour={colour:.2f}"]
            else:
                LOG.warning(f"Invalid parameter type for overdrive effect: {type(params)}, expected dict")

        elif effect_type == "reverb":
            if isinstance(params, dict):
                reverberance = np.random.randint(params.get("reverberance_min", 0), params.get("reverberance_max", 100)) * t
                damping = np.random.randint(params.get("damping_min", 0), params.get("damping_max", 100))
                room_scale = np.random.randint(params.get("room_scale_min", 0), params.get("room_scale_max", 100))
                stereo_depth = np.random.randint(params.get("stereo_depth_min", 0), params.get("stereo_depth_max", 100))
                delay = np.random.randint(params.get("delay_min", 0), params.get("delay_max", 50)) if params.get("proba_delay", 0) > np.random.random() else 0
                wet_gain = np.random.uniform(params.get("wet_gain_min", -10), params.get("wet_gain_max", 10)) * t
                params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                debug_info = [f"reverberance={reverberance}", f"wet_gain={wet_gain}"]
            else:
                LOG.warning(f"Invalid parameter type for reverb effect: {type(params)}, expected dict")

        elif effect_type == "echo":
            if isinstance(params, dict):
                gain_in = np.random.uniform(params.get("gain_in_min", 0.9), params.get("gain_in_max", 1.0))
                gain_out = np.random.uniform(params.get("gain_out_min", 0.1), params.get("gain_out_max", 0.9)) * t
                delay = np.random.uniform(params.get("delay_min", 100), params.get("delay_max", 500))
                decay = np.random.uniform(params.get("decay_min", 0.1), params.get("decay_max", 0.9))
                params_string = f"echo {gain_in} {gain_out} {delay} {decay}"
                debug_info = [f"gain_in={gain_in:.2f}", f"gain_out={gain_out:.2f}", f"delay={delay:.1f}ms", f"decay={decay:.2f}"]
            else:
                LOG.warning(f"Invalid parameter type for echo effect: {type(params)}, expected dict")

        elif effect_type == "sinc":
            if isinstance(params, dict):
                attenuation = np.random.uniform(0, params.get("att_max", 100)) * t
                cutoff_freq = np.random.uniform(params.get("min_freq", 450), params.get("max_freq", 8000))
                params_string = f"sinc -{attenuation} {cutoff_freq}"
                debug_info = [f"attenuation={attenuation:.1f}dB", f"cutoff={cutoff_freq:.1f}Hz"]
            else:
                LOG.warning(f"Invalid parameter type for sinc effect: {type(params)}, expected dict")

        elif effect_type == "band":
            if isinstance(params, dict):
                center_freq = np.random.uniform(params.get("min_freq", 500), params.get("max_freq", 8000))
                width_q = np.random.uniform(0.5, 2.0)
                params_string = f"band -n {center_freq} {width_q}"
                debug_info = [f"center={center_freq:.1f}Hz", f"width_q={width_q:.2f}"]
            else:
                LOG.warning(f"Invalid parameter type for band effect: {type(params)}, expected dict")
            
        elif effect_type == "flanger":
            # Flanger with default parameters
            params_string = "flanger"
            debug_info = ["default parameters"]
            
        elif effect_type == "riaa":
            # Simple effect with no parameters
            params_string = "riaa"
            debug_info = ["default parameters"]
            
        elif effect_type == "hilbert":
            # Simple effect with no parameters
            params_string = "hilbert"
            debug_info = ["default parameters"]
            
        else:
            LOG.warning(f"Unknown effect type: {effect_type}")
            params_string = ""
            debug_info = []
            
        return params_string, debug_info

    @staticmethod
    def get_effects_transform(preset_cfg: Any) -> Union[SoxEffectTransform, List[SoxEffectTransform]]:
        """
        Create a SoxEffectTransform with random or specific effects based on the preset configuration.
        
        This method handles both random effects selection and specific effects configurations.
        For random effects, it selects effects from a predefined list in the configuration
        and applies them with random parameters to create controlled audio degradations.
        For specific effects, it creates transforms based on explicitly defined effects in the config.
        
        Args:
            preset_cfg: Configuration object containing settings for effects and presets
                
        Returns:
            For random effects: SoxEffectTransform object
            For specific effects: List[SoxEffectTransform] objects
        """
        # Check if this is a specific effects configuration
        if 'transform_type' in preset_cfg and preset_cfg['transform_type'] == 'sox_audio_effect':
            LOG.debug("Creating specific SoxEffectTransform objects from configuration: {preset_cfg}")
        else:
            LOG.error("Invalid or missing transform type")
            return None
        
        if "available_effects" not in preset_cfg:
            LOG.error(f"available_effects not found in configuration. Preset config: {preset_cfg}")
            return None

        if "randomise_effects" not in preset_cfg:
            LOG.error(f"randomise_effects not found in configuration. Preset config: {preset_cfg}")
            return None
        
        available_effects = preset_cfg["available_effects"]

        if preset_cfg["randomise_effects"]:
            LOG.debug("Randomising effects")
            min_effects = preset_cfg["min_effects"]
            max_effects = preset_cfg["max_effects"]
            # Select a random number of effects to apply
            if min_effects == max_effects:
                num_effects_to_apply = min_effects
            else:
                num_effects_to_apply = np.random.randint(min_effects, max_effects + 1)
            LOG.debug(f"Choosing {num_effects_to_apply} effects in {available_effects}")
        else:
            num_effects_to_apply = len(preset_cfg["effects"])
            LOG.debug("Not randomising effects")
        
        # Convert effects list to a dict for easier lookup
        effects_dict = {}
        if isinstance(preset_cfg["effects"], list):
            for effect in preset_cfg["effects"]:
                if "name" in effect:
                    effects_dict[effect["name"]] = effect
        else:
            # If effects is already a dict, use it directly
            effects_dict = preset_cfg["effects"]
        
        # Calculate weights for available effects
        effects_weights = []
        for effect_name in available_effects:
            if effect_name in effects_dict:
                # Check if effect has a specific weight defined
                effect_cfg = effects_dict[effect_name]
                weight = effect_cfg.get("weight", 1.0)
                effects_weights.append(weight)
            else:
                LOG.warning(f"Effect {effect_name} not found in effects configuration")
                effects_weights.append(0.0)  # Zero weight for missing effects

        # Normalize weights
        total_weight = sum(effects_weights)
        if total_weight > 0:
            effects_weights = [w / total_weight for w in effects_weights]
        else:
            # If all weights are zero, use equal weights
            effects_weights = [1.0 / len(available_effects)] * len(available_effects)
        
        # Choose which effects to apply from the available list based on their weights
        effects_to_apply = np.random.choice(
            available_effects, 
            size=min(num_effects_to_apply, len(available_effects)), 
            replace=False,  # No duplicates
            p=np.array(effects_weights)  # Already normalized
        )
        
        # Create a string representation of the selected effects
        effects_name = "_".join(effects_to_apply)
        
        # Initialize transform with the string name
        sox_effect_transform = SoxEffectTransform(effects_name)
        
        # Apply each selected effect with random parameters
        for effect_name in effects_to_apply:
            effect_template = next((e for e in effects_dict.values() if e["name"] == effect_name), None)
            if effect_template:
                # Use the new centralized method to generate effects, t=1.0 for full effect
                generated_effects = SoxEffectTransform._generate_effects_from_template(effect_template, t=1.0)
                for effect_parts in generated_effects:
                    sox_effect_transform.add_effect(effect_parts)

        # Add debug information about the final transform
        LOG.debug(f"Created SoxEffectTransform with {len(sox_effect_transform.effects)} effects")
        
        # Normalize gain if configured
        if preset_cfg.get("normalize_inputs_post_eq", False):
            sox_effect_transform.normalize_gain()
            LOG.debug("Applied gain normalization as specified in config")
        
        return sox_effect_transform

    @staticmethod
    def _generate_effects_from_template(effect_template: Dict[str, Any], t: float = 1.0) -> List[List[str]]:
        """Generate a list of SoX effect arguments for a single effect template, applying randomization and time-based interpolation."""
        effects_to_add = []
        effect_name = effect_template.get("name")
        effect_cfg = effect_template

        if "param_sets" not in effect_cfg:
            LOG.warning(f"param_sets not found in effect {effect_name}")
            return effects_to_add
        
        param_sets = effect_cfg.get("param_sets", [])
        if not param_sets:
            LOG.warning(f"param_sets is empty for effect {effect_name}")
            return effects_to_add

        for i, param_set in enumerate(param_sets):
            if isinstance(param_set, str):
                effect_parts = param_set.strip().split(" ")
                if effect_parts:
                    effects_to_add.append(effect_parts)
                    LOG.debug(f"Added direct string effect: {param_set}")
            else:
                if "effect_type" in effect_cfg:
                    effect_type = effect_cfg["effect_type"]
                    params_string, debug_info = "", []

                    params_string, debug_info = SoxEffectTransform._process_effect_params(
                        effect_name, effect_type, param_set, is_flat=True, t=t)

                    if params_string:
                        effects_to_add.append(params_string.strip().split(" "))
                        LOG.debug(f"Added {effect_type} effect {i} with: {', '.join(debug_info)}")
                else:
                    LOG.warning(f"No effect_type specified for {effect_name}")
        
        return effects_to_add

    def to_mono(self, prepend: bool = True) -> SoxEffectTransform:
        """If prepend is True, the first effect will transform the audio to mono and then apply the effects"""
        effect = ["remix", "-"]
        if prepend:
            self.effects.insert(0, effect)
        else:
            self.effects.append(effect)
        return self

    def add_effect(self, effect: List[Any]) -> SoxEffectTransform:
        effect = list(map(lambda x: str(x), effect))
        self.effects.append(effect)
        return self

    def normalize_gain(self, level: float = None) -> SoxEffectTransform:
        effect = ["gain", "-n"]
        if level is not None:
            effect.append(str(level))
        self.effects.append(effect)
        return self

    def apply_tensor(self, tensor: Tensor, sample_rate: int) -> Tuple[Tensor, int]:        
        # Apply SOX effects directement
        audio, sr = sox.apply_effects_tensor(tensor, sample_rate, self.effects, channels_first=True)
        LOG.debug(f"Audio shape after SOX effects: {audio.shape}")
        
        # Debug the amplitude
        max_amp = audio.abs().max().item()
        LOG.debug(f"Maximum amplitude after effects: {max_amp}")

        return audio, sr

    def forward(self, tensor: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
        return self.apply_tensor(tensor, sample_rate)
