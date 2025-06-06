from __future__ import annotations
import logging
import os

from typing import Any, List, Tuple, Dict, Optional
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
logging.basicConfig(level=logging.DEBUG, 
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

def apply_config_to_audio(info, audio, preset_path, effects_mode):
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

        # Load the configuration
        try:
            preset_cfg = load_yaml_config(preset_path)
            LOG.debug(f"Configuration loaded: {preset_path}")
            
            if effects_mode == "random":
                # In random mode, we only apply one transformation (as each random preset contains one list of effects)
                transform = SoxEffectTransform.get_random_effects_transform(preset_cfg)
                transforms = [transform]
            elif effects_mode == "specific":
                # In specific mode, we apply all transformations specified in "effects" in the preset_cfg
                transforms = SoxEffectTransform.get_specific_effects_transforms(preset_cfg)
            
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
        except Exception as e:
            LOG.error(f"Error when applying configuration: {str(e)}")
            import traceback
            LOG.error(traceback.format_exc())
            return audio

        return audio

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
            
            # Convertir en mono si nécessaire
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
    def _process_effect_params(effect_name: str, effect_type: str, param_set: Dict[str, Any], is_flat: bool) -> Tuple[str, List[str]]:
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
            )
            
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
                    )
                else:
                    gain = np.random.uniform(
                        param_set.get("gain_min", -20),
                        param_set.get("gain_max", 20)
                    )
                
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
                )
                
                # Q factor (width)
                q_factor = np.random.uniform(
                    set_params.get("q_min", 0.7),
                    set_params.get("q_max", 2.0)
                )
            
            # Correct order for Sox: gain frequency width_q
            # Note: In Sox, for bass effect, the width parameter must be followed by 'q' to specify Q factor
            params_string = f"bass {gain} {center_freq} {q_factor}q"
            debug_info = [f"gain={gain:.1f}dB", f"freq={center_freq:.1f}Hz", f"Q={q_factor:.2f}"]
            
        elif effect_type == "treble":
            # Check if params is a dictionary
            if not isinstance(params, dict):
                LOG.warning(f"Invalid parameter type for treble effect: {type(params)}, expected dict")
                return "", []
                
            # Get gain (only parameter we need for simplified treble effect)
            if "gain_min" in params and "gain_max" in params:
                gain = np.random.uniform(
                    params.get("gain_min", -10),
                    params.get("gain_max", 10)
                )
                # Ensure gain is within safe range
                gain = max(-10, min(10, gain))
                
                # Use simplified treble effect format
                params_string = f"treble {gain}"
                debug_info = [f"gain={gain:.2f}dB"]
            else:
                LOG.warning(f"Missing required gain parameters for treble effect")
                return "", []
            
        elif effect_type == "overdrive":
            # Get gain
            gain = np.random.uniform(
                params.get("gain_min", 5),
                params.get("gain_max", 40)
            )
            
            # Get color
            color = np.random.uniform(
                params.get("color_min", 20),
                params.get("color_max", 100)
            )
            
            params_string = f"overdrive {gain} {color}"
            debug_info = [f"gain={gain:.2f}dB", f"color={color:.2f}"]
            
        elif effect_type == "reverb":
            # Get reverberance
            reverberance = np.random.randint(
                params.get("reverberance_min", 0),
                params.get("reverberance_max", 100)
            )
            
            # Get damping
            damping = np.random.randint(
                params.get("damping_min", 0),
                params.get("damping_max", 100)
            )
            
            # Get room scale
            room_scale = np.random.randint(
                params.get("room_scale_min", 0),
                params.get("room_scale_max", 100)
            )
            
            # Get stereo depth
            stereo_depth = np.random.randint(
                params.get("stereo_depth_min", 0),
                params.get("stereo_depth_max", 100)
            )
            
            # Get delay (if applicable)
            use_delay = params.get("use_delay", False)
            proba_delay = params.get("proba_delay", 0)
            apply_delay = use_delay or (np.random.random() < proba_delay)
            
            delay = np.random.randint(
                params.get("delay_min", 0),
                params.get("delay_max", 50)
            ) if apply_delay else 0
            
            # Get wet gain
            wet_gain = np.random.uniform(
                params.get("wet_gain_min", -10),
                params.get("wet_gain_max", 10)
            )
            
            params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
            debug_info = [f"reverberance={reverberance}", f"damping={damping}", 
                          f"room_scale={room_scale}", f"stereo_depth={stereo_depth}"]
            
        elif effect_type == "flanger":
            # Flanger with default parameters
            params_string = "flanger"
            debug_info = ["default parameters"]
            
        elif effect_type == "echo":
            # Get gain in
            gain_in = np.random.uniform(
                params.get("gain_in_min", 0.9),
                params.get("gain_in_max", 1.0)
            )
            
            # Get gain out
            gain_out = np.random.uniform(
                params.get("gain_out_min", 0.1),
                params.get("gain_out_max", 0.9)
            )
            
            # Get delay
            delay = np.random.uniform(
                params.get("delay_min", 100),
                params.get("delay_max", 500)
            )
            
            # Get decay
            decay = np.random.uniform(
                params.get("decay_min", 0.1),
                params.get("decay_max", 0.9)
            )
            
            params_string = f"echo {gain_in} {gain_out} {delay} {decay}"
            debug_info = [f"gain_in={gain_in:.2f}", f"gain_out={gain_out:.2f}", 
                          f"delay={delay:.1f}ms", f"decay={decay:.2f}"]
            
        elif effect_type == "riaa":
            # Simple effect with no parameters
            params_string = "riaa"
            debug_info = ["default parameters"]
            
        elif effect_type == "hilbert":
            # Simple effect with no parameters
            params_string = "hilbert"
            debug_info = ["default parameters"]
            
        elif effect_type == "sinc":
            # Get attenuation
            attenuation = np.random.uniform(
                0,
                params.get("att_max", 100)
            )
            
            # Get cutoff frequency
            cutoff_freq = np.random.uniform(
                params.get("min_freq", 450),
                params.get("max_freq", 8000)
            )
            
            params_string = f"sinc -{attenuation} {cutoff_freq}"
            debug_info = [f"attenuation={attenuation:.1f}dB", f"cutoff={cutoff_freq:.1f}Hz"]
            
        elif effect_type == "band":
            # Get center frequency
            center_freq = np.random.uniform(
                params.get("min_freq", 500),
                params.get("max_freq", 8000)
            )
            
            # Get width Q
            width_q = np.random.uniform(0.5, 2.0)
            
            params_string = f"band -n {center_freq} {width_q}"
            debug_info = [f"center={center_freq:.1f}Hz", f"width_q={width_q:.2f}"]
            
        else:
            LOG.warning(f"Unknown effect type: {effect_type}")
            params_string = ""
            debug_info = []
            
        return params_string, debug_info
    
    # Méthode get_specific_effects_transforms conservée pour référence, mais non utilisée dans le code actuel
    @staticmethod
    def get_specific_effects_transforms(preset_cfg: Any) -> List[SoxEffectTransform]:
        """From a configuration dictionary, create a list of SoxEffectTransform objects.

        Args:
            preset_cfg: Configuration dictionary
            
        Returns:
            List of SoxEffectTransform objects
        """
        transforms = []
        LOG.debug(f"Creating SoxEffectTransform objects from configuration")
        
        # Check if transform_type exists
        if 'transform_type' not in preset_cfg:
            LOG.error("'transform_type' not found in preset_config")
            return []
        
        # Check transform type
        if preset_cfg['transform_type'] == 'sox_audio_effect':
            # Check if effects exists
            if 'effects' not in preset_cfg:
                LOG.error("'effects' not found in preset_config")
                return []
                                    
            for effect in preset_cfg['effects']: # Most of the time, effect contains only one element
                sox_effect_transform = SoxEffectTransform()
                
                # Check if params exists
                if 'params' not in effect:
                    LOG.error(f"'params' not found in effect: {effect}")
                    continue
                    
                for param_string in effect["params"]: 
                    # Check if the effect is recognized by Sox
                    if param_string.split(" ")[0] not in sox.effect_names():
                        LOG.error(f"Sox does not recognize the effect: {param_string.split(' ')[0]}")
                        continue

                    LOG.debug(f"Adding effect: {param_string}")

                    # Special handling for gain -n parameter
                    if param_string.startswith("gain -n"):
                        parts = param_string.strip().split(" ")
                        if len(parts) >= 3:
                            level = parts[2]
                            LOG.debug(f"Found gain -n parameter with level: {level}")
                            sox_effect_transform.normalize_gain(float(level))
                        else:
                            sox_effect_transform.normalize_gain()
                    else:
                        params = param_string.strip().split(" ")
                        sox_effect_transform.add_effect(params)
                
                # Check for effect-level gain parameter
                if 'gain' in effect:
                    effect_gain = effect['gain']
                    LOG.debug(f"Found effect-level gain parameter: {effect_gain} (will be applied separately, not added to SoX effects)")
                    sox_effect_transform.original_gain = float(effect_gain) # Store gain here
                    # sox_effect_transform.add_gain(float(effect_gain)) # Don't add gain here, it's handled later
                elif 'normalize_inputs_post_eq' in preset_cfg and preset_cfg['normalize_inputs_post_eq']:
                    sox_effect_transform.normalize_gain()
                    
                transforms.append(sox_effect_transform)
        else:
            LOG.error(f"Transformation type '{preset_cfg['transform_type']}' is not supported. Only sox_audio_effect is supported for now.")
            return []

        return transforms

    @staticmethod
    def from_mode_config_old(cfg: Any) -> Dict[str, List[SoxEffectTransform]]:
        transforms = {}
        for mode in ['train','test','validation']:
            if cfg.dataset.low_quality_effect[mode]:
                transforms[mode] = []
                for effect in cfg.dataset.low_quality_effect[mode]['effects']:
                    sox_effect_transform = SoxEffectTransform(effect["name"])
                    for param_string in effect['params']:
                        params = param_string.strip().split(" ")
                        sox_effect_transform.add_effect(params)
                    if cfg.dataset.normalize_inputs_post_eq:
                        sox_effect_transform.normalize_gain()
                    transforms[mode].append(sox_effect_transform)
        return transforms

    @staticmethod
    
    def get_random_effects_transform(preset_cfg: Any) -> SoxEffectTransform:
        """
        Create a SoxEffectTransform with randomly selected effects from a preset list.
        
        This method selects effects from a predefined list in the configuration and applies them
        with random parameters to create controlled audio degradations.
        
        Args:
            preset_cfg: Configuration object containing settings for effects and presets
                
        Returns:
            SoxEffectTransform object
        """
        LOG.debug(f"Creating random effects transform from configuration: {preset_cfg}")
        # Initialize transform
        sox_effect_transform = SoxEffectTransform("random_preset")
        
        if "available_effects" not in preset_cfg:
            LOG.error(f"available_effects not found in configuration. Preset config: {preset_cfg}")
            return None
        
        # Get global parameters
        min_effects = preset_cfg["min_effects"]
        max_effects = preset_cfg["max_effects"]
        available_effects = preset_cfg["available_effects"]
        
        # Select a random number of effects to apply
        if min_effects == max_effects:
            num_effects_to_apply = min_effects
        else:
            num_effects_to_apply = np.random.randint(min_effects, max_effects + 1)
        LOG.debug(f"Choosing {num_effects_to_apply} effects in {available_effects}")
        
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
        
        # Apply each selected effect with random parameters
        for effect_name in effects_to_apply:
            if effect_name not in effects_dict:
                LOG.warning(f"Skipping effect {effect_name} as it's not in effects configuration")
                continue
                
            effect_cfg = effects_dict[effect_name]
            
            # Check for the effect_type
            if "effect_type" in effect_cfg:
                if effect_cfg["effect_type"] == "equalizer":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} equalizer parameter sets for {effect_name}")
                        
                        # Calculate weights for param sets if specified
                        set_weights = []
                        for param_set in param_sets:
                            # Handle string parameter sets directly
                            if isinstance(param_set, str):
                                set_weight = 1.0
                            # Check if param_set is a dict with multiple direct key-value pairs
                            elif isinstance(param_set, dict) and "weight" in param_set:
                                # Direct structure with keys like proba, weight, etc.
                                set_weight = param_set.get("weight", 1.0)
                            # Old format: dict with a single key-value pair where value is another dict
                            elif isinstance(param_set, dict):
                                try:
                                    set_name, set_params = next(iter(param_set.items()))
                                    set_weight = set_params.get("weight", 1.0) if isinstance(set_params, dict) else 1.0
                                except (AttributeError, StopIteration):
                                    LOG.warning(f"Invalid parameter set format: {param_set}")
                                    set_weight = 0.0
                            else:
                                LOG.warning(f"Unexpected parameter set type: {type(param_set)}")
                                set_weight = 0.0
                                
                            set_weights.append(set_weight)
                        
                        # Normalize weights
                        total_weight = sum(set_weights)
                        if total_weight > 0:
                            set_weights = [w / total_weight for w in set_weights]
                        # Apply all parameter sets as defined in intense_equalizer.yaml
                        for i, param_set in enumerate(param_sets):
                            # Handle string parameter sets directly
                            if isinstance(param_set, str):
                                # Split the string into command parts to be used as effect parameters
                                effect_parts = param_set.strip().split(" ")
                                if effect_parts:
                                    sox_effect_transform.add_effect(effect_parts)
                                    LOG.debug(f"Added direct string effect: {param_set}")
                            # Check if param_set is a flat dictionary with direct parameters
                            elif isinstance(param_set, dict) and ("freq_min" in param_set or "freq_mean" in param_set):
                                # Utiliser la fonction helper pour traiter les paramètres
                                LOG.debug(f"Processing param set {i}, params: {param_set}")
                                params_string, debug_info = SoxEffectTransform._process_effect_params(
                                    effect_name, "equalizer", param_set, is_flat=True)
                                
                                # Ajouter l'effet au transform
                                if params_string:
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added equalizer effect {i} with: {', '.join(debug_info)}")
                            # Old format: dict with a single key-value pair where value is another dict
                            elif isinstance(param_set, dict):
                                try:
                                    for set_name, set_params in param_set.items():
                                        LOG.debug(f"Processing set: {set_name}, params: {set_params}")
                                        
                                        if isinstance(set_params, dict):
                                            # Utiliser la fonction helper pour traiter les paramètres
                                            params_string, debug_info = SoxEffectTransform._process_effect_params(
                                                effect_name, "equalizer", set_params, is_flat=True)
                                            
                                            # Ajouter l'effet au transform
                                            if params_string:
                                                sox_effect_transform.add_effect(params_string.strip().split(" "))
                                                LOG.debug(f"Added equalizer effect {set_name} with: {', '.join(debug_info)}")
                                        else:
                                            LOG.warning(f"Skipping non-dict set_params: {set_params} of type {type(set_params)}")
                                except (AttributeError, TypeError) as e:
                                    LOG.warning(f"Error processing parameter set: {param_set}, error: {str(e)}")
                            else:
                                LOG.warning(f"Skipping unsupported parameter set type: {type(param_set)}")
                                    
                    else:
                        LOG.warning(f"param_sets should be a list for {effect_name}, found {type(param_sets)}")
                elif effect_cfg["effect_type"] in ["treble", "highpass", "lowpass"]:
                    # Handle all filter-type effects (treble, highpass, lowpass) similarly
                    effect_type = effect_cfg["effect_type"]
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} {effect_type} parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Handle string parameter sets directly
                            if isinstance(param_set, str):
                                # Split the string into command parts to be used as effect parameters
                                effect_parts = param_set.strip().split(" ")
                                if effect_parts:
                                    sox_effect_transform.add_effect(effect_parts)
                                    LOG.debug(f"Added direct string {effect_type} effect: {param_set}")
                            elif isinstance(param_set, dict):
                                LOG.debug(f"Processing treble param set {i}, params: {param_set}")
                                
                                # Extract gain parameters directly
                                if "gain_min" in param_set and "gain_max" in param_set:
                                    gain_min = param_set.get("gain_min", -10)
                                    gain_max = param_set.get("gain_max", 10)
                                    gain = np.random.uniform(gain_min, gain_max)
                                    
                                    # Ensure gain is within safe range
                                    gain = max(-10, min(10, gain))
                                    
                                    # Simplified treble effect - only use gain parameter
                                    effect = ["treble", str(gain)]
                                    sox_effect_transform.add_effect(effect)
                                    LOG.debug(f"Added treble effect {i} with gain={gain:.2f}dB")
                                else:
                                    LOG.warning(f"Missing gain parameters for treble effect in param set {i}")
                            else:
                                LOG.warning(f"Invalid parameter type for treble effect: {type(param_set)}, expected dict")
                    else:
                        LOG.warning(f"param_sets should be a list for treble effect {effect_name}")
                elif effect_cfg["effect_type"] == "bass":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} bass parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "freq_min" in param_set or "db_min" in param_set:
                                # Direct structure with keys like freq_min, freq_max, etc.
                                LOG.debug(f"Processing bass param set {i}, params: {param_set}")
                                
                                # Utiliser la fonction helper pour traiter les paramètres
                                params_string, debug_info = SoxEffectTransform._process_effect_params(
                                    effect_name, "bass", param_set, is_flat=True)
                                
                                # Ajouter l'effet au transform
                                if params_string:
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added bass effect {i} with: {', '.join(debug_info)}")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing bass set: {set_name}, params: {set_params}")
                                    
                                    # Utiliser la fonction helper pour traiter les paramètres
                                    params_string, debug_info = SoxEffectTransform._process_effect_params(
                                        effect_name, "bass", set_params, is_flat=True)
                                    
                                    # Ajouter l'effet au transform
                                    if params_string:
                                        sox_effect_transform.add_effect(params_string.strip().split(" "))
                                        LOG.debug(f"Added bass effect {set_name} with: {', '.join(debug_info)}")
                    else:
                        LOG.warning(f"param_sets should be a list for bass effect {effect_name}")
                elif effect_cfg["effect_type"] == "overdrive":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} overdrive parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "min_int" in param_set or "gain_min" in param_set or "colour_min" in param_set:
                                # Direct structure with keys like gain_min, gain_max, etc.
                                LOG.debug(f"Processing overdrive param set {i}, params: {param_set}")
                                
                                # Utiliser la fonction helper pour traiter les paramètres
                                params_string, debug_info = SoxEffectTransform._process_effect_params(
                                    effect_name, "overdrive", param_set, is_flat=True)
                                
                                # Ajouter l'effet au transform
                                if params_string:
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added overdrive effect {i} with: {', '.join(debug_info)}")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing overdrive set: {set_name}, params: {set_params}")
                                    # Utiliser la fonction helper pour traiter les paramètres
                                    params_string, debug_info = SoxEffectTransform._process_effect_params(
                                        effect_name, "overdrive", param_set, is_flat=False)
                                    
                                    # Ajouter l'effet au transform
                                    if params_string:
                                        sox_effect_transform.add_effect(params_string.strip().split(" "))
                                        LOG.debug(f"Added overdrive effect {set_name} with: {', '.join(debug_info)}")
                    else:
                        LOG.warning(f"param_sets should be a list for overdrive effect {effect_name}")
                elif effect_cfg["effect_type"] == "reverb":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} reverb parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "reverberance_min" in param_set:
                                LOG.debug(f"Processing reverb param set {i}, params: {param_set}")
                                
                                # Get default values from the flat dictionary
                                reverberance = np.random.randint(
                                    param_set.get("reverberance_min", 0),
                                    param_set.get("reverberance_max", 100)
                                )
                                damping = np.random.randint(
                                    param_set.get("damping_min", 0),
                                    param_set.get("damping_max", 100)
                                )
                                room_scale = np.random.randint(
                                    param_set.get("room_scale_min", 0),
                                    param_set.get("room_scale_max", 100)
                                )
                                stereo_depth = np.random.randint(
                                    param_set.get("stereo_depth_min", 0),
                                    param_set.get("stereo_depth_max", 100)
                                )
                                delay = np.random.randint(
                                    param_set.get("delay_min", 0),
                                    param_set.get("delay_max", 50)
                                ) if param_set.get("proba_delay", 0) > np.random.random() else 0
                                wet_gain = np.random.uniform(
                                    param_set.get("wet_gain_min", -10),
                                    param_set.get("wet_gain_max", 10)
                                )
                                
                                params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                                sox_effect_transform.add_effect(params_string.strip().split(" "))
                                LOG.debug(f"Added reverb {i} with: reverberance={reverberance}, damping={damping}, room_scale={room_scale}, stereo_depth={stereo_depth}")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing reverb set: {set_name}, params: {set_params}")
                                    reverberance = np.random.randint(
                                        set_params.get("reverberance_min", 0),
                                        set_params.get("reverberance_max", 100)
                                    )
                                    damping = np.random.randint(
                                        set_params.get("damping_min", 0),
                                        set_params.get("damping_max", 100)
                                    )
                                    room_scale = np.random.randint(
                                        set_params.get("room_scale_min", 0),
                                        set_params.get("room_scale_max", 100)
                                    )
                                    stereo_depth = np.random.randint(
                                        set_params.get("stereo_depth_min", 0),
                                        set_params.get("stereo_depth_max", 100)
                                    )
                                    delay = np.random.randint(
                                        set_params.get("delay_min", 0),
                                        set_params.get("delay_max", 50)
                                    ) if set_params.get("use_delay", False) else 0
                                    wet_gain = np.random.uniform(
                                        set_params.get("wet_gain_min", -10),
                                        set_params.get("wet_gain_max", 10)
                                    )
                                    
                                    params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added reverb {set_name} with: reverberance={reverberance}, damping={damping}, room_scale={room_scale}, stereo_depth={stereo_depth}")
                    else:
                        LOG.warning(f"param_sets should be a list for reverb effect {effect_name}")
                elif effect_cfg["effect_type"] == "flanger":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} flanger parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "weight" in param_set:
                                # Direct structure with keys
                                LOG.debug(f"Processing flanger param set {i}, params: {param_set}")
                                
                                # Add flanger effect with default parameters
                                params_string = 'flanger'
                                sox_effect_transform.add_effect(params_string.strip().split(" "))
                                LOG.debug(f"Added flanger effect {i}")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing flanger set: {set_name}, params: {set_params}")
                                    params_string = 'flanger'
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added flanger effect {set_name}")
                    else:
                        LOG.warning(f"param_sets should be a list for flanger effect {effect_name}")
                elif effect_cfg["effect_type"] == "echo":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} echo parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "weight" in param_set in param_set:
                                # Direct structure with keys
                                LOG.debug(f"Processing echo param set {i}, params: {param_set}")
                                
                                # Get random values for echo parameters
                                gain_in = np.random.uniform(
                                    param_set.get("gain_in_min", 0.9),
                                    param_set.get("gain_in_max", 1.0)
                                )
                                gain_out = np.random.uniform(
                                    param_set.get("gain_out_min", 0.1),
                                    param_set.get("gain_out_max", 0.9)
                                )
                                delay = np.random.uniform(
                                    param_set.get("delay_min", 100),
                                    param_set.get("delay_max", 500)
                                )
                                decay = np.random.uniform(
                                    param_set.get("decay_min", 0.1),
                                    param_set.get("decay_max", 0.9)
                                )
                                
                                # Create echo effect string
                                params_string = f"echo {gain_in} {gain_out} {delay} {decay}"
                                sox_effect_transform.add_effect(params_string.strip().split(" "))
                                LOG.debug(f"Added echo effect {i} with: gain_in={gain_in:.2f}, gain_out={gain_out:.2f}, delay={delay:.1f}ms, decay={decay:.2f}")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing echo set: {set_name}, params: {set_params}")
                                    
                                    gain_in = np.random.uniform(
                                        set_params.get("gain_in_min", 0.9),
                                        set_params.get("gain_in_max", 1.0)
                                    )
                                    gain_out = np.random.uniform(
                                        set_params.get("gain_out_min", 0.1),
                                        set_params.get("gain_out_max", 0.9)
                                    )
                                    delay = np.random.uniform(
                                        set_params.get("delay_min", 100),
                                        set_params.get("delay_max", 500)
                                    )
                                    decay = np.random.uniform(
                                        set_params.get("decay_min", 0.1),
                                        set_params.get("decay_max", 0.9)
                                    )
                                    
                                    params_string = f"echo {gain_in} {gain_out} {delay} {decay}"
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added echo effect {set_name} with: gain_in={gain_in:.2f}, gain_out={gain_out:.2f}, delay={delay:.1f}ms, decay={decay:.2f}")
                    else:
                        LOG.warning(f"param_sets should be a list for echo effect {effect_name}")
                elif effect_cfg["effect_type"] == "riaa":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} riaa parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Simple effect with no parameters
                            params_string = "riaa"
                            sox_effect_transform.add_effect(params_string.strip().split(" "))
                            LOG.debug(f"Added RIAA effect")
                    else:
                        LOG.warning(f"param_sets should be a list for riaa effect {effect_name}")
                elif effect_cfg["effect_type"] == "hilbert":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} hilbert parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Simple effect with no parameters
                            params_string = "hilbert"
                            sox_effect_transform.add_effect(params_string.strip().split(" "))
                            LOG.debug(f"Added Hilbert transform effect")
                    else:
                        LOG.warning(f"param_sets should be a list for hilbert effect {effect_name}")
                elif effect_cfg["effect_type"] == "sinc":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} sinc parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "weight" in param_set or "att_max" in param_set:
                                # Direct structure with keys
                                LOG.debug(f"Processing sinc param set {i}, params: {param_set}")
                                
                                # Get random values for sinc parameters
                                attenuation = np.random.uniform(
                                    0,
                                    param_set.get("att_max", 100)
                                )
                                cutoff_freq = np.random.uniform(
                                    param_set.get("min_freq", 450),
                                    param_set.get("max_freq", 8000)
                                )
                                
                                # Create sinc effect string (low-pass filter)
                                params_string = f"sinc -{attenuation} {cutoff_freq}"
                                sox_effect_transform.add_effect(params_string.strip().split(" "))
                                LOG.debug(f"Added sinc low-pass filter {i} with: attenuation={attenuation:.1f}dB, cutoff={cutoff_freq:.1f}Hz")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing sinc set: {set_name}, params: {set_params}")
                                    
                                    attenuation = np.random.uniform(
                                        0,
                                        set_params.get("att_max", 100)
                                    )
                                    cutoff_freq = np.random.uniform(
                                        set_params.get("min_freq", 450),
                                        set_params.get("max_freq", 8000)
                                    )
                                    
                                    params_string = f"sinc -{attenuation} {cutoff_freq}"
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added sinc low-pass filter {set_name} with: attenuation={attenuation:.1f}dB, cutoff={cutoff_freq:.1f}Hz")
                    else:
                        LOG.warning(f"param_sets should be a list for sinc effect {effect_name}")
                elif effect_cfg["effect_type"] == "band":
                    param_sets = effect_cfg.get("param_sets", [])
                    if isinstance(param_sets, list):
                        LOG.debug(f"Processing {len(param_sets)} band parameter sets for {effect_name}")
                        for i, param_set in enumerate(param_sets):
                            # Check if param_set is a flat dictionary with direct parameters
                            if "weight" in param_set or "max_freq" in param_set:
                                # Direct structure with keys
                                LOG.debug(f"Processing band param set {i}, params: {param_set}")
                                
                                # Get random values for band parameters
                                center_freq = np.random.uniform(
                                    param_set.get("min_freq", 500),
                                    param_set.get("max_freq", 8000)
                                )
                                width_q = np.random.uniform(0.5, 2.0)
                                
                                # Create band effect string (band-pass filter)
                                params_string = f"band -n {center_freq} {width_q}"
                                sox_effect_transform.add_effect(params_string.strip().split(" "))
                                LOG.debug(f"Added band-pass filter {i} with: center={center_freq:.1f}Hz, width_q={width_q:.2f}")
                            else:
                                # Old format: dict with a single key-value pair where value is another dict
                                for set_name, set_params in param_set.items():
                                    LOG.debug(f"Processing band set: {set_name}, params: {set_params}")
                                    
                                    center_freq = np.random.uniform(
                                        set_params.get("min_freq", 500),
                                        set_params.get("max_freq", 8000)
                                    )
                                    width_q = np.random.uniform(0.5, 2.0)
                                    
                                    params_string = f"band -n {center_freq} {width_q}"
                                    sox_effect_transform.add_effect(params_string.strip().split(" "))
                                    LOG.debug(f"Added band-pass filter {set_name} with: center={center_freq:.1f}Hz, width_q={width_q:.2f}")
                    else:
                        LOG.warning(f"param_sets should be a list for band effect {effect_name}")
                else:
                    LOG.warning(f"Unknown effect_type: {effect_cfg['effect_type']} for {effect_name}")
            else:
                LOG.warning(f"No effect_type specified for {effect_name}")
        
        # Add debug information about the final transform
        LOG.debug(f"Created SoxEffectTransform with {len(sox_effect_transform.effects)} effects")
        
        # Normalize gain if configured
        if preset_cfg.get("normalize_inputs_post_eq", False):
            sox_effect_transform.normalize_gain()
            LOG.debug("Applied gain normalization as specified in config")
        
        return sox_effect_transform

    @staticmethod
    def random_effects_old(cfg: Any) -> SoxEffectTransform:
        """
        Create a SoxEffectTransform with randomly selected audio effects.
        
        This method generates a collection of audio effects with random parameters
        to degrade audio quality in a controlled manner. It's useful for creating
        synthetic low-quality datasets for training audio restoration models.
        
        Args:
            cfg: Configuration object containing settings for different audio effects
            
        Returns:
            SoxEffectTransform: An object containing the randomly selected audio effects
        """
        # Initialize a new SoxEffectTransform object with name "random"
        sox_effect_transform = SoxEffectTransform("random")
        
        # Get the configuration section for random effects
        cfg_effect = cfg.dataset.low_quality_effect['random']
        
        # Get the minimum and maximum number of effects to apply
        max_effect = cfg_effect.random_effects.max_effect  # Maximum effects to apply
        min_effect = cfg_effect.random_effects.min_effect  # Minimum effects to apply
        
        # Initialize a list to track equalizer frequencies (to avoid overlapping frequencies)
        eq_freq = []
        
        # Initialize flags to ensure each effect type is applied at most once
        treble = False     # High frequency adjustment
        bass = False       # Low frequency adjustment
        overdrive = False  # Distortion effect
        reverb = False     # Echo/reverberation effect
        riaa = False       # Record Industry Association of America equalization curve
        echo = False       # Distinct repeated sound
        band = False       # Band-pass filter
        tremolo = False    # Amplitude modulation effect
        sinc = False       # Sinc filter (brick-wall filter)
        hilbert = False    # Hilbert transform (phase shifting)
        flanger = False    # Delayed copy mixed with original

        # Keep adding effects until we reach the minimum number required
        # while not exceeding the maximum number allowed
        while len(sox_effect_transform.effects) < min_effect and len(sox_effect_transform.effects) < max_effect:
            
            # EQUALIZER EFFECT: Boosts or cuts specific frequency bands
            # Applies with probability specified in config, can apply multiple equalizers
            while np.random.random() < cfg_effect.equalizer.proba and len(sox_effect_transform.effects) < max_effect:
                # Get center frequency from config
                center_freq = cfg_effect.equalizer.freq_mean
                std_freq = cfg_effect.equalizer.freq_std
                
                # Skip frequencies that are too close to ones we've already used
                for freq in eq_freq:
                    if np.abs(freq-center_freq) < 20:  # Skip if within 20Hz of existing frequency
                        continue
                        
                # Remember this frequency so we don't use one too close to it later
                eq_freq.append(center_freq)
                
                # 50% chance to boost the frequency, 50% chance to cut it
                if np.random.random() > 0.5:  # Boost frequency (positive gain)
                    sox_effect_transform.add_equalizer(
                        max(np.random.normal(center_freq, std_freq), 150),  # Random frequency, at least 150Hz
                        np.random.uniform(cfg_effect.equalizer.db_min, cfg_effect.equalizer.db_max),  # Random gain (dB)
                        np.random.uniform(0.7, 5)  # Random Q factor (bandwidth control)
                    )
                else:  # Cut frequency (negative gain)
                    sox_effect_transform.add_equalizer(
                        max(np.random.normal(center_freq, std_freq), 150),  # Random frequency, at least 150Hz
                        np.random.uniform(-cfg_effect.equalizer.db_max, -cfg_effect.equalizer.db_min),  # Random attenuation 
                        np.random.uniform(0.7, 5)  # Random Q factor (bandwidth control)
                    )
            # TREBLE EFFECT: Boost or cut high frequencies
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.treble.proba and len(sox_effect_transform.effects) < max_effect and not treble:
                treble = True  # Mark as applied so we don't add it again
                
                # 50% chance to boost treble, 50% chance to cut it
                if np.random.random() > 0.5:  # Boost high frequencies
                    sox_effect_transform.add_treble(
                        np.random.randint(cfg_effect.treble.freq_min, cfg_effect.treble.freq_max),  # Random frequency 
                        np.random.uniform(cfg_effect.treble.db_min, cfg_effect.treble.db_max),  # Random gain (dB)
                        format(np.random.uniform(0.1, 1), '.2f')  # Random width factor (between 0.1 and 1)
                    )
                else:  # Cut high frequencies
                    sox_effect_transform.add_treble(
                        np.random.randint(cfg_effect.treble.freq_min, cfg_effect.treble.freq_max),  # Random frequency
                        np.random.uniform(-cfg_effect.treble.db_max, cfg_effect.treble.db_min),  # Random attenuation
                        format(np.random.uniform(0.1, 1), '.2f')  # Random width factor (between 0.1 and 1)
                    )
            # BASS EFFECT: Boost or cut low frequencies
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.bass.proba and len(sox_effect_transform.effects) < max_effect and not bass:
                bass = True  # Mark as applied so we don't add it again
                
                # 50% chance to boost bass, 50% chance to cut it
                if np.random.random() > 0.5:  # Boost low frequencies
                    sox_effect_transform.add_bass(
                        np.random.randint(cfg_effect.bass.freq_min, cfg_effect.bass.freq_max),  # Random frequency
                        np.random.uniform(cfg_effect.bass.db_min, cfg_effect.bass.db_max),  # Random gain (dB)
                        format(np.random.uniform(0.1, 1), '.2f')  # Random width factor (between 0.1 and 1)
                    )
                else:  # Cut low frequencies
                    sox_effect_transform.add_bass(
                        np.random.randint(cfg_effect.bass.freq_min, cfg_effect.bass.freq_max),  # Random frequency
                        np.random.uniform(-cfg_effect.bass.db_max, -cfg_effect.bass.db_min),  # Random attenuation
                        format(np.random.uniform(0.1, 1), '.2f')  # Random width factor (between 0.1 and 1)
                    )
            # OVERDRIVE EFFECT: Adds distortion to the audio
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.overdrive.proba and len(sox_effect_transform.effects) < max_effect and not overdrive:
                overdrive = True  # Mark as applied so we don't add it again
                
                # Add overdrive effect with random gain and color parameters
                sox_effect_transform.add_overdrive(
                    np.random.randint(cfg_effect.overdrive.min_int, cfg_effect.overdrive.max_int),  # Random gain (0-100)
                    np.random.randint(0, 15)  # Random color/tone of distortion (0-15)
                )

            # REVERB EFFECT: Adds echo/room ambience to the audio
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.reverb.proba and len(sox_effect_transform.effects) < max_effect and not reverb:
                reverb = True  # Mark as applied so we don't add it again
                
                # Determine whether to use -w flag (weighted algorithm)
                # Note: There seems to be a bug here as both branches set w_true to False
                if np.random.random() < cfg_effect.reverb.proba_w:
                    w_true = False  # Should probably be True, but keeping original behavior
                else: 
                    w_true = False

                # Generate random parameters for the reverb effect
                reverberance = np.random.randint(1, 100)  # Amount of reverb (percentage)
                damping = np.random.randint(1, 100)       # Higher values = more high frequency absorption
                room_scale = np.random.randint(1, 100)    # Room size (percentage)
                stereo_depth = np.random.randint(1, 100)  # Stereo spread (percentage)
                
                # Determine if we should add pre-delay
                if cfg_effect.reverb.proba_delay:
                    delay = np.random.randint(1, 500)  # Random delay in milliseconds
                else:
                    delay = 0  # No pre-delay
                    
                wet_gain = np.random.uniform(-10, 10)  # Wet/reverb signal gain in dB
                
                # Build the command string with all parameters
                if w_true:  # Use -w flag (weighted algorithm)
                    params_string = f"reverb -w {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                else:  # Standard algorithm
                    params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                    
                # Add the reverb effect and convert to mono afterward
                sox_effect_transform.add_effect(params_string.strip().split(" "))
                sox_effect_transform.add_effect('channels 1'.strip().split(" "))
            
            # FLANGER EFFECT: Delayed copy of signal mixed with original, creates swooshing effect
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.flanger.proba and len(sox_effect_transform.effects) < max_effect and not flanger:
                flanger = True  # Mark as applied so we don't add it again
                
                # Add flanger effect with default parameters
                params_string = 'flanger'  # Using default SoX flanger parameters
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # RIAA EFFECT: Record Industry Association of America equalization curve
            # Simulates vinyl record equalization - applies with probability from config, only once per transform
            if np.random.random() < cfg_effect.riaa.proba and len(sox_effect_transform.effects) < max_effect and not riaa:
                riaa = True  # Mark as applied so we don't add it again
                
                # Add RIAA effect with default parameters
                params_string = 'riaa'  # RIAA equalization curve simulation
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # HILBERT EFFECT: Performs a Hilbert transform - shifts phase by 90 degrees
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.hilbert.proba and len(sox_effect_transform.effects) < max_effect and not hilbert:
                hilbert = True  # Mark as applied so we don't add it again
                
                # Add Hilbert transform effect
                params_string = 'hilbert'  # Creates phase-shifted version of the signal
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # BAND EFFECT: Band-pass filter to isolate a frequency range
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.band.proba and len(sox_effect_transform.effects) < max_effect and not band:
                band = True  # Mark as applied so we don't add it again
                
                # Generate random parameters for the band-pass filter
                width = np.random.uniform(100, 3000)  # Width of the frequency band in Hz
                center = np.random.uniform(cfg_effect.band.min_freq, cfg_effect.band.max_freq)  # Center frequency in Hz
                
                # Determine whether to use noise-based filter (-n flag)
                if np.random.random() < cfg_effect.band.noise_proba:
                    params_string = f"band -n {center} {width}"  # Noise-based band-pass filter
                else:
                    params_string = f"band {center} {width}"  # Standard band-pass filter
                    
                # Add the band effect
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # SINC EFFECT: Sinc filter (brick-wall filter - sharp cutoff)
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.sinc.proba and len(sox_effect_transform.effects) < max_effect and not sinc:
                sinc = True  # Mark as applied so we don't add it again
                
                # Generate random parameters for the sinc filter
                att = np.random.uniform(40, cfg_effect.sinc.att_max)  # Attenuation in dB for frequencies beyond cutoff
                sign = np.random.choice([-1, 1])  # Random sign to determine high-pass (-1) or low-pass (1) filtering
                freq = np.random.uniform(cfg_effect.sinc.min_freq, cfg_effect.sinc.max_freq)  # Cutoff frequency in Hz
                
                # Build the sinc filter command
                params_string = f"sinc -a {att} {sign * freq} "  # -a specifies attenuation
                # If sign is positive: low-pass filter (keeps frequencies below freq)
                # If sign is negative: high-pass filter (keeps frequencies above freq)
                
                # Add the sinc effect
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # TREMOLO EFFECT (currently disabled): Modulates amplitude to create trembling sound
            # This section is commented out but would apply tremolo effect if enabled
            # if np.random.random() < cfg_effect.tremolo.proba and len(sox.effects) < max_effect and not tremolo:
            #     tremolo = True  # Mark as applied so we don't add it again
            #     speed = np.random.uniform(cfg_effect.tremolo.speed_min, cfg_effect.tremolo.speed_max)  # Modulation speed in Hz
            #     depth = np.random.uniform(cfg_effect.tremolo.depth_min, cfg_effect.tremolo.depth_max)  # Modulation depth (0-1)
            #     params_string = f"tremolo {speed} {depth}"
            #     sox.add_effect(params_string.strip().split(" "))

            # ECHO EFFECT: Adds echoes to the audio signal
            # Applies with probability specified in config, only once per transform
            if np.random.random() < cfg_effect.echo.proba and len(sox_effect_transform.effects) < max_effect and not echo:
                # Generate random parameters for input and output gain
                gain_in = np.random.uniform(cfg_effect.echo.gain_in_min, cfg_effect.echo.gain_in_max)   # Input volume
                gain_out = np.random.uniform(cfg_effect.echo.gain_out_min, cfg_effect.echo.gain_out_max) # Output volume
                
                # Initialize lists to store delay and decay values for multiple echoes
                delay = []  # Time delays for echoes in milliseconds
                decay = []  # Decay factors for each echo
                
                # Start building the echo command
                params_string = f"echos {gain_in} {gain_out} "  # 'echos' allows multiple echoes
                
                # Add echoes randomly based on probability configuration
                while (not echo) or cfg_effect.echo.proba_next_echo > np.random.random():
                    echo = True  # Mark as applied so we don't use this effect again
                    
                    # Generate random delay and decay values
                    delay_value = np.random.uniform(cfg_effect.echo.delay_min, cfg_effect.echo.delay_max)  # Echo delay
                    delay.append(delay_value)
                    
                    decay_value = np.random.uniform(cfg_effect.echo.decay_min, cfg_effect.echo.decay_max)  # Echo decay
                    decay.append(decay_value)
                    
                # Add only the first echo to the parameter string
                # Note: This appears to be a bug - only using the first echo despite collecting multiple
                for i, _ in enumerate(delay):
                    if i == 0:  # Only using the first echo pair
                        params_string += f"{delay[i]} {decay[i]} "               
                        
                # Add the echo effect
                sox_effect_transform.add_effect(params_string.strip().split(" "))
              
        # Normalize gain after all effects if specified in config
        # This prevents clipping and ensures consistent volume levels
        if cfg.dataset.normalize_inputs_post_eq:
            sox_effect_transform.normalize_gain()
            
        # Return the complete transform with all random effects applied
        return sox_effect_transform

    def to_mono(self, prepend: bool = True) -> SoxEffectTransform:
        """If prepend is True, the first effect will transform the audio to mono and then apply the effects"""
        effect = ["remix", "-"]
        if prepend:
            self.effects.insert(0, effect)
        else:
            self.effects.append(effect)
        return self

    def add_equalizer(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        Apply a two-pole peaking equalisation (EQ) filter. With this filter, the signal-level at and around a selected frequency can be increased or decreased, whilst (unlike band-pass and band-reject filters) that at all other frequencies is unchanged.
        frequency gives the filter's central frequency in Hz, width, the band-width, and gain the required gain or attenuation in dB. Beware of Clipping when using a positive gain

        Taken from: https://howtoeq.wordpress.com/2010/10/07/q-factor-and-bandwidth-in-eq-what-it-all-means/
        Q factor (float) controls the bandwidth—or number of frequencies—that will be cut or boosted by the equaliser. The lower the Q factor, the wider the bandwidth (and the more frequencies will be affected).
        The higher the Q factor, the narrower the bandwidth (and the fewer frequencies will be affected).
        Q-factor
        0.7  = 2 octaves
        1    = 1 1/3 octaves
        1.4  = 1 octave
        2.8  = 1/2 octave
        4.3  = 1/3 octave
        8.6  = 1/6 octave
        """
        effect = ["equalizer", str(center_freq), str(Q), str(gain)]
        self.effects.append(effect)
        return self

    def add_overdrive(self, gain: float, colour: float) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [0 to 100]
        colour	(float): controls the amount of even harmonic content in the over-driven output [0, 100]
        """
        effect = ["overdrive", str(gain), str(colour)]
        self.effects.append(effect)
        return self

    def add_treble(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [-20 to 20]
        Q factor (float) controls the bandwidth—or number of frequencies—that will be impacted
        """
        # Ensure parameters are within safe ranges for SoX
        gain = max(-10, min(10, gain))  # Limit gain to very safe range
        
        # Use the simplest form of the treble effect which is more reliable
        # Format: treble gain
        effect = ['treble', str(gain)]
        self.effects.append(effect)
        
        LOG.debug(f"Using simplified treble effect: {effect}")
        return self

    def add_bass(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [-100 to 100]
        Q factor (float) controls the bandwidth—or number of frequencies—that will be impacted
        """
        # The sox bass effect expects: bass gain(dB) [frequency(Hz) [width_q]]
        # Ensure frequency is within the acceptable range for sox (usually 10-1000Hz for bass)
        center_freq = max(10, min(1000, center_freq))
        effect = ['bass', str(gain), str(center_freq), str(Q)]
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

    def add_gain(self, gain: float = 0) -> SoxEffectTransform:
        effect = ["gain", str(gain)]
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

    def apply_file(self, filepath: str) -> Tuple[Tensor, int]:
        return sox.apply_effects_file(filepath, self.effects, channels_first=True)

    def process_file(self, input_filepath: str, output_folder: str, override: bool = False) -> str:
        """Apply effect directly on an audio file and creates the transformed version by
        using the original name and self.name returns the filepath of the transformed file
        if transformed file already exists skip unless override is True
        """
        eq_name = self.name
        if not eq_name:
            eq_name = "eq_def"
        inpath = Path(input_filepath).expanduser().resolve()
        filename = str(inpath.stem)
        ext = inpath.suffix

        # Create folder if needed
        outfolder = Path(output_folder).expanduser().resolve()
        outfolder.mkdir(parents=True, exist_ok=True)

        output_path = outfolder / (filename + "_" + eq_name + ext)
        if output_path.exists() and not override:
            LOG.info(f"File exist, skipping: {output_path}")
            return str(output_path)

        waveform, sr = self.apply_file(inpath)
        torchaudio.save(output_path, waveform, sr)
        LOG.info(f"Wrote: {output_path}")
        return str(output_path)