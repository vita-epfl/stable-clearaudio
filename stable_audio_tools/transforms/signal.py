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
import yaml
from omegaconf import OmegaConf

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

def apply_config_to_audio(info, audio, low_quality_effects_dir, effects_file=None):
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
        # Conversion des chemins relatifs en chemins absolus
        if low_quality_effects_dir.startswith("/") and not os.path.exists(low_quality_effects_dir):
            # Obtenir le chemin de base du projet
            base_path = Path(__file__).resolve().parent.parent.parent
            
            if low_quality_effects_dir.startswith("/stable-clearaudio/"):
                relative_path = low_quality_effects_dir[len("/stable-clearaudio/"):]
                low_quality_effects_dir = str(base_path / relative_path)
            elif low_quality_effects_dir.startswith("/stable_audio_tools/"):
                relative_path = low_quality_effects_dir[len("/stable_audio_tools/"):]
                low_quality_effects_dir = str(base_path / "stable_audio_tools" / relative_path)
            
            LOG.debug(f"Chemin résolu: {low_quality_effects_dir}")
            
        # Construire le chemin final
        effects_config_path = os.path.join(
            low_quality_effects_dir,
            effects_file + ".yaml",
        )
        LOG.debug(f"Applying effects from {effects_config_path}")

        # Check if the configuration file exists
        if not effects_config_path or not Path(effects_config_path).exists():
            LOG.error(f"Configuration file not found: {effects_config_path}")
            return audio

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
            transforms = SoxEffectTransform.from_config(config, low_quality_effect_files)
            
            if transforms:
                # Apply each transformation
                for transform in transforms:
                    LOG.debug(f"Applying SoX transformation: {transform.name}")
                    LOG.debug(f"Effects to apply: {transform.effects}")

                    # Apply the transformation
                    processed_wave, processed_sr = transform.apply_tensor(
                        audio, info.sample_rate
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
                    audio = processed_wave
                    info.sample_rate = processed_sr
                    info.num_samples = processed_wave.shape[1]
                    
                    LOG.debug(f"SoX effects applied successfully. New shape: {processed_wave.shape}")
            else:
                LOG.warning(f"No SoX effect transformations found in configuration {effects_config_path}")
                
            # Process external sounds if any
            if "external_sounds" in config_raw:
                LOG.debug("External sounds found in configuration. Adding...")

                # Create external sound transformations
                transforms = ExternalSoundTransform.from_config(
                    config, effects_file
                )

                if transforms:
                    # Apply each transformation
                    for transform in transforms:
                        LOG.debug(f"Applying external sound: {transform.sound_name}")

                        # Apply the transformation
                        processed_wave, processed_sr = transform.apply_external_sound(
                            audio, info.sample_rate
                        )

                        # Update the audio clip
                        audio = processed_wave
                        info.sample_rate = processed_sr

                        LOG.debug(f"External sound added successfully: {transform.sound_name}")
                else:
                    LOG.error(
                        f"No external sound transformation found in configuration {effects_config_path}"
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
    def from_config(cfg: Any, external_sounds_cfg_name: str) -> List[ExternalSoundTransform]:
        external_sounds_cfg = cfg.dataset.low_quality_effect[external_sounds_cfg_name]
        transforms = []
        for sound in external_sounds_cfg['external_sounds']:
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
                LOG.debug(f"Resampling external sound from {external_sr}Hz to {sample_rate}Hz")
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
    def from_config(cfg: Any, eq_cfg_name: Any) -> Dict[str, List[SoxEffectTransform]]:
        """
        From a configuration dictionary, create a list of SoxEffectTransform objects.

        Args:
            cfg: Configuration dictionary
            eq_cfgs: Dictionary of effect configurations
        
        Returns:
            Dictionary of effect names and their corresponding SoxEffectTransform objects
        """
        transforms = []
        try:
            LOG.debug(f"Creating SoxEffectTransform objects from configuration: {cfg}")
            # Check if eq_cfg exists in low_quality_effect
            if eq_cfg_name not in cfg.dataset.low_quality_effect:
                LOG.error(f"Configuration '{eq_cfg_name}' not found in cfg.dataset.low_quality_effect")
                return []
            
            # Check if transform_type exists
            if 'transform_type' not in cfg.dataset.low_quality_effect[eq_cfg_name]:
                LOG.error(f"'transform_type' not found in cfg.dataset.low_quality_effect[{eq_cfg_name}]")
                return []
                
            # Check transform type
            if cfg.dataset.low_quality_effect[eq_cfg_name]['transform_type'] == 'sox_audio_effect':
                # Check if effects exists
                if 'effects' not in cfg.dataset.low_quality_effect[eq_cfg_name]:
                    LOG.error(f"'effects' not found in cfg.dataset.low_quality_effect[{eq_cfg_name}]")
                    return []
                                        
                for effect in cfg.dataset.low_quality_effect[eq_cfg_name]['effects']:
                    sox_effect_transform = SoxEffectTransform(effect.name)
                    
                    # Check if params exists
                    if not hasattr(effect, 'params'):
                        LOG.error(f"'params' not found in effect {effect.name}")
                        continue
                        
                    for param_string in effect.params:
                        # si sox ne connait pas le nom de l'effet il renvoie une erreur
                        if param_string.split(" ")[0] not in sox.effect_names():
                            LOG.error(f"Sox does not recognize the effect: {param_string.split(' ')[0]} in {effect.name}")
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
                    if hasattr(effect, 'gain'):
                        effect_gain = effect.gain
                        LOG.debug(f"Found effect-level gain parameter for {effect.name}: {effect_gain} (will be applied separately, not added to SoX effects)")
                        sox_effect_transform.original_gain = float(effect_gain) # Store gain here
                        # sox_effect_transform.add_gain(float(effect_gain)) # Don't add gain here, it's handled later
                    elif cfg.dataset.normalize_inputs_post_eq:
                        sox_effect_transform.normalize_gain()
                        
                    transforms.append(sox_effect_transform)
            else:
                LOG.error(f"Transformation type '{cfg.dataset.low_quality_effect[eq_cfg_name]['transform_type']}' is not supported. Only sox_audio_effect is supported for now.")
                return []
        except Exception as e:
            LOG.error(f"Error in from_config for eq_cfg={eq_cfg_name}: {str(e)}")
            return []

        return transforms

    @staticmethod
    def from_mode_config(cfg: Any) -> Dict[str, List[SoxEffectTransform]]:
        transforms = {}
        for mode in ['train','test','validation']:
            if cfg.dataset.low_quality_effect[mode]:
                transforms[mode] = []
                for effect in cfg.dataset.low_quality_effect[mode]['effects']:
                    sox_effect_transform = SoxEffectTransform(effect.name)
                    for param_string in effect.params:
                        params = param_string.strip().split(" ")
                        sox_effect_transform.add_effect(params)
                    if cfg.dataset.normalize_inputs_post_eq:
                        sox_effect_transform.normalize_gain()
                    transforms[mode].append(sox_effect_transform)
        return transforms


    @staticmethod
    def random_effects(cfg: Any) -> SoxEffectTransform:
        sox_effect_transform = SoxEffectTransform("random")
        cfg_effect = cfg.dataset.low_quality_effect['random']
        max_effect = cfg_effect.random_effects.max_effect
        min_effect = cfg_effect.random_effects.min_effect
        eq_freq = []
        treble = False
        bass = False
        overdrive = False
        reverb = False
        riaa = False
        echo = False
        band = False
        tremolo = False
        sinc = False
        hilbert = False
        flanger = False

        while len(sox_effect_transform.effects) < min_effect and len(sox_effect_transform.effects) < max_effect:
            
            #Equalizer
            while np.random.random() < cfg_effect.equalizer.proba and len(sox_effect_transform.effects) < max_effect:
                center_freq = cfg_effect.equalizer.freq_mean
                std_freq = cfg_effect.equalizer.freq_std
                for freq in eq_freq:
                    if np.abs(freq-center_freq) < 20:
                        continue
                eq_freq.append(center_freq)
                if np.random.random() > 0.5:
                    sox_effect_transform.add_equalizer(max(np.random.normal(center_freq, std_freq), 150), 
                        np.random.uniform(cfg_effect.equalizer.db_min,cfg_effect.equalizer.db_max), 
                        np.random.uniform(0.7, 5)
                    )
                else:
                    sox_effect_transform.add_equalizer(max(np.random.normal(center_freq, std_freq), 150), 
                        np.random.uniform(-cfg_effect.equalizer.db_max,-cfg_effect.equalizer.db_min), 
                        np.random.uniform(0.7, 5)
                    )
            #Treble
            if np.random.random() < cfg_effect.treble.proba and len(sox_effect_transform.effects) < max_effect and not treble:
                treble = True
                if np.random.random() > 0.5:
                    sox_effect_transform.add_treble(np.random.randint(cfg_effect.treble.freq_min, cfg_effect.treble.freq_max), 
                        np.random.uniform(cfg_effect.treble.db_min,cfg_effect.treble.db_max), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
                else:
                    sox_effect_transform.add_treble(np.random.randint(cfg_effect.treble.freq_min, cfg_effect.treble.freq_max), 
                        np.random.uniform(-cfg_effect.treble.db_max,cfg_effect.treble.db_min), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
            #Bass
            if np.random.random() < cfg_effect.bass.proba and len(sox_effect_transform.effects) < max_effect and not bass:
                bass = True
                if np.random.random() > 0.5:
                    sox_effect_transform.add_bass(np.random.randint(cfg_effect.bass.freq_min, cfg_effect.bass.freq_max),
                        np.random.uniform(cfg_effect.bass.db_min,cfg_effect.bass.db_max), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
                else:
                    sox_effect_transform.add_bass(np.random.randint(cfg_effect.bass.freq_min, cfg_effect.bass.freq_max),
                        np.random.uniform(-cfg_effect.bass.db_max,-cfg_effect.bass.db_min), 
                        format(np.random.uniform(0.1, 1), '.2f')
                    )
            #Overdrive
            if np.random.random() < cfg_effect.overdrive.proba and len(sox_effect_transform.effects) < max_effect and not overdrive:
                overdrive = True
                sox_effect_transform.add_overdrive(np.random.randint(cfg_effect.overdrive.min_int, cfg_effect.overdrive.max_int), 
                    np.random.randint(0, 15)
                )

            #Reverb
            if np.random.random() < cfg_effect.reverb.proba and len(sox_effect_transform.effects) < max_effect and not reverb:
                reverb = True
                if np.random.random() < cfg_effect.reverb.proba_w:
                    w_true = False
                else: 
                    w_true = False

                reverberance = np.random.randint(1, 100)
                damping = np.random.randint(1, 100)
                room_scale = np.random.randint(1, 100)
                stereo_depth = np.random.randint(1, 100)
                if cfg_effect.reverb.proba_delay:
                    delay = np.random.randint(1, 500) #in millisecond
                else:
                    delay = 0
                wet_gain = np.random.uniform(-10, 10)
                if w_true:
                    params_string = f"reverb -w {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                else:
                    params_string = f"reverb {reverberance} {damping} {room_scale} {stereo_depth} {delay} {wet_gain}"
                sox_effect_transform.add_effect(params_string.strip().split(" "))
                sox_effect_transform.add_effect('channels 1'.strip().split(" "))
            
            # Flanger
            if np.random.random() < cfg_effect.flanger.proba and len(sox_effect_transform.effects) < max_effect and not flanger:
                flanger = True
                params_string = 'flanger'
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # riaa
            if np.random.random() < cfg_effect.riaa.proba and len(sox_effect_transform.effects) < max_effect and not riaa:
                riaa = True
                params_string = 'riaa'
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # hilbert
            if np.random.random() < cfg_effect.hilbert.proba and len(sox_effect_transform.effects) < max_effect and not hilbert:
                hilbert = True
                params_string = 'hilbert'
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            #band
            if np.random.random() < cfg_effect.band.proba and len(sox_effect_transform.effects) < max_effect and not band:
                band = True
                width = np.random.uniform(100,3000)
                center = np.random.uniform(cfg_effect.band.min_freq, cfg_effect.band.max_freq)
                if np.random.random() < cfg_effect.band.noise_proba:
                    params_string = f"band -n {center} {width}"
                else :
                    params_string = f"band {center} {width}"
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            #sinc
            if np.random.random() < cfg_effect.sinc.proba and len(sox_effect_transform.effects) < max_effect and not sinc:
                sinc = True
                att = np.random.uniform(40, cfg_effect.sinc.att_max)
                sign = np.random.choice([-1, 1])
                freq = np.random.uniform(cfg_effect.sinc.min_freq, cfg_effect.sinc.max_freq)
                params_string = f"sinc -a {att} {sign * freq} "
                sox_effect_transform.add_effect(params_string.strip().split(" "))

            # #tremolo
            # if np.random.random() < cfg_effect.tremolo.proba and len(sox.effects) < max_effect and not tremolo:
            #     tremolo = True
            #     speed = np.random.uniform(cfg_effect.tremolo.speed_min, cfg_effect.tremolo.speed_max)
            #     depth = np.random.uniform(cfg_effect.tremolo.depth_min, cfg_effect.tremolo.depth_max)
            #     params_string = f"tremolo {speed} {depth}"
            #     sox.add_effect(params_string.strip().split(" "))

            #echo
            if np.random.random() < cfg_effect.echo.proba and len(sox_effect_transform.effects) < max_effect and not echo:
                gain_in = np.random.uniform(cfg_effect.echo.gain_in_min, cfg_effect.echo.gain_in_max)
                gain_out = np.random.uniform(cfg_effect.echo.gain_out_min, cfg_effect.echo.gain_out_max)
                delay = []
                decay = []
                params_string = f"echos {gain_in} {gain_out} "
                while (not echo) or cfg_effect.echo.proba_next_echo > np.random.random():
                    echo = True
                    delay_value = np.random.uniform(cfg_effect.echo.delay_min, cfg_effect.echo.delay_max)
                    delay.append(delay_value)
                    decay_value = np.random.uniform(cfg_effect.echo.decay_min, cfg_effect.echo.decay_max)
                    decay.append(decay_value)
                for i, _ in enumerate(delay):
                    if i == 0 :  
                        params_string += f"{delay[i]} {decay[i]} "               
                sox_effect_transform.add_effect(params_string.strip().split(" "))
              
        if cfg.dataset.normalize_inputs_post_eq:
            sox_effect_transform.normalize_gain()
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
        gain (float) desired gain at the boost (or attenuation) in dB [-100 to 100]
        Q factor (float) controls the bandwidth—or number of frequencies—that will be impacted
        """
        effect = ['treble', str(gain), str(center_freq), str(Q)]
        self.effects.append(effect)
        return self

    def add_bass(self, center_freq: float, gain: float, Q: float = 0.707) -> SoxEffectTransform:
        """
        gain (float) desired gain at the boost (or attenuation) in dB [-100 to 100]
        Q factor (float) controls the bandwidth—or number of frequencies—that will be impacted
        """
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
        LOG.debug(f"Applying effects with SoxEffectTransform: {self.name}")
        LOG.debug(f"Effects to apply: {self.effects}")
        
        # Ne pas normaliser l'entrée pour préserver les effets de gain
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