import gc
import numpy as np
import re
import io
import os
import gc
import time
import glob
import json
import shutil
import torch
import gradio as gr
import logging
import torchaudio
import threading
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
# Suppress verbose matplotlib debug logs
import logging as _logging
_logging.getLogger('matplotlib').setLevel(_logging.WARNING)
_logging.getLogger('matplotlib.font_manager').setLevel(_logging.WARNING)
import matplotlib.pyplot as plt
from io import BytesIO

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..aeiou import audio_spectrogram_image
from ...inference.generation import (
    generate_diffusion_cond_restoration, generate_cold_diffusion_uncond_restoration
)  # , generate_diffusion_uncond

# from ..models.factory import create_model_from_config
# from ..models.pretrained import get_pretrained_model
# from ..models.utils import load_ckpt_state_dict
from ...inference.utils import prepare_audio
# from ..training.utils import copy_state_dict

import logging

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = True


# when using a prompt in a filename
def condense_prompt(prompt):
    pattern = r'[\\/:*?"<>|]'
    # Replace special characters with hyphens
    prompt = re.sub(pattern, "-", prompt)
    # set a character limit
    prompt = prompt[:150]
    # zero length prompts may lead to filenames (ie ".wav") which seem cause problems with gradio
    if len(prompt) == 0:
        prompt = "_"
    return prompt


def generate_restoration(
    degraded_audio,
    steps=250,
    preview_every=None,
    metrics_every=0,
    seed=-1,
    sampler_type="dpmpp-3m-sde",
    sigma_min=0.03,
    sigma_max=1000,
    rho=1.0,
    cfg_rescale=0.0,
    file_format="wav",
    clean_audio=None,
    effects_list=None,
    batch_size=1,
    degraded_audio_filename=None,
    custom_output_dir=None,
    t_start=1.0,
):
    LOG.info("Starting audio restoration")
    
    # Access global variables
    global sample_rate, model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)
    # if seed is -1, define the seed value now, randomly, so we can save it in the filename
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    LOG.info(f"Using seed: {seed}")

    input_sample_size = degraded_audio[1].shape[-1]

    if degraded_audio is not None:
        if not isinstance(degraded_audio, tuple) or len(degraded_audio) != 2:
            raise ValueError(f"Invalid audio format: {degraded_audio}. Expected a tuple of (sample_rate, audio_data).")

        in_sr, degraded_audio = degraded_audio

        if degraded_audio.dtype == np.float32:
            degraded_audio = torch.from_numpy(degraded_audio)
        elif degraded_audio.dtype == np.int16:
            degraded_audio = torch.from_numpy(degraded_audio).float().div(32767)
        elif degraded_audio.dtype == np.int32:
            degraded_audio = torch.from_numpy(degraded_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {degraded_audio.dtype}")

        if model_half:
            degraded_audio = degraded_audio.to(torch.float16)

        if degraded_audio.dim() == 1:
            degraded_audio = degraded_audio.unsqueeze(0)

        if in_sr != sample_rate:
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(degraded_audio.device)
                .to(degraded_audio.dtype)
            )
            degraded_audio = resample_tf(degraded_audio)

        audio_length = degraded_audio.shape[-1]

        if audio_length > input_sample_size:
            degraded_audio = degraded_audio[:, :input_sample_size]

        if degraded_audio.shape[0] == 1:
            degraded_audio = degraded_audio.repeat(2, 1)
        
        degraded_audio = (sample_rate, degraded_audio)

    # Handle clean audio if provided
    if clean_audio is not None:
        in_sr, clean_audio = clean_audio
        
        if clean_audio.dtype == np.float32:
            clean_audio = torch.from_numpy(clean_audio)
        elif clean_audio.dtype == np.int16:
            clean_audio = torch.from_numpy(clean_audio).float().div(32767)
        elif clean_audio.dtype == np.int32:
            clean_audio = torch.from_numpy(clean_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {clean_audio.dtype}")

        if model_half:
            clean_audio = clean_audio.to(torch.float16)

        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)
        elif clean_audio.dim() == 2:
            clean_audio = clean_audio.transpose(0, 1)

        if in_sr != sample_rate:
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(clean_audio.device)
                .to(clean_audio.dtype)
            )
            clean_audio = resample_tf(clean_audio)

        audio_length = clean_audio.shape[-1]
        if audio_length > input_sample_size:
            clean_audio = clean_audio[:, :input_sample_size]
        
        if clean_audio.shape[0] == 1:
            clean_audio = clean_audio.repeat(2, 1)
        
        clean_audio = (sample_rate, clean_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if preview_every is not None and (current_step) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(
                denoised, sample_rate=sample_rate
            )
            preview_images.append(
                (audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})")
            )

    # Check if we should skip creating a date-based folder
    skip_date_folder = os.environ.get("STABLE_AUDIO_NO_DATE_FOLDER", "0") == "1"

    if custom_output_dir is not None:
        output_dir = custom_output_dir
    elif skip_date_folder:
        # Utiliser le dossier temporaire personnalisé si spécifié
        custom_tmp_dir = os.environ.get("STABLE_AUDIO_CUSTOM_TMP_DIR")
        if custom_tmp_dir:
            output_dir = custom_tmp_dir
        else:
            # Utiliser le dossier batch_processing pour éviter de créer des dossiers temporaires
            output_dir = os.path.join("output", "batch_processing", "tmp")
            os.makedirs(output_dir, exist_ok=True)
    else:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join("output", date_string, "degradation_processing")

    os.makedirs(output_dir, exist_ok=True)
    
    # Use a dedicated folder for all degradation processing
    generation_dir = os.path.join(output_dir, "degradation_processing")
    os.makedirs(generation_dir, exist_ok=True)
    LOG.info(f"Files will be saved in directory: {generation_dir}")

    # Do the audio generation
    LOG.info("Generating audio")
    
    if model_type == "diffusion_cond_restoration":
        conditioning_dict = {
            "degraded_audio": degraded_audio[1] if degraded_audio is not None else None,
        }
        conditioning = [conditioning_dict] * batch_size

        generate_args = {
            "model": model,
            "conditioning": conditioning,
            "steps": steps,
            "batch_size": batch_size,
            "sample_size": input_sample_size,
            "seed": seed,
            "device": device,
            "sampler_type": sampler_type,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "callback": progress_callback if (preview_every is not None) else None,
            "scale_phi": cfg_rescale,
            "rho": rho,
            "clean_audio": clean_audio,
            "output_dir": generation_dir,
            "metrics_every": metrics_every
        }
        audio, final_metrics = generate_diffusion_cond_restoration(**generate_args)
    elif model_type == "cold_diffusion_uncond_restoration":
        generate_args = {
            "model": model,
            "steps": steps,
            "batch_size": batch_size,
            "sample_size": input_sample_size,
            "seed": seed,
            "device": device,
            "callback": progress_callback if (preview_every is not None) else None,
            "clean_audio": clean_audio,
            "degraded_audio": degraded_audio,
            "effects_list": effects_list,
            "output_dir": generation_dir,
            "metrics_every": metrics_every,
            "t_start": t_start
        }
        audio, final_metrics = generate_cold_diffusion_uncond_restoration(**generate_args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Prepare the file names to avoid reference errors
    output_wav = os.path.join(generation_dir, "output.wav")
    filename_extension = file_format.split(" ")[0].lower() if file_format else "wav"
    output_filename = os.path.join(generation_dir, f"output.{filename_extension}")

    # Combine per-step metrics with final metrics
    if final_metrics is not None:
        final_metrics["generation_params"] = {
            "steps": steps,
            "metrics_every": metrics_every,
            "preview_every": preview_every,
        }
        
        try:
            # Define NumpyEncoder for JSON serialization
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            # Use the provided filename if available
            if degraded_audio_filename is not None:
                # Use the filename that was passed as a parameter
                metrics_filename = f"{os.path.splitext(os.path.basename(degraded_audio_filename))[0]}_metrics.json"
                LOG.info(f"Using provided filename for metrics: {metrics_filename}")
            else:
                metrics_filename = "detailed_metrics.json"
                LOG.info("No filename provided for metrics, using default: detailed_metrics.json")
            
            # Save detailed metrics file only
            detailed_metrics_file = os.path.join(generation_dir, metrics_filename)
            
            # Create a new detailed metrics file with degraded and restored metrics
            detailed_metrics = {}
            
            # Get degraded audio path
            degraded_audio_path = os.path.join(generation_dir, "degraded_audio.wav")
            degraded_audio_path_relative = degraded_audio_path.split('clean_audio_pairs/')[-1] if 'clean_audio_pairs/' in degraded_audio_path else degraded_audio_path
            
            # Create a dictionary for degraded metrics
            degraded_losses = {
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "steps": 0
            }
            
            # Handle sample_rate and sample_size specially to ensure they're single values
            metric_sample_rate = final_metrics.get("sample_rate", sample_rate)
            if isinstance(metric_sample_rate, list) and len(metric_sample_rate) > 0:
                metric_sample_rate = metric_sample_rate[0]
            degraded_losses["sample_rate"] = metric_sample_rate
            
            metric_sample_size = final_metrics.get("sample_size", sample_size)
            if isinstance(metric_sample_size, list) and len(metric_sample_size) > 0:
                metric_sample_size = metric_sample_size[0]
            degraded_losses["sample_size"] = metric_sample_size
            
            # Extract only metrics with the 'degraded_' prefix but remove the prefix
            for key in final_metrics:
                if key.startswith("degraded_"):
                    # Remove the 'degraded_' prefix to get a clean key name
                    clean_key = key[len("degraded_"):]  # This removes 'degraded_' prefix
                    # Ensure we're storing just a single value
                    value = final_metrics[key]
                    if isinstance(value, list) and len(value) > 0:
                        value = value[0]  # Take the first value if it's a list
                    degraded_losses[clean_key] = value
            
            detailed_metrics["degraded"] = {
                "losses": degraded_losses,
                "audio": degraded_audio_path_relative
            }
            
            # Get restored audio path
            restored_audio_path = output_filename if file_format != "wav" else output_wav
            restored_audio_path_relative = restored_audio_path.split('clean_audio_pairs/')[-1] if 'clean_audio_pairs/' in restored_audio_path else restored_audio_path
            
            # Create a dictionary for restored metrics
            restored_losses = {}
            
            # Handle sample_rate and sample_size specially to ensure they're single values
            metric_sample_rate = final_metrics.get("sample_rate", sample_rate)
            if isinstance(metric_sample_rate, list) and len(metric_sample_rate) > 0:
                metric_sample_rate = metric_sample_rate[0]
            restored_losses["sample_rate"] = metric_sample_rate
            
            metric_sample_size = final_metrics.get("sample_size", sample_size)
            if isinstance(metric_sample_size, list) and len(metric_sample_size) > 0:
                metric_sample_size = metric_sample_size[0]
            restored_losses["sample_size"] = metric_sample_size
            
            # Add timestamp
            restored_losses["timestamp"] = time.strftime("%Y-%m-%d_%H-%M-%S")
            
            # Only copy demo metrics and other relevant metrics with simplified keys
            for key, value in final_metrics.items():
                # Process demo metrics - remove the prefix
                if key.startswith("demo_"):
                    clean_key = key[len("demo_"):]  # Remove 'demo_' prefix
                    # Ensure single value
                    if isinstance(value, list) and len(value) > 0:
                        value = value[0]  # Take first value if it's a list
                    restored_losses[clean_key] = value
                # Process restoration success metrics - remove the prefix
                elif key.startswith("restoration_success_"):
                    clean_key = f"restoration_{key[len('restoration_success_'):]}"
                    if isinstance(value, list) and len(value) > 0:
                        value = value[0]  # Take first value if it's a list
                    restored_losses[clean_key] = value
                # Include other relevant metrics (not prefixed, not degraded, not dicts)
                elif (key not in ["generation_params"] and 
                      not key.startswith("degraded_") and 
                      not isinstance(value, dict)):
                    if isinstance(value, list) and len(value) > 0:
                        value = value[0]  # Take first value if it's a list
                    restored_losses[key] = value
            
            detailed_metrics["restored"] = {
                "losses": restored_losses,
                "audio": restored_audio_path_relative
            }
            
            # Save the detailed metrics file
            with open(detailed_metrics_file, 'w') as f:
                json.dump(detailed_metrics, f, indent=4, cls=NumpyEncoder)
            LOG.info(f"Detailed metrics saved to {detailed_metrics_file}")
            
        except Exception as e:
            LOG.error(f"Error saving metrics to file: {e}")
            import traceback
            LOG.error(traceback.format_exc())

    # File names are already defined above

    audio = rearrange(audio, "b d n -> d (b n)")

    # Check if the audio tensor is empty before normalization
    if audio.numel() == 0:
        LOG.warning("Generated audio is empty")
        return (None, preview_images, final_metrics)

    # If audio is not empty, proceed with normalization
    max_abs_val = torch.max(torch.abs(audio))
    if max_abs_val > 1e-7:
        audio = (
            audio.to(torch.float32)
            .div(max_abs_val)
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
    else:
        audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    try:
        torchaudio.save(output_wav, audio, sample_rate)
        LOG.info(f"Saved WAV file to {output_wav}")
    except Exception as e:
        LOG.error(f"Error saving WAV file {output_wav}: {e}")
        return (None, preview_images, final_metrics)

    # If file_format is other than wav, convert to other file format
    if file_format and file_format != "wav":
        cmd = ""
        if file_format == "m4a aac_he_v2 32k":
            cmd = f'ffmpeg -i "{output_wav}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y "{output_filename}"'
        elif file_format == "m4a aac_he_v2 64k":
            cmd = f'ffmpeg -i "{output_wav}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y "{output_filename}"'
        elif file_format == "flac":
            cmd = f'ffmpeg -i "{output_wav}" -y "{output_filename}"'
        elif file_format == "mp3 320k":
            cmd = f'ffmpeg -i "{output_wav}" -b:a 320k -y "{output_filename}"'
        elif file_format == "mp3 v0":
            cmd = f'ffmpeg -i "{output_wav}" -q:a 0 -y "{output_filename}"'
        elif file_format == "mp3 128k":
            cmd = f'ffmpeg -i "{output_wav}" -b:a 128k -y "{output_filename}"'
        
        if cmd:
            cmd += " -loglevel error"  # make output less verbose in the cmd window
            try:
                subprocess.run(cmd, shell=True, check=True)
                LOG.info(f"Converted to {file_format} format: {output_filename}")
            except Exception as e:
                LOG.error(f"Error converting to {file_format}: {e}")
                return (output_wav, preview_images, final_metrics)

    # Generate spectrogram
    try:
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
        clean_spectrogram = None
        if clean_audio is not None:
            clean_audio = clean_audio[1]
            max_abs_val = torch.max(torch.abs(clean_audio))
            clean_audio = clean_audio.to(torch.float32).div(max_abs_val).clamp(-1, 1).mul(32767).to(torch.int16).cpu() if max_abs_val > 1e-7 else clean_audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            clean_spectrogram = audio_spectrogram_image(clean_audio, sample_rate=sample_rate)
    except Exception as e:
        LOG.warning(f"Could not generate spectrogram: {e}")
        audio_spectrogram = None
        clean_spectrogram = None

    LOG.info("Audio restoration completed")
    # Build gallery images, filtering out None entries to avoid Gradio errors
    images_to_show = []
    if audio_spectrogram is not None:
        images_to_show.append((audio_spectrogram, "Generated Audio"))
    if clean_spectrogram is not None:
        images_to_show.append((clean_spectrogram, "Clean Reference"))
    images_to_show.extend(preview_images)
    return (output_filename if file_format != "wav" else output_wav, images_to_show, final_metrics)

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes.
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file

    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start()

