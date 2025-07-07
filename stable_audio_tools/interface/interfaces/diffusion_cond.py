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
import matplotlib.pyplot as plt
from io import BytesIO

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..aeiou import audio_spectrogram_image
from ...inference.generation import (
    generate_diffusion_cond,
    generate_diffusion_cond_inpaint,
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


def generate_cond(
    prompt,
    negative_prompt=None,
    seconds_start=0,
    seconds_total=30,
    cfg_scale=6.0,
    steps=250,
    preview_every=None,
    seed=-1,
    sampler_type="dpmpp-3m-sde",
    sigma_min=0.03,
    sigma_max=1000,
    rho=1.0,
    cfg_interval_min=0.0,
    cfg_interval_max=1.0,
    cfg_rescale=0.0,
    file_format="wav",
    file_naming="verbose",
    cut_to_seconds_total=False,
    init_audio=None,
    init_noise_level=1.0,
    mask_maskstart=None,
    mask_maskend=None,
    inpaint_audio=None,
    batch_size=1,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Create date-based directory structure
    current_date = time.strftime("%Y-%m-%d")
    output_dir = os.path.join("output", current_date)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a dedicated folder for all degradation processing
    generation_dir = os.path.join(output_dir, "degradation_processing")
    os.makedirs(generation_dir, exist_ok=True)
    LOG.info(f"Files will be saved in directory: {generation_dir}")

    # Return fake stereo audio
    conditioning_dict = {
        "prompt": prompt,
        "seconds_start": seconds_start,
        "seconds_total": seconds_total,
    }

    conditioning = [conditioning_dict] * batch_size

    if negative_prompt:
        negative_conditioning_dict = {
            "prompt": negative_prompt,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
        }

        negative_conditioning = [negative_conditioning_dict] * batch_size
    else:
        negative_conditioning = None

    # Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)
    # if seed is -1, define the seed value now, randomly, so we can save it in the filename
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)

    input_sample_size = init_audio[1].shape[-1]

    if init_audio is not None:
        in_sr, init_audio = init_audio

        if init_audio.dtype == np.float32:
            init_audio = torch.from_numpy(init_audio)
        elif init_audio.dtype == np.int16:
            init_audio = torch.from_numpy(init_audio).float().div(32767)
        elif init_audio.dtype == np.int32:
            init_audio = torch.from_numpy(init_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")

        if model_half:
            init_audio = init_audio.to(torch.float16)

        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)  # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)  # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(init_audio.device)
                .to(init_audio.dtype)
            )
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            # input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            init_audio = init_audio[:, :sample_size]

        init_audio = (sample_rate, init_audio)

    if inpaint_audio is not None:
        in_sr, inpaint_audio = inpaint_audio

        if inpaint_audio.dtype == np.float32:
            inpaint_audio = torch.from_numpy(inpaint_audio)
        elif inpaint_audio.dtype == np.int16:
            inpaint_audio = torch.from_numpy(inpaint_audio).float().div(32767)
        elif inpaint_audio.dtype == np.int32:
            inpaint_audio = torch.from_numpy(inpaint_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {inpaint_audio.dtype}")

        if model_half:
            inpaint_audio = inpaint_audio.to(torch.float16)

        if inpaint_audio.dim() == 1:
            inpaint_audio = inpaint_audio.unsqueeze(0)  # [1, n]
        elif inpaint_audio.dim() == 2:
            inpaint_audio = inpaint_audio.transpose(0, 1)  # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(inpaint_audio.device)
                .to(inpaint_audio.dtype)
            )
            inpaint_audio = resample_tf(inpaint_audio)

        audio_length = inpaint_audio.shape[-1]

        if audio_length > sample_size:
            # input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            inpaint_audio = inpaint_audio[:, :sample_size]

        inpaint_audio = (sample_rate, inpaint_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
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

    demo_samples = input_sample_size
    if model.pretransform is not None:
        demo_samples = demo_samples // model.pretransform.downsampling_ratio
    
    conditioning_tensors = model.conditioner(conditioning, device)
    
    cond_inputs = model.get_conditioning_inputs(conditioning_tensors)
    
    if "input_concat_cond" in cond_inputs and cond_inputs["input_concat_cond"].shape[2] > demo_samples:
        cond_inputs["input_concat_cond"] = cond_inputs["input_concat_cond"][:, :, :demo_samples]
    
    generate_args = {
        "model": model,
        "conditioning_tensors": conditioning_tensors,
        "negative_conditioning": negative_conditioning,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": (cfg_interval_min, cfg_interval_max),
        "batch_size": batch_size,
        "sample_size": input_sample_size,
        "seed": seed,
        "device": device,
        "sampler_type": sampler_type,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "init_audio": init_audio,
        "init_noise_level": init_noise_level,
        "callback": progress_callback if preview_every is not None else None,
        "scale_phi": cfg_rescale,
        "rho": rho,
    }

    if model_type == "diffusion_cond_inpaint":
        if inpaint_audio is not None:
            # Convert mask start and end from percentages to sample indices
            mask_start = int(mask_maskstart * sample_rate)
            mask_end = int(mask_maskend * sample_rate)

            inpaint_mask = torch.ones(1, sample_size, device=device)
            inpaint_mask[:, mask_start:mask_end] = 0

            generate_args.update(
                {"inpaint_audio": inpaint_audio, "inpaint_mask": inpaint_mask}
            )
        audio, metrics = generate_diffusion_cond_inpaint(**generate_args)
    else:
        audio, metrics = generate_diffusion_cond(**generate_args)
    
    LOG.info("Generation completed")

    # Filenaming convention
    prompt_condensed = condense_prompt(prompt)
    if file_naming == "verbose":
        cfg_filename = "cfg%s" % (cfg_scale)
        seed_filename = seed
        if negative_prompt:
            prompt_condensed += ".neg-%s" % condense_prompt(negative_prompt)
        basename = "%s.%s.%s" % (prompt_condensed, cfg_filename, seed_filename)
    elif file_naming == "prompt":
        basename = prompt_condensed
    else:
        # simple e.g. "output.wav"
        basename = "output"

    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else:
        filename_extension = "wav"
    output_filename = os.path.join(generation_dir, f"{basename}.{filename_extension}")
    output_wav = os.path.join(generation_dir, f"{basename}.wav")

    # Cut the extra silence off the end, if the user requested a smaller seconds_total
    if cut_to_seconds_total:
        audio = audio[:, :, : seconds_total * sample_rate]

    # Encode the audio to WAV format
    audio = rearrange(audio, "b d n -> d (b n)")

    # Check if the audio tensor is empty before normalization
    if audio.numel() == 0:
        print(
            f"Warning: Generated audio is empty for prompt '{prompt}'. Skipping normalization and saving."
        )
        # Return None for the audio path, but keep preview images
        return (None, preview_images)

    # If audio is not empty, proceed with normalization
    # Adding a small epsilon to prevent division by zero if max(abs(audio)) is zero
    max_abs_val = torch.max(torch.abs(audio))
    if (
        max_abs_val > 1e-7
    ):  # Use a small threshold instead of exact zero for float stability
        audio = (
            audio.to(torch.float32)
            .div(max_abs_val)
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
    else:
        # Handle silence or very near silence: just convert type
        audio = audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # save as wav file
    try:
        torchaudio.save(output_wav, audio, sample_rate)
        LOG.info(f"Saved WAV file to {output_wav}")
    except Exception as e:
        print(f"Error saving WAV file {output_wav}: {e}")
        # If saving fails, return None for the path
        return (None, preview_images)

    # If file_format is other than wav, convert to other file format
    if file_format and file_format != "wav":
        cmd = ""
        if file_format == "m4a aac_he_v2 32k":
            # note: need to compile ffmpeg with --enable-libfdk_aac
            cmd = f'ffmpeg -i "{output_wav}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y "{output_filename}"'
        elif file_format == "m4a aac_he_v2 64k":
            cmd = f'ffmpeg -i "{output_wav}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y "{output_filename}"'
        elif file_format == "flac":
            cmd = f'ffmpeg -i "{output_wav}" -y "{output_filename}"'
        elif file_format == "mp3 320k":
            cmd = f'ffmpeg -i "{output_wav}" -b:a 320k -y "{output_filename}"'
        elif file_format == "mp3 128k":
            cmd = f'ffmpeg -i "{output_wav}" -b:a 128k -y "{output_filename}"'
        elif file_format == "mp3 v0":
            cmd = f'ffmpeg -i "{output_wav}" -q:a 0 -y "{output_filename}"'
        
        if cmd:
            cmd += " -loglevel error"  # make output less verbose in the cmd window
            try:
                subprocess.run(cmd, shell=True, check=True)
                LOG.info(f"Converted to {file_format} format: {output_filename}")
            except Exception as e:
                LOG.error(f"Error converting to {file_format}: {e}")
                return (output_wav, preview_images)

    # Generate spectrogram
    try:
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
        # Generate clean audio spectrogram if available
        clean_spectrogram = None
        if init_audio is not None:
            clean_spectrogram = audio_spectrogram_image(init_audio[1], sample_rate=sample_rate)
    except Exception as e:
        LOG.warning(f"Could not generate spectrogram: {e}")
        audio_spectrogram = None
        clean_spectrogram = None

    return (output_filename if file_format != "wav" else output_wav, [(audio_spectrogram, "Generated Audio"), (clean_spectrogram, "Clean Reference") if clean_spectrogram else None, *preview_images])

def generate_cond_restoration(
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
    file_naming="verbose",
    degraded_audio=None,
    clean_audio=None,
    batch_size=1,
    degraded_audio_filename=None,
    custom_output_dir=None,
):
    LOG.info("Starting audio restoration")
    
    # Access global variables
    global sample_rate, model
    
    # Initialize metrics dictionary
    metrics_dict = {}

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

    conditioning_dict = {
        "degraded_audio": degraded_audio[1] if degraded_audio is not None else None,
    }
    conditioning = [conditioning_dict] * batch_size

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
    
    # Disable separate metrics processing for batch processing
    is_batch_processing = os.environ.get("STABLE_AUDIO_BATCH_PROCESSING", "0") == "1"

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

    # Do the audio generation
    LOG.info("Generating audio")
    
    audio, final_metrics = generate_diffusion_cond(**generate_args)
    
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
            sample_rate = final_metrics.get("sample_rate", 0)
            if isinstance(sample_rate, list) and len(sample_rate) > 0:
                sample_rate = sample_rate[0]
            degraded_losses["sample_rate"] = sample_rate
            
            sample_size = final_metrics.get("sample_size", 0)
            if isinstance(sample_size, list) and len(sample_size) > 0:
                sample_size = sample_size[0]
            degraded_losses["sample_size"] = sample_size
            
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
            sample_rate = final_metrics.get("sample_rate", 0)
            if isinstance(sample_rate, list) and len(sample_rate) > 0:
                sample_rate = sample_rate[0]
            restored_losses["sample_rate"] = sample_rate
            
            sample_size = final_metrics.get("sample_size", 0)
            if isinstance(sample_size, list) and len(sample_size) > 0:
                sample_size = sample_size[0]
            restored_losses["sample_size"] = sample_size
            
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
    return (output_filename if file_format != "wav" else output_wav, [(audio_spectrogram, "Generated Audio"), (clean_spectrogram, "Clean Reference") if clean_spectrogram else None, *preview_images], final_metrics)

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes.
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file

    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start()


def create_metric_plots(metrics_data_list, labels):
    try:
        if not metrics_data_list:
            LOG.warning("No metrics data provided to plot.")
            return None

        # Determine the set of all metric keys across all files
        all_metric_keys = set()
        for metrics_data in metrics_data_list:
            if metrics_data:
                all_metric_keys.update(k for k in metrics_data.keys() if k not in ["steps", "timestamp", "sample_rate", "sample_size", "generation_params"])

        # Exclude all degraded_ metrics from plotting
        filtered_metric_keys = [k for k in all_metric_keys if not k.startswith('degraded_')]
        
        # Now separate the remaining metrics into restoration and main categories
        restoration_metrics_keys = sorted([k for k in filtered_metric_keys if k.startswith('restoration_success_')])
        main_metrics_keys = sorted([k for k in filtered_metric_keys if k not in restoration_metrics_keys])

        if not main_metrics_keys and not restoration_metrics_keys:
            LOG.warning("No metrics found to plot.")
            return None

        n_plots = len(main_metrics_keys)
        num_restoration_plots = 0
        if restoration_metrics_keys:
            for metrics_data in metrics_data_list:
                if metrics_data and any(k in metrics_data for k in restoration_metrics_keys):
                    num_restoration_plots += 1
        
        n_plots += num_restoration_plots

        if n_plots == 0:
            return None
        
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create figure with extra space for legend
        fig = plt.figure(figsize=(15, 4 * n_rows + 1))
        gs = plt.GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.2])
        axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(n_plots)]

        plot_idx = 0

        # Plot main metrics
        for metric_name in main_metrics_keys:
            ax = axes[plot_idx]
            for i, metrics_data in enumerate(metrics_data_list):
                if metrics_data and metric_name in metrics_data and "steps" in metrics_data and metrics_data["steps"]:
                    steps = metrics_data["steps"]
                    # Vérifier si l'index i est valide dans la liste labels
                    label = labels[i] if i < len(labels) else f'Audio {i+1}'
                    ax.plot(steps, metrics_data[metric_name], '-o', label=label, markersize=4)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.set_title(f'{metric_name} over steps')
            ax.grid(True)
            ax.legend(loc='best')
            plot_idx += 1
        
        # Plot restoration success metrics, one plot per file
        if restoration_metrics_keys:
            # Create a single legend for all restoration plots
            legend_handles = []
            legend_labels = []
            
            for i, metrics_data in enumerate(metrics_data_list):
                file_restoration_metrics = [k for k in restoration_metrics_keys if k in metrics_data] if metrics_data else []

                if file_restoration_metrics and "steps" in metrics_data and metrics_data["steps"]:
                    ax = axes[plot_idx]
                    steps = metrics_data["steps"]
                    for metric_name in file_restoration_metrics:
                        label = metric_name.replace('restoration_success_', '')
                        line = ax.plot(steps, metrics_data[metric_name], '-o', label=label, markersize=4)[0]
                        if i == 0:  # Only collect handles and labels from first plot
                            legend_handles.append(line)
                            legend_labels.append(label)
                    
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Value')
                    ax.set_title(f'NRS for {labels[i]}')
                    ax.grid(True)
                    plot_idx += 1

            # Add single legend at the bottom
            if legend_handles:
                legend_ax = fig.add_subplot(gs[-1, :])
                legend_ax.axis('off')
                legend_ax.set_title('Legend for Normalized Restoration Success (NRS)')
                legend_ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_handles))

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(pad=3.0)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        
        from PIL import Image
        img = Image.open(buf)
        return np.array(img)
    except Exception as e:
        LOG.error(f"Error creating metric plots: {str(e)}", exc_info=True)
        return None
