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


# def compute_metrics(audio, clean_audio=None, degraded_audio=None, model=None, device="cuda"):
#     """Compute audio quality metrics between the generated audio and clean reference if provided."""
#     metrics = {}
    
#     if clean_audio is not None and model is not None:
#         from stable_audio_tools.training.losses.metrics import (
#             LogSpectralDistance,
#             LTASDistance,
#             SISDRMetric,
#             SNRMetric,
#             STFTDistance,
#             MelDistance
#         )
        
#         # Move tensors to the correct device and ensure they're float32
#         audio = audio.to(device)
#         clean_audio = clean_audio.to(device)
        
#         LOG.info(f"Audio shapes - Generated: {audio.shape}, Clean: {clean_audio.shape}")
#         LOG.info(f"Audio dtypes - Generated: {audio.dtype}, Clean: {clean_audio.dtype}")
#         LOG.info(f"Audio ranges - Generated: [{audio.min().item():.3f}, {audio.max().item():.3f}], Clean: [{clean_audio.min().item():.3f}, {clean_audio.max().item():.3f}]")
        
#         # If using a pretransform model, process both audios through the same pipeline
#         if model.pretransform is not None:
#             # Encode both audios to latent space
#             clean_audio = clean_audio.to(device)
#             audio = audio.to(device)
#             degraded_audio = degraded_audio.to(device)
#             clean_audio_latent = model.pretransform.encode(clean_audio.unsqueeze(0))
#             audio_latent = model.pretransform.encode(audio.unsqueeze(0))
#             degraded_audio_latent = model.pretransform.encode(degraded_audio.unsqueeze(0))
#             # Decode back to waveform space for consistent comparison
#             clean_audio = model.pretransform.decode(clean_audio_latent)
#             degraded_audio = model.pretransform.decode(degraded_audio_latent)
            
#             # Remove batch dimension and ensure correct shape
#             audio = audio.squeeze(0)
#             clean_audio = clean_audio.squeeze(0)
#             degraded_audio = degraded_audio.squeeze(0)
#             LOG.info(f"After transform - Audio shapes - Generated: {audio.shape}, Clean: {clean_audio.shape}, Degraded: {degraded_audio.shape}")
        
#         # Ensure both audios are the same length
#         if audio.shape[-1] != clean_audio.shape[-1]:
#             min_length = min(audio.shape[-1], clean_audio.shape[-1])
#             audio = audio[..., :min_length]
#             clean_audio = clean_audio[..., :min_length]
#             LOG.info(f"After length adjustment - Audio shapes - Generated: {audio.shape}, Clean: {clean_audio.shape}")
        
#         metrics_instances = {
#             'lsd': LogSpectralDistance().to(device),
#             'ltas': LTASDistance().to(device),
#             'sisdr': SISDRMetric().to(device),
#             'snr': SNRMetric().to(device),
#             'stft': STFTDistance().to(device),
#             'mel': MelDistance(sample_rate=model.sample_rate).to(device)
#         }
        
#         # Calculate metrics between generated and clean audio
#         for name, metric in metrics_instances.items():
#             try:
#                 # Calculate demo metric (generated vs clean)
#                 demo_metric = metric(audio, clean_audio).item()
#                 metrics[name] = demo_metric
                
#                 # Calculate degraded metric (degraded vs clean)
#                 degraded_metric = metric(degraded_audio, clean_audio).item()
                
#                 # Calculate clean metric (clean vs clean)
#                 clean_metric = metric(clean_audio, model.pretransform.decode(model.pretransform.encode(clean_audio.unsqueeze(0))).squeeze(0)).item()
                
#                 # Calculate restoration success metric
#                 restoration_success = 1 - abs((demo_metric - degraded_metric)) / abs((clean_metric - degraded_metric))
#                 metrics[f'restoration_success_{name}'] = restoration_success
                
#                 LOG.info(f"Computed {name}: {demo_metric:.3f}")
#                 LOG.info(f"Restoration success for {name}: {restoration_success:.3f}")
#             except Exception as e:
#                 LOG.error(f"Error computing {name}: {str(e)}")
        
#         # Latent space losses if using a pretransform model
#         if model.pretransform is not None:
#             try:
#                 LOG.info("Computing latent space metrics...")
#                 # MSE Loss
#                 demo_mse = F.mse_loss(audio_latent, clean_audio_latent)
#                 degraded_mse = F.mse_loss(degraded_audio_latent, clean_audio_latent)
#                 clean_mse = F.mse_loss(clean_audio_latent, clean_audio_latent)
#                 metrics['latent_mse_loss'] = demo_mse.item()
#                 metrics['restoration_success_latent_mse_loss'] = 1 - abs((demo_mse.item() - degraded_mse.item())) / abs((clean_mse.item() - degraded_mse.item()))
#                 LOG.info(f"Latent MSE Loss: {demo_mse.item():.3f}")
#                 LOG.info(f"Restoration Success Latent MSE: {metrics['restoration_success_latent_mse_loss']:.3f}")
                
#                 # L1 Loss
#                 demo_l1 = F.l1_loss(audio_latent, clean_audio_latent)
#                 degraded_l1 = F.l1_loss(degraded_audio_latent, clean_audio_latent)
#                 clean_l1 = F.l1_loss(clean_audio_latent, clean_audio_latent)
#                 metrics['latent_l1_loss'] = demo_l1.item()
#                 metrics['restoration_success_latent_l1_loss'] = 1 - abs((demo_l1.item() - degraded_l1.item())) / abs((clean_l1.item() - degraded_l1.item()))
#                 LOG.info(f"Latent L1 Loss: {demo_l1.item():.3f}")
#                 LOG.info(f"Restoration Success Latent L1: {metrics['restoration_success_latent_l1_loss']:.3f}")
#             except Exception as e:
#                 LOG.error(f"Error computing latent space metrics: {str(e)}")
        
#         # Waveform domain losses
#         try:
#             LOG.info("Computing waveform domain losses...")
#             # MSE Loss
#             demo_waveform_mse = F.mse_loss(audio, clean_audio)
#             degraded_waveform_mse = F.mse_loss(degraded_audio, clean_audio)
#             clean_waveform_mse = F.mse_loss(clean_audio, clean_audio)
#             metrics['waveform_mse_loss'] = demo_waveform_mse.item()
#             metrics['restoration_success_waveform_mse_loss'] = 1 - abs((demo_waveform_mse.item() - degraded_waveform_mse.item())) / abs((clean_waveform_mse.item() - degraded_waveform_mse.item()))
#             LOG.info(f"Waveform MSE Loss: {demo_waveform_mse.item():.3f}")
#             LOG.info(f"Restoration Success Waveform MSE: {metrics['restoration_success_waveform_mse_loss']:.3f}")
            
#             # L1 Loss
#             demo_waveform_l1 = F.l1_loss(audio, clean_audio)
#             degraded_waveform_l1 = F.l1_loss(degraded_audio, clean_audio)
#             clean_waveform_l1 = F.l1_loss(clean_audio, clean_audio)
#             metrics['waveform_l1_loss'] = demo_waveform_l1.item()
#             metrics['restoration_success_waveform_l1_loss'] = 1 - abs((demo_waveform_l1.item() - degraded_waveform_l1.item())) / abs((clean_waveform_l1.item() - degraded_waveform_l1.item()))
#             LOG.info(f"Waveform L1 Loss: {demo_waveform_l1.item():.3f}")
#             LOG.info(f"Restoration Success Waveform L1: {metrics['restoration_success_waveform_l1_loss']:.3f}")
#         except Exception as e:
#             LOG.error(f"Error computing waveform domain losses: {str(e)}")
    
#     return metrics


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

def create_sampling_ui(model_config):
    has_inpainting = model_config["model_type"] == "diffusion_cond_inpaint"

    model_conditioning_config = model_config["model"].get("conditioning", None)
    
    # Check if this is a specific audio restoration model
    input_concat_ids = model_config["model"].get("diffusion", {}).get("input_concat_ids", [])
    is_audio_restoration = "degraded_audio" in input_concat_ids
    
    LOG.info(f"Input concat IDs: {input_concat_ids}")
    LOG.info(f"Is audio restoration model: {is_audio_restoration}")
    
    # Define noise_level_slider as a global variable to access it from generate_cond
    global noise_level_slider

    diffusion_objective = model.diffusion_objective

    is_rf = diffusion_objective == "rectified_flow"
    
    with gr.Row():
        with gr.Column(scale=6):
            prompt_visible = not is_audio_restoration
            prompt = gr.Textbox(show_label=False, placeholder="Prompt", visible=prompt_visible)
            negative_prompt = gr.Textbox(
                show_label=False, placeholder="Negative prompt", visible=prompt_visible
            )
            
            # Information message for audio restoration model
            if is_audio_restoration:
                gr.Markdown("### Audio Restoration Model\nUpload an audio file to restore below")
                
        generate_button = gr.Button("Generate", variant="primary", scale=1)
        
    with gr.Row(equal_height=False):
        with gr.Column():
            has_seconds_start = False
            has_seconds_total = False
            
            if model_conditioning_config:
                for config in model_conditioning_config.get("configs", []):
                    if config.get("id") == "seconds_start":
                        has_seconds_start = True
                    if config.get("id") == "seconds_total":
                        has_seconds_total = True
            
            timing_visible = (has_seconds_start or has_seconds_total) and not is_audio_restoration
            with gr.Row(visible=timing_visible):
                # Timing controls
                seconds_start_slider = gr.Slider(
                    minimum=0,
                    maximum=512,
                    step=1,
                    value=0,
                    label="Seconds start",
                    visible=has_seconds_start,
                )
                seconds_total_slider = gr.Slider(
                    minimum=0,
                    maximum=512,
                    step=1,
                    value=sample_size // sample_rate,
                    label="Seconds total",
                    visible=has_seconds_total,
                )

            with gr.Row():
                # Controls for audio restoration
                if is_audio_restoration:
                    # Add information about how the model uses the audio
                    gr.Markdown("The uploaded audio will be used as conditioning input for the model")
                
                # Steps slider
                default_steps = 30 if is_audio_restoration else (50 if is_rf else 100)
                steps_slider = gr.Slider(
                    minimum=1, maximum=500, step=1, value=default_steps, label="Steps"
                )

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    # Seed
                    seed_textbox = gr.Textbox(
                        label="Seed (set to -1 for random seed)", value="-1"
                    )

                    cfg_interval_min_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.01,
                        value=0.0,
                        label="CFG interval min",
                    )
                    cfg_interval_max_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.01,
                        value=1.0,
                        label="CFG interval max",
                    )

                with gr.Row():
                    cfg_rescale_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.01,
                        value=0.0,
                        label="CFG rescale amount",
                    )

                with gr.Row():
                    # Sampler params
                    if is_rf:
                        sampler_types = ["euler", "rk4", "dpmpp"]
                        default_sampler_type = "euler"
                    else:
                        sampler_types = [
                            "dpmpp-2m-sde",
                            "dpmpp-3m-sde",
                            "dpmpp-2m",
                            "k-heun",
                            "k-lms",
                            "k-dpmpp-2s-ancestral",
                            "k-dpm-2",
                            "k-dpm-adaptive",
                            "k-dpm-fast",
                            "v-ddim",
                            "v-ddim-cfgpp",
                        ]
                        default_sampler_type = "dpmpp-3m-sde"
                    sampler_type_dropdown = gr.Dropdown(
                        sampler_types, label="Sampler type", value=default_sampler_type
                    )
                    sigma_min_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=0.01,
                        label="Sigma min",
                        visible=not is_rf,
                    )
                    sigma_max_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1000.0,
                        step=0.1,
                        value=100,
                        label="Sigma max",
                        visible=not is_rf,
                    )
                    rho_slider = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        step=0.01,
                        value=1.0,
                        label="Sigma curve strength",
                        visible=not is_rf,
                    )

            with gr.Accordion("Output params", open=False):
                with gr.Row():
                    file_format_dropdown = gr.Dropdown(
                        [
                            "wav",
                            "flac",
                            "mp3 320k",
                            "mp3 v0",
                            "mp3 128k",
                            "m4a aac_he_v2 64k",
                            "m4a aac_he_v2 32k",
                        ],
                        label="File format",
                        value="wav",
                    )
                    file_naming_dropdown = gr.Dropdown(
                        ["verbose", "prompt", "output.wav"],
                        label="File naming",
                        value="output.wav",
                    )
                    cut_to_seconds_total_checkbox = gr.Checkbox(
                        label="Cut to seconds total", value=True
                    )
                    preview_every_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                        label="Spec Preview Every N Steps",
                    )
                    metrics_every_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=30,
                        label="Compute Metrics Every N Steps",
                        visible=is_audio_restoration,
                    )
            if is_audio_restoration:
                with gr.Accordion("Audio Inputs", open=True):
                    with gr.Row():
                        degraded_audio_files = gr.File(label="Degraded audio files", file_count="multiple")
                        clean_audio = gr.Audio(label="Clean reference audio (optional)", visible=True)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            degraded_audio_dropdown = gr.Dropdown(label="Select degraded audio to play", interactive=True)
                        with gr.Column(scale=1):
                            degraded_audio_player = gr.Audio(label="Degraded audio player")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Batch Processing Options")
                        
                    with gr.Row():
                        process_folder_path = gr.Textbox(
                            label="Audio Folder",
                            placeholder="Path to folder where the algorithm should search for audio files (e.g. audio/degraded/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_04_Track04_wav)",
                        )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            model_name = gr.Textbox(
                                label="Model name",
                                placeholder="Name of the model for results file (e.g. intense_equalizer)",
                            )
                        with gr.Column(scale=1):
                            effects_list = gr.Textbox(
                                label="Effects list",
                                placeholder="Comma-separated list of effects (e.g. equalizer,bass,overdrive)",
                            )
            else:
                # Default generation tab
                with gr.Accordion("Init audio", open=False):
                    init_audio_input = gr.Audio(label="Init audio")
                    min_noise_level = 0.01 if is_rf else 0.1
                    max_noise_level = 1.0 if is_rf else 100.0
                    init_noise_level_slider = gr.Slider(
                        minimum=min_noise_level,
                        maximum=max_noise_level,
                        step=0.01,
                        value=0.1,
                        label="Init noise level",
                    )

            with gr.Accordion("Inpainting", open=False, visible=has_inpainting):
                inpaint_audio_input = gr.Audio(label="Inpaint audio")
                mask_maskstart_slider = gr.Slider(
                    minimum=0.0,
                    maximum=sample_size // sample_rate,
                    step=0.1,
                    value=10,
                    label="Mask Start (sec)",
                )
                mask_maskend_slider = gr.Slider(
                    minimum=0.0,
                    maximum=sample_size // sample_rate,
                    step=0.1,
                    value=sample_size // sample_rate,
                    label="Mask End (sec)",
                )

            if is_audio_restoration:
                LOG.info("Audio restoration mode enabled")
                inputs = [
                        steps_slider,
                        preview_every_slider,
                        metrics_every_slider,
                        seed_textbox,
                        sampler_type_dropdown,
                        sigma_min_slider,
                        sigma_max_slider,
                        rho_slider,
                        cfg_rescale_slider,
                        file_format_dropdown,
                        file_naming_dropdown,
                        degraded_audio_files,
                        clean_audio,
                        process_folder_path,
                        model_name,
                        effects_list,
                    ]
            else:
                LOG.info("Default generation mode enabled")
                inputs = [
                    prompt,
                    negative_prompt,
                    seconds_start_slider,
                    seconds_total_slider,
                    steps_slider,
                    preview_every_slider,
                    seed_textbox,
                    sampler_type_dropdown,
                    sigma_min_slider,
                    sigma_max_slider,
                    rho_slider,
                    cfg_interval_min_slider,
                    cfg_interval_max_slider,
                    cfg_rescale_slider,
                    file_format_dropdown,
                    file_naming_dropdown,
                    cut_to_seconds_total_checkbox,
                    init_audio_input,
                    init_noise_level_slider,
                    mask_maskstart_slider,
                    mask_maskend_slider,
                    inpaint_audio_input,
                ] 
            

        with gr.Column():
            output_audio_dropdown = gr.Dropdown(label="Select generated audio")
            output_audio_player = gr.Audio(label="Audio player")
            audio_spectrogram_output = gr.Gallery(label="Output spectrograms", show_label=True, elem_id="spectrogram_gallery", columns=2, height=400)
            
            # Add metrics display section
            with gr.Accordion("Restoration Metrics", open=True, visible=is_audio_restoration):
                metrics_plots = gr.Image(label="Metrics Plots", visible=is_audio_restoration, type="numpy")
            
            # Use different target based on model type
            if is_audio_restoration:
                send_to_init_button = gr.Button("Send to degraded audio", scale=1)
                send_to_init_button.click(
                    fn=lambda audio: [audio] if audio else None,
                    inputs=[output_audio_player],
                    outputs=[degraded_audio_files]
                    )
            else:
                send_to_init_button = gr.Button("Send to init audio", scale=1)
                send_to_init_button.click(
                    fn=lambda audio: audio,
                    inputs=[output_audio_player],
                    outputs=[init_audio_input],
                )

            if has_inpainting:
                send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
                send_to_inpaint_button.click(
                    fn=lambda audio: audio,
                    inputs=[output_audio_player],
                    outputs=[inpaint_audio_input],
                )

    def generate_multiple_with_plots(steps, preview_every, metrics_every, seed, sampler_type, sigma_min, sigma_max, rho, cfg_rescale, file_format, file_naming, degraded_audio_files, clean_audio, process_folder_path=None, model_name=None, effects_list=None):
        if not is_audio_restoration:
            return [], [], None
            
        # Check if we have a folder path to process
        folder_files = []
        effects_files = {}
        consolidated_results = None
        
        # Define output directory for JSON results - always in output/batch_processing
        output_dir = os.path.join("output", "batch_processing")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define directories for restored and degraded audio files
        restored_audio_dir = os.path.join("output", "audio", "restored")
        degraded_audio_dir = os.path.join("output", "audio", "degraded")
        os.makedirs(restored_audio_dir, exist_ok=True)
        os.makedirs(degraded_audio_dir, exist_ok=True)
        
        if process_folder_path and os.path.isdir(process_folder_path):
            LOG.info(f"Processing folder: {process_folder_path}")
            
            # Process effects list if provided
            if model_name and effects_list:
                effects = [effect.strip() for effect in effects_list.split(",")]
                LOG.info(f"Looking for effects: {effects}")
                
                # Initialize the consolidated results JSON file with the expected structure
                consolidated_results_file = os.path.join(output_dir, f"results_{model_name}.json")
                
                # If the file already exists, load it to keep existing data
                if os.path.exists(consolidated_results_file):
                    try:
                        with open(consolidated_results_file, 'r') as f:
                            consolidated_results = json.load(f)
                            
                            # Reformat existing data to ensure correct structure
                            # Fix audio paths and losses format for all existing entries
                            if "data" in consolidated_results:
                                for audio_name, audio_data in list(consolidated_results["data"].items()):
                                    if audio_name not in ["light", "standard", "strong", "unknown"]:
                                        for duration, duration_data in list(audio_data.items()):
                                            for intensity, intensity_data in list(duration_data.items()):
                                                if "effects" in intensity_data:
                                                    for i, effect in enumerate(intensity_data["effects"]):
                                                        # Fix degraded audio path
                                                        if "degraded" in effect and "audio" in effect["degraded"]:
                                                            old_path = effect["degraded"]["audio"]
                                                            if "/" in old_path.replace("audio/degraded/", ""):
                                                                # Extract parts from path
                                                                parts = old_path.split("/")
                                                                audio_base = parts[2]  # After audio/degraded/
                                                                
                                                                # Find effect name (last part of path without extension)
                                                                effect_name = os.path.splitext(parts[-1])[0]
                                                                
                                                                # Get duration and intensity from middle parts
                                                                dur_part = next((p for p in parts[3:-1] if p.endswith('s') and p[:-1].isdigit()), "5s")
                                                                int_part = next((p for p in parts[3:-1] if p in ["light", "standard", "strong"]), "standard")
                                                                
                                                                # Create new path format
                                                                new_path = f"audio/degraded/{audio_base}_{dur_part}_{int_part}_{effect_name}.wav"
                                                                effect["degraded"]["audio"] = new_path
                                                                LOG.info(f"Fixed degraded audio path: {old_path} -> {new_path}")
                                                        
                                                        # Fix losses format for degraded
                                                        if "degraded" in effect and "losses" in effect["degraded"] and isinstance(effect["degraded"]["losses"], dict):
                                                            for loss_type in ["l1", "l2", "snr"]:
                                                                if loss_type in effect["degraded"]["losses"] and isinstance(effect["degraded"]["losses"][loss_type], list):
                                                                    # Convert array to single value (use first value or average)
                                                                    values = effect["degraded"]["losses"][loss_type]
                                                                    if values:
                                                                        effect["degraded"]["losses"][loss_type] = values[0] if len(values) > 0 else 0
                                                        
                                                        # Fix losses format for restored
                                                        if "restored" in effect and "losses" in effect["restored"] and isinstance(effect["restored"]["losses"], dict):
                                                            for loss_type in ["l1", "l2", "snr"]:
                                                                if loss_type in effect["restored"]["losses"] and isinstance(effect["restored"]["losses"][loss_type], list):
                                                                    # Convert array to single value (use first value or average)
                                                                    values = effect["restored"]["losses"][loss_type]
                                                                    if values:
                                                                        effect["restored"]["losses"][loss_type] = values[0] if len(values) > 0 else 0
                            
                            LOG.info("Restructured existing results JSON to fix formats")
                    except Exception as e:
                        LOG.warning(f"Could not load existing results file: {e}")
                        consolidated_results = {
                            "model": model_name,
                            "data": {}
                        }
                else:
                    consolidated_results = {
                        "model": model_name,
                        "data": {}
                    }
                
                # Supprimer les entrées "unknown" si elles existent
                keys_to_delete = []
                for key in consolidated_results["data"].keys():
                    if key in ["light", "standard", "strong", "unknown"]:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del consolidated_results["data"][key]
                
                # Ensure model_name is sanitized for filenames
                model_name = model_name.replace(" ", "_")
                
                # Find all audio files that match the effects in subfolders
                for root, dirs, files in os.walk(process_folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_name = os.path.basename(file_path)
                        file_name_without_ext, ext = os.path.splitext(file_name)
                        
                        if ext.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                            # Check if this file contains any of the effects names
                            for effect in effects:
                                if effect in file_name_without_ext:
                                    if effect not in effects_files:
                                        effects_files[effect] = []
                                    effects_files[effect].append(file_path)
                                    LOG.info(f"Found {effect} effect in file: {file_path}")
                
                # Create a list of files to process from the effects files
                for effect, files in effects_files.items():
                    folder_files.extend(files)
                
                if not folder_files:
                    raise gr.Error(f"No audio files matching effects {effects} found in folder {process_folder_path}")
                
                LOG.info(f"Found {len(folder_files)} audio files to process")
                
                # Utilisation du output_dir déjà défini
                LOG.info(f"Results will be saved in: {output_dir}")
            else:
                # Standard folder processing without effects list
                extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
                for ext in extensions:
                    folder_files.extend(glob.glob(os.path.join(process_folder_path, f"*{ext}")))
                
                if not folder_files:
                    raise gr.Error(f"No audio files found in folder {process_folder_path}")
                
                LOG.info(f"Found {len(folder_files)} audio files to process")
                
                # Using the centralized output directory already defined
                LOG.info(f"Results will be saved in: {output_dir}")
        elif not degraded_audio_files:
            raise gr.Error("No degraded audio files provided.")

        labels = []
        
        # If we're processing from a folder
        if process_folder_path and folder_files:
            for f in folder_files:
                filename = os.path.basename(f)
                filename_no_ext = os.path.splitext(filename)[0]
                labels.append(filename_no_ext)
                
                # If using effects list, extract effect name from filename
                if effects_list:
                    # Find which effect this file corresponds to
                    current_effect = None
                    for effect in effects_files.keys():
                        if effect in filename_no_ext:
                            current_effect = effect
                            break
                    
                    if current_effect:
                        # Build structure for this effect in the consolidated results
                        # Extract base audio name and duration from path
                        rel_path = os.path.relpath(f, process_folder_path)
                        parts = rel_path.split(os.sep)
                        
                        # Try to identify the structure: audio_name/duration_category/intensity/file
                        # We'll attempt to extract audio name and duration assuming common patterns
                        audio_name = parts[0] if len(parts) > 1 else "unknown"
                        
                        # Look for duration marker (e.g. 5s, 10s)
                        duration = None
                        intensity = "standard"  # Default intensity
                        
                        for part in parts:
                            if part.endswith("s") and part[:-1].isdigit():
                                duration = part
                            if part in ["light", "standard", "strong"]:
                                intensity = part
                        
                        if not duration:
                            duration = "unknown"
                            
                        # Initialize nested structures if they don't exist
                        if audio_name not in consolidated_results["data"]:
                            consolidated_results["data"][audio_name] = {}
                        
                        if duration not in consolidated_results["data"][audio_name]:
                            consolidated_results["data"][audio_name][duration] = {}
                            
                        if intensity not in consolidated_results["data"][audio_name][duration]:
                            consolidated_results["data"][audio_name][duration][intensity] = {"effects": []}
        # If we're uploading individual files
        if degraded_audio_files and not process_folder_path:
            for f in degraded_audio_files:
                filename = os.path.basename(f.name)
                filename_no_ext = os.path.splitext(filename)[0]
                labels.append(filename_no_ext)

        all_metrics = []
        output_audios_list = []
        output_spectrograms_list = []
        labels = []
        # Configure environment variables for batch processing
        # Disable creation of temporary directories
        os.environ["STABLE_AUDIO_NO_DATE_FOLDER"] = "1"
        os.environ["STABLE_AUDIO_BATCH_PROCESSING"] = "1"
        
        # Utiliser le dossier batch_processing directement pour les fichiers temporaires
        batch_processing_dir = os.path.join("output", "batch_processing")
        os.makedirs(batch_processing_dir, exist_ok=True)
        os.environ["STABLE_AUDIO_CUSTOM_TMP_DIR"] = batch_processing_dir
        
        # Determine which files to process - either from folder or uploaded files
        files_to_process = folder_files if process_folder_path and folder_files else [f.name for f in degraded_audio_files]
            
        for i, degraded_audio_path in enumerate(files_to_process):
            LOG.info(f"Processing file {i+1}/{len(files_to_process)}: {os.path.basename(degraded_audio_path)}")
            
            try:
                audio_data, sr = torchaudio.load(degraded_audio_path)
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.repeat(2,1)
                degraded_audio_input = (sr, audio_data.numpy())
            except Exception as e:
                LOG.error(f"Error loading audio file {degraded_audio_path}: {str(e)}")
                continue

            audio, spectrograms, metrics = generate_cond_restoration(
                steps=steps,
                preview_every=preview_every,
                metrics_every=metrics_every,
                seed=seed,
                sampler_type=sampler_type,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                cfg_rescale=cfg_rescale,
                file_format=file_format,
                file_naming=file_naming,
                degraded_audio=degraded_audio_input,
                clean_audio=clean_audio,
                batch_size=1,
                degraded_audio_filename=degraded_audio_path  # Pass the filename here
            )
            
            # Process based on whether we're using effects list or standard folder processing
            if model_name and effects_list and consolidated_results:
                original_filename = os.path.basename(degraded_audio_path)
                original_filename_without_ext = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(audio)[1]
                
                # Find the effect this file corresponds to
                current_effect = None
                
                # Sort effects by length (descending) to avoid partial matches
                # This ensures 'intense_equalizer' is checked before 'equalizer'
                sorted_effects = sorted(effects_files.keys(), key=len, reverse=True)
                
                for effect in sorted_effects:
                    # Use a more precise matching to avoid 'equalizer' matching in 'intense_equalizer'
                    # Check for exact matches or effect name with underscore/hyphen boundaries
                    parts = original_filename_without_ext.split('_')
                    if effect == original_filename_without_ext or effect in parts:
                        current_effect = effect
                        LOG.info(f"Matched effect '{effect}' for file '{original_filename_without_ext}'")
                        break
                
                if current_effect:
                    # Extract path components for structured data
                    rel_path = os.path.relpath(degraded_audio_path, process_folder_path)
                    parts = rel_path.split(os.sep)
                    
                    # Get audio name, duration, and intensity
                    audio_name = parts[0] if len(parts) > 1 else "unknown"
                    
                    duration = "unknown"
                    intensity = "standard"
                    
                    for part in parts:
                        if part.endswith("s") and part[:-1].isdigit():
                            duration = part
                        if part in ["light", "standard", "strong"]:
                            intensity = part
                    
                    # Analyse the file path to extract complete information
                    path_parts = degraded_audio_path.split('/')
                    
                    # Extract the original audio name, duration, intensity, and effect
                    # Expected format: audio/degraded/AUDIO_NAME/DURATION/INTENSITY/EFFECT.wav
                    audio_name = None
                    duration = None
                    intensity = None
                    effect_name = None
                    
                    # Search for path parts
                    for i, part in enumerate(path_parts):
                        if part == "degraded" and i < len(path_parts) - 1:
                            # The audio name is just after 'degraded'
                            audio_name = path_parts[i+1]
                            
                            # The duration is two levels after 'degraded', if available
                            if i + 2 < len(path_parts):
                                duration = path_parts[i+2]
                                
                            # The intensity is three levels after 'degraded', if available
                            if i + 3 < len(path_parts):
                                intensity = path_parts[i+3]
                                
                            # The effect is four levels after 'degraded', if available
                            if i + 4 < len(path_parts):
                                # Remove extension if present
                                effect_name = os.path.splitext(path_parts[i+4])[0]
                            break
                    
                    LOG.info(f"Extracted: Audio={audio_name}, Duration={duration}, Intensity={intensity}, Effect={effect_name}")
                    
                    # Use extracted values or default values
                    audio_name = audio_name or os.path.splitext(os.path.basename(degraded_audio_path))[0].split('_')[0]
                    duration = duration or "5s"
                    intensity = intensity or "standard"
                    effect_name = effect_name or current_effect or "unknown_effect"
                    
                    # Format the filename with audio name, duration, intensity, effect, and restored suffix
                    # Example: MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_04_Track04_wav_5s_light_bass_restored.wav
                    restored_basename = f"{audio_name}_{duration}_{intensity}_{effect_name}_restored{ext}"
                    
                    # Copy the audio file to restored folder
                    new_filepath = os.path.join(restored_audio_dir, restored_basename)
                    LOG.info(f"Saving restored audio to: {new_filepath}")
                    shutil.copy2(audio, new_filepath)
                    
                    # Vérifier si la structure existe déjà et la créer si nécessaire
                    if audio_name not in consolidated_results["data"]:
                        consolidated_results["data"][audio_name] = {}
                    
                    if duration not in consolidated_results["data"][audio_name]:
                        consolidated_results["data"][audio_name][duration] = {}
                        
                    if intensity not in consolidated_results["data"][audio_name][duration]:
                        consolidated_results["data"][audio_name][duration][intensity] = {}
                    
                    if "effects" not in consolidated_results["data"][audio_name][duration][intensity]:
                        consolidated_results["data"][audio_name][duration][intensity]["effects"] = []
                    
                    # Check if this effect entry already exists
                    effect_exists = False
                    for existing_effect in consolidated_results["data"][audio_name][duration][intensity]["effects"]:
                        # Update comparison to check just the effect name, not the full path
                        if existing_effect["name"] == current_effect:
                            effect_exists = True
                            LOG.info(f"Effect {current_effect} already processed for this audio, updating it instead")
                            
                            # We already have metrics from generate_cond_restoration call at line 1615
                            # No need to reload them from file - just use what we have
                            LOG.info(f"Using metrics from generate_cond_restoration for effect {current_effect}")
                            # Log available metrics types
                            if metrics and isinstance(metrics, dict):
                                LOG.info(f"Available metrics keys: {list(metrics.keys())}")
                                if "degraded_metrics" in metrics and isinstance(metrics["degraded_metrics"], dict):
                                    LOG.info(f"Degraded metrics: {list(metrics['degraded_metrics'].keys())}")
                                if "restored_metrics" in metrics and isinstance(metrics["restored_metrics"], dict):
                                    LOG.info(f"Restored metrics: {list(metrics['restored_metrics'].keys())}")
                            else:
                                LOG.warning(f"No metrics available from generate_cond_restoration for effect {current_effect}")
                            
                            # Restructure metrics if needed - the metrics from generate_cond_restoration may be flat, not nested
                            degraded_metrics = {}
                            restored_metrics = {}
                            
                            # Check if metrics come in the expected nested format
                            if "degraded_metrics" in metrics and isinstance(metrics["degraded_metrics"], dict):
                                # Already in expected format
                                degraded_metrics = metrics["degraded_metrics"]
                                LOG.info("Using nested degraded_metrics structure")
                            elif isinstance(metrics, dict):
                                # Metrics are in a flat structure, need to sort them into degraded and restored
                                LOG.info("Restructuring flat metrics into degraded/restored categories")
                                
                                # Process metrics with 'degraded_' prefix or other indicators that they are degraded
                                for key, value in metrics.items():
                                    # Skip generation_params and timestamp
                                    if key in ['generation_params', 'timestamp', 'steps', 'sample_rate', 'sample_size']:
                                        continue
                                        
                                    # Put degraded_ prefixed metrics into degraded_metrics
                                    if key.startswith('degraded_'):
                                        # Remove the "degraded_" prefix for cleaner naming
                                        clean_key = key.replace('degraded_', '')
                                        degraded_metrics[clean_key] = value
                                    # Put restoration success metrics into restored metrics
                                    elif key.startswith('restoration_success_'):
                                        restored_metrics[key] = value
                                    # Any demo_ metrics go to restored (these are the model outputs)
                                    elif key.startswith('demo_'):
                                        # Remove the "demo_" prefix for cleaner naming
                                        clean_key = key.replace('demo_', '')
                                        restored_metrics[clean_key] = value
                                    # Handle losses
                                    elif key.endswith('_loss'):
                                        if 'degraded_' + key in metrics:
                                            restored_metrics[key] = value
                                        else:
                                            # If we don't have a degraded version, this might be a shared metric
                                            degraded_metrics[key] = value
                                            restored_metrics[key] = value
                                            
                                # Copy basic parameters to both metrics
                                for key in ['timestamp', 'steps', 'sample_rate', 'sample_size']:
                                    if key in metrics:
                                        degraded_metrics[key] = metrics[key]
                                        restored_metrics[key] = metrics[key]
                                        
                            LOG.info(f"Restructured metrics - degraded: {len(degraded_metrics)} keys, restored: {len(restored_metrics)} keys")
                            if len(degraded_metrics) > 0:
                                LOG.info(f"Degraded metrics keys: {list(degraded_metrics.keys())[:5]}...")
                            if len(restored_metrics) > 0:
                                LOG.info(f"Restored metrics keys: {list(restored_metrics.keys())[:5]}...")
                            
                            # Update degraded path and losses
                            degraded_basename = f"{audio_name}_{duration}_{intensity}_{effect_name}{ext}"
                            formatted_degraded_path = f"audio/degraded/{degraded_basename}"
                            
                            # Update the existing entry with correct format
                            existing_effect["degraded"]["audio"] = formatted_degraded_path
                            
                            # Preserve all metrics from _calculate_metrics and ensure they are scalars, not arrays
                            degraded_losses = {}
                            for metric_key, metric_value in degraded_metrics.items():
                                # Convert array values to scalars
                                if isinstance(metric_value, list) and len(metric_value) > 0:
                                    degraded_losses[metric_key] = metric_value[0]
                                else:
                                    degraded_losses[metric_key] = metric_value
                            
                            existing_effect["degraded"]["losses"] = degraded_losses
                            
                            # Copy the degraded audio file to the degraded folder
                            degraded_filepath = os.path.join(degraded_audio_dir, degraded_basename)
                            LOG.info(f"Saving degraded audio to: {degraded_filepath}")
                            try:
                                if os.path.exists(degraded_audio_path):
                                    shutil.copy2(degraded_audio_path, degraded_filepath)
                                else:
                                    LOG.warning(f"Source degraded audio not found: {degraded_audio_path}")
                            except Exception as e:
                                LOG.error(f"Error copying degraded audio file: {str(e)}")
                            
                            # Update restored entry too
                            json_restored_path = f"audio/restored/{audio_name}_{duration}_{intensity}_{effect_name}_restored{ext}"
                            existing_effect["restored"]["audio"] = json_restored_path
                            
                            # Preserve all metrics from _calculate_metrics and ensure they are scalars, not arrays
                            restored_losses = {}
                            for metric_key, metric_value in restored_metrics.items():
                                # Convert array values to scalars
                                if isinstance(metric_value, list) and len(metric_value) > 0:
                                    restored_losses[metric_key] = metric_value[0]
                                else:
                                    restored_losses[metric_key] = metric_value
                                    
                            existing_effect["restored"]["losses"] = restored_losses
                            break
                    
                    if not effect_exists:
                        # Get complete metrics from the JSON file generated during restoration
                        src_metrics_path = os.path.join(os.path.dirname(audio), f"{original_filename_without_ext}_metrics.json")
                        # Use metrics in the same format as model.json
                        # Initialize with empty structure
                        detailed_metrics = {
                            "degraded_metrics": {},
                            "restored_metrics": {}
                        }
                        
                        # Get metrics from the model via metrics (from final_metrics)
                        if metrics and isinstance(metrics, dict):
                            if "degraded_metrics" in metrics:
                                detailed_metrics["degraded_metrics"] = metrics["degraded_metrics"]
                                LOG.info(f"Retrieved {len(metrics['degraded_metrics'])} degraded metrics from final_metrics")
                            if "restored_metrics" in metrics:
                                detailed_metrics["restored_metrics"] = metrics["restored_metrics"]
                                LOG.info(f"Retrieved {len(metrics['restored_metrics'])} restored metrics from final_metrics")
                        else:
                            LOG.warning("No metrics available in final_metrics!")
                        
                        if os.path.exists(src_metrics_path):
                            try:
                                with open(src_metrics_path, 'r') as f:
                                    loaded_metrics = json.load(f)
                                    if "degraded_metrics" in loaded_metrics:
                                        detailed_metrics["degraded_metrics"] = loaded_metrics["degraded_metrics"]
                                    if "restored_metrics" in loaded_metrics:
                                        detailed_metrics["restored_metrics"] = loaded_metrics["restored_metrics"]
                                LOG.info(f"Loaded detailed metrics from {src_metrics_path}")
                            except Exception as e:
                                LOG.warning(f"Could not load detailed metrics from {src_metrics_path}: {str(e)}")
                        else:
                            LOG.warning(f"Metrics file not found: {src_metrics_path}, using default metrics")
                        
                        # Build paths with the exact format from model.json
                        # Format degraded: audio/degraded/AUDIO_NAME_DURATION_INTENSITY_EFFECT.wav
                        degraded_basename = f"{audio_name}_{duration}_{intensity}_{effect_name}{ext}"
                        formatted_degraded_path = f"audio/degraded/{degraded_basename}"
                        
                        # Format restored: audio/restored/AUDIO_NAME_DURATION_INTENSITY_EFFECT_restored.wav
                        json_restored_path = f"audio/restored/{audio_name}_{duration}_{intensity}_{effect_name}_restored{ext}"
                        
                        # Copy the degraded audio file to the degraded folder
                        degraded_filepath = os.path.join(degraded_audio_dir, degraded_basename)
                        LOG.info(f"Saving degraded audio to: {degraded_filepath}")
                        try:
                            if os.path.exists(degraded_audio_path):
                                shutil.copy2(degraded_audio_path, degraded_filepath)
                            else:
                                LOG.warning(f"Source degraded audio not found: {degraded_audio_path}")
                        except Exception as e:
                            LOG.error(f"Error copying degraded audio file: {str(e)}")
                        
                        # Apply the same metrics restructuring logic for new entries
                        degraded_metrics = {}
                        restored_metrics = {}
                        
                        # First check if we have nested metrics structure from detailed_metrics
                        if "degraded_metrics" in detailed_metrics and isinstance(detailed_metrics["degraded_metrics"], dict):
                            degraded_metrics = detailed_metrics["degraded_metrics"]
                            if "restored_metrics" in detailed_metrics and isinstance(detailed_metrics["restored_metrics"], dict):
                                restored_metrics = detailed_metrics["restored_metrics"]
                            LOG.info("Using nested metrics structure from detailed_metrics")
                            
                        # If not, check if metrics has the nested structure
                        elif "degraded_metrics" in metrics and isinstance(metrics["degraded_metrics"], dict):
                            degraded_metrics = metrics["degraded_metrics"]
                            if "restored_metrics" in metrics and isinstance(metrics["restored_metrics"], dict):
                                restored_metrics = metrics["restored_metrics"]
                            LOG.info("Using nested metrics structure from metrics")
                            
                        # If we still don't have metrics, try to restructure flat metrics
                        elif isinstance(metrics, dict):
                            # Metrics are in a flat structure, need to sort them into degraded and restored
                            LOG.info("Restructuring flat metrics into degraded/restored categories for new entry")
                            
                            # Process metrics with 'degraded_' prefix or other indicators that they are degraded
                            for key, value in metrics.items():
                                # Skip generation_params and timestamp
                                if key in ['generation_params', 'timestamp', 'steps', 'sample_rate', 'sample_size']:
                                    continue
                                    
                                # Put degraded_ prefixed metrics into degraded_metrics
                                if key.startswith('degraded_'):
                                    # Remove the "degraded_" prefix for cleaner naming
                                    clean_key = key.replace('degraded_', '')
                                    degraded_metrics[clean_key] = value
                                # Put restoration success metrics into restored metrics
                                elif key.startswith('restoration_success_'):
                                    restored_metrics[key] = value
                                # Any demo_ metrics go to restored (these are the model outputs)
                                elif key.startswith('demo_'):
                                    # Remove the "demo_" prefix for cleaner naming
                                    clean_key = key.replace('demo_', '')
                                    restored_metrics[clean_key] = value
                                # Handle losses
                                elif key.endswith('_loss'):
                                    if 'degraded_' + key in metrics:
                                        restored_metrics[key] = value
                                    else:
                                        # If we don't have a degraded version, this might be a shared metric
                                        degraded_metrics[key] = value
                                        restored_metrics[key] = value
                                        
                            # Copy basic parameters to both metrics
                            for key in ['timestamp', 'steps', 'sample_rate', 'sample_size']:
                                if key in metrics:
                                    degraded_metrics[key] = metrics[key]
                                    restored_metrics[key] = metrics[key]
                        
                        LOG.info(f"Final metrics for new entry - degraded: {len(degraded_metrics)} keys, restored: {len(restored_metrics)} keys")
                        
                        # Format losses in the correct structure - all metrics, ensuring they are single values
                        # Process degraded metrics - preserve all metrics but ensure they are scalar values
                        degraded_losses = {}
                        for metric_key, metric_value in degraded_metrics.items():
                            # Convert array values to scalars
                            if isinstance(metric_value, list) and len(metric_value) > 0:
                                degraded_losses[metric_key] = metric_value[0]
                            else:
                                degraded_losses[metric_key] = metric_value
                        
                        # Process restored metrics - preserve all metrics but ensure they are scalar values
                        restored_losses = {}
                        for metric_key, metric_value in restored_metrics.items():
                            # Convert array values to scalars
                            if isinstance(metric_value, list) and len(metric_value) > 0:
                                restored_losses[metric_key] = metric_value[0]
                            else:
                                restored_losses[metric_key] = metric_value
                        
                        effect_entry = {
                            "name": current_effect,
                            "degraded": {
                                "losses": degraded_losses,  # Include ALL metrics
                                "audio": formatted_degraded_path  # Formatted path to degraded audio
                            },
                            "restored": {
                                "losses": restored_losses,  # Include ALL metrics
                                "audio": json_restored_path  # Format restored audio path
                            }
                        }
                        
                        LOG.info(f"Created entry for {current_effect} with {len(degraded_losses)} degraded metrics and {len(restored_losses)} restored metrics")
                        
                        # Add to the corresponding effects array
                        consolidated_results["data"][audio_name][duration][intensity]["effects"].append(effect_entry)
                    
                    # Save the consolidated results after each file to avoid loss if interrupted
                    results_filename = f"results_{model_name}.json"
                    results_path = os.path.join(output_dir, results_filename)
                    with open(results_path, "w") as f:
                        json.dump(consolidated_results, f, indent=4)
                    
                    LOG.info(f"Updated consolidated results in {results_path}")
                    audio = new_filepath
                else:
                    # Standard handling if effect not identified
                    LOG.warning(f"Could not identify effect for file {degraded_audio_path}")
                    # Use basename to avoid paths in filenames
                    audio_basename = os.path.basename(os.path.dirname(degraded_audio_path))
                    # Create a fallback name with identifiable components
                    fallback_name = f"{audio_basename}_{original_filename_without_ext}_restored{ext}"
                    new_filepath = os.path.join(restored_audio_dir, fallback_name)
                    LOG.info(f"Saving restored audio to: {new_filepath} (effect not identified)")
                    shutil.copy2(audio, new_filepath)
                    audio = new_filepath
            elif process_folder_path and output_dir:
                # Standard folder processing (without effects list)
                original_filename = os.path.basename(degraded_audio_path)
                original_filename_without_ext = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(audio)[1]
                restored_filename = f"{original_filename_without_ext}_restored{ext}"
                
                # Copy the audio file to our custom output directory
                new_filepath = os.path.join(output_dir, restored_filename)
                LOG.info(f"Copying restored audio from {audio} to {new_filepath}")
                shutil.copy2(audio, new_filepath)
                
                # Copy the metrics file to our custom output directory
                src_metrics_path = os.path.join(os.path.dirname(audio), f"{original_filename_without_ext}_metrics.json")
                dest_metrics_path = os.path.join(output_dir, f"{original_filename_without_ext}_metrics.json")
                if os.path.exists(src_metrics_path):
                    LOG.info(f"Copying metrics from {src_metrics_path} to {dest_metrics_path}")
                    shutil.copy2(src_metrics_path, dest_metrics_path)
                
                audio = new_filepath
            else:
                # Default behavior for uploaded files
                original_filename = os.path.basename(degraded_audio_path)
                original_filename_without_ext = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(audio)[1]
                restored_filename = f"{original_filename_without_ext}_restored{ext}"
                
                output_dir = os.path.dirname(audio)
                new_filepath = os.path.join(output_dir, restored_filename)
                
                os.rename(audio, new_filepath)
                audio = new_filepath
            
            all_metrics.append(metrics)
            output_audios_list.append(new_filepath)
            output_spectrograms_list.extend(spectrograms)
        
        # Disable plots for batch processing
        if model_name and effects_list:
            plots = None
        elif clean_audio is not None:
            plots = create_metric_plots(all_metrics, labels) if all_metrics else None
        else:
            plots = None
        
        return output_audios_list, output_spectrograms_list, plots

    def generate_with_plots(*args):
        if is_audio_restoration:
            try:
                audios, spectrograms, plots = generate_multiple_with_plots(*args)
                first_audio = audios[0] if audios else None
                return gr.update(choices=audios, value=first_audio), first_audio, spectrograms, plots
            except Exception as e:
                LOG.error(f"Error in generate_with_plots: {str(e)}")
                import traceback
                LOG.error(traceback.format_exc())
                raise gr.Error(f"Error processing audios: {str(e)}")
        else:
            audio, spectrograms = generate_cond(*args)
            return gr.update(choices=[audio], value=audio), audio, spectrograms, gr.update(value=None, visible=False)

    generate_button.click(
        fn=generate_with_plots,
        inputs=inputs,
        outputs=[output_audio_dropdown, output_audio_player, audio_spectrogram_output, metrics_plots],
        api_name="generate",
    )

    def select_audio(audio_file):
        return audio_file
    
    output_audio_dropdown.change(fn=select_audio, inputs=output_audio_dropdown, outputs=output_audio_player)

    if is_audio_restoration:
        def update_degraded_dropdown(files):
            if not files:
                return gr.update(choices=[], value=None), None
            
            choices = [(os.path.basename(f.name), f.name) for f in files]
            first_filepath = choices[0][1] if choices else None

            return gr.update(choices=choices, value=first_filepath), first_filepath

        degraded_audio_files.upload(
            fn=update_degraded_dropdown,
            inputs=[degraded_audio_files],
            outputs=[degraded_audio_dropdown, degraded_audio_player]
        )
        
        degraded_audio_dropdown.change(
            fn=lambda x: x, 
            inputs=[degraded_audio_dropdown], 
            outputs=[degraded_audio_player]
        )

def create_diffusion_cond_ui(model_config, in_model, in_model_half=True):
    global model, sample_size, sample_rate, model_type, model_half

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_type = model_config["model_type"]

    model_half = in_model_half

    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config)
    return ui
