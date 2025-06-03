import gc
import numpy as np
import gradio as gr
import json
import re
import subprocess
import torch
import torchaudio
import threading
import os, time
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
LOG.setLevel(logging.DEBUG)

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
    current_time = time.strftime("%H-%M-%S")
    output_dir = os.path.join("output", current_date)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for this specific generation
    generation_dir = os.path.join(output_dir, f"generation_{current_time}")
    os.makedirs(generation_dir, exist_ok=True)

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

    input_sample_size = sample_size

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
        "conditioning_tensors": conditioning_tensors,  # Utiliser les tenseurs pré-traités plutôt que le dictionnaire brut
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

    if model_type == "diffusion_cond":
        # Do the audio generation
        audio = generate_diffusion_cond(**generate_args)

    elif model_type == "diffusion_cond_inpaint":
        if inpaint_audio is not None:
            # Convert mask start and end from percentages to sample indices
            mask_start = int(mask_maskstart * sample_rate)
            mask_end = int(mask_maskend * sample_rate)

            inpaint_mask = torch.ones(1, sample_size, device=device)
            inpaint_mask[:, mask_start:mask_end] = 0

            generate_args.update(
                {"inpaint_audio": inpaint_audio, "inpaint_mask": inpaint_mask}
            )

        audio = generate_diffusion_cond_inpaint(**generate_args)
    
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


def compute_metrics(audio, clean_audio=None, degraded_audio=None, model=None, device="cuda"):
    """Compute audio quality metrics between the generated audio and clean reference if provided."""
    metrics = {}
    
    if clean_audio is not None and model is not None:
        from stable_audio_tools.training.losses.metrics import (
            LogSpectralDistance,
            LTASDistance,
            SISDRMetric,
            SNRMetric,
            STFTDistance,
            MelDistance
        )
        
        # Move tensors to the correct device and ensure they're float32
        audio = audio.to(device)
        clean_audio = clean_audio.to(device)
        
        LOG.info(f"Audio shapes - Generated: {audio.shape}, Clean: {clean_audio.shape}")
        LOG.info(f"Audio dtypes - Generated: {audio.dtype}, Clean: {clean_audio.dtype}")
        LOG.info(f"Audio ranges - Generated: [{audio.min().item():.3f}, {audio.max().item():.3f}], Clean: [{clean_audio.min().item():.3f}, {clean_audio.max().item():.3f}]")
        
        # If using a pretransform model, process both audios through the same pipeline
        if model.pretransform is not None:
            # Encode both audios to latent space
            clean_audio = clean_audio.to(device)
            audio = audio.to(device)
            degraded_audio = degraded_audio.to(device)
            clean_audio_latent = model.pretransform.encode(clean_audio.unsqueeze(0))
            audio_latent = model.pretransform.encode(audio.unsqueeze(0))
            degraded_audio_latent = model.pretransform.encode(degraded_audio.unsqueeze(0))
            # Decode back to waveform space for consistent comparison
            clean_audio = model.pretransform.decode(clean_audio_latent)
            degraded_audio = model.pretransform.decode(degraded_audio_latent)
            
            # Remove batch dimension and ensure correct shape
            audio = audio.squeeze(0)
            clean_audio = clean_audio.squeeze(0)
            degraded_audio = degraded_audio.squeeze(0)
            LOG.info(f"After transform - Audio shapes - Generated: {audio.shape}, Clean: {clean_audio.shape}, Degraded: {degraded_audio.shape}")
        
        # Ensure both audios are the same length
        if audio.shape[-1] != clean_audio.shape[-1]:
            min_length = min(audio.shape[-1], clean_audio.shape[-1])
            audio = audio[..., :min_length]
            clean_audio = clean_audio[..., :min_length]
            LOG.info(f"After length adjustment - Audio shapes - Generated: {audio.shape}, Clean: {clean_audio.shape}")
        
        metrics_instances = {
            'lsd': LogSpectralDistance().to(device),
            'ltas': LTASDistance().to(device),
            'sisdr': SISDRMetric().to(device),
            'snr': SNRMetric().to(device),
            'stft': STFTDistance().to(device),
            'mel': MelDistance(sample_rate=model.sample_rate).to(device)
        }
        
        # Calculate metrics between generated and clean audio
        for name, metric in metrics_instances.items():
            try:
                # Calculate demo metric (generated vs clean)
                demo_metric = metric(audio, clean_audio).item()
                metrics[name] = demo_metric
                
                # Calculate degraded metric (degraded vs clean)
                degraded_metric = metric(degraded_audio, clean_audio).item()
                
                # Calculate clean metric (clean vs clean)
                clean_metric = metric(clean_audio, model.pretransform.decode(model.pretransform.encode(clean_audio.unsqueeze(0))).squeeze(0)).item()
                
                # Calculate restoration success metric
                restoration_success = 1 - abs((demo_metric - degraded_metric)) / abs((clean_metric - degraded_metric))
                metrics[f'restoration_success_{name}'] = restoration_success
                
                LOG.info(f"Computed {name}: {demo_metric:.3f}")
                LOG.info(f"Restoration success for {name}: {restoration_success:.3f}")
            except Exception as e:
                LOG.error(f"Error computing {name}: {str(e)}")
        
        # Latent space losses if using a pretransform model
        if model.pretransform is not None:
            try:
                LOG.info("Computing latent space metrics...")
                # MSE Loss
                demo_mse = F.mse_loss(audio_latent, clean_audio_latent)
                degraded_mse = F.mse_loss(degraded_audio_latent, clean_audio_latent)
                clean_mse = F.mse_loss(clean_audio_latent, clean_audio_latent)
                metrics['latent_mse_loss'] = demo_mse.item()
                metrics['restoration_success_latent_mse_loss'] = 1 - abs((demo_mse.item() - degraded_mse.item())) / abs((clean_mse.item() - degraded_mse.item()))
                LOG.info(f"Latent MSE Loss: {demo_mse.item():.3f}")
                LOG.info(f"Restoration Success Latent MSE: {metrics['restoration_success_latent_mse_loss']:.3f}")
                
                # L1 Loss
                demo_l1 = F.l1_loss(audio_latent, clean_audio_latent)
                degraded_l1 = F.l1_loss(degraded_audio_latent, clean_audio_latent)
                clean_l1 = F.l1_loss(clean_audio_latent, clean_audio_latent)
                metrics['latent_l1_loss'] = demo_l1.item()
                metrics['restoration_success_latent_l1_loss'] = 1 - abs((demo_l1.item() - degraded_l1.item())) / abs((clean_l1.item() - degraded_l1.item()))
                LOG.info(f"Latent L1 Loss: {demo_l1.item():.3f}")
                LOG.info(f"Restoration Success Latent L1: {metrics['restoration_success_latent_l1_loss']:.3f}")
            except Exception as e:
                LOG.error(f"Error computing latent space metrics: {str(e)}")
        
        # Waveform domain losses
        try:
            LOG.info("Computing waveform domain losses...")
            # MSE Loss
            demo_waveform_mse = F.mse_loss(audio, clean_audio)
            degraded_waveform_mse = F.mse_loss(degraded_audio, clean_audio)
            clean_waveform_mse = F.mse_loss(clean_audio, clean_audio)
            metrics['waveform_mse_loss'] = demo_waveform_mse.item()
            metrics['restoration_success_waveform_mse_loss'] = 1 - abs((demo_waveform_mse.item() - degraded_waveform_mse.item())) / abs((clean_waveform_mse.item() - degraded_waveform_mse.item()))
            LOG.info(f"Waveform MSE Loss: {demo_waveform_mse.item():.3f}")
            LOG.info(f"Restoration Success Waveform MSE: {metrics['restoration_success_waveform_mse_loss']:.3f}")
            
            # L1 Loss
            demo_waveform_l1 = F.l1_loss(audio, clean_audio)
            degraded_waveform_l1 = F.l1_loss(degraded_audio, clean_audio)
            clean_waveform_l1 = F.l1_loss(clean_audio, clean_audio)
            metrics['waveform_l1_loss'] = demo_waveform_l1.item()
            metrics['restoration_success_waveform_l1_loss'] = 1 - abs((demo_waveform_l1.item() - degraded_waveform_l1.item())) / abs((clean_waveform_l1.item() - degraded_waveform_l1.item()))
            LOG.info(f"Waveform L1 Loss: {demo_waveform_l1.item():.3f}")
            LOG.info(f"Restoration Success Waveform L1: {metrics['restoration_success_waveform_l1_loss']:.3f}")
        except Exception as e:
            LOG.error(f"Error computing waveform domain losses: {str(e)}")
    
    return metrics


def generate_cond_restoration(
    steps=250,
    preview_every=None,
    metrics_every=0,  # New parameter for metrics computation frequency
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
):
    LOG.info("Starting audio restoration")
    
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

    input_sample_size = sample_size

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
        elif degraded_audio.dim() == 2:
            degraded_audio = degraded_audio.transpose(0, 1)

        if in_sr != sample_rate:
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(degraded_audio.device)
                .to(degraded_audio.dtype)
            )
            degraded_audio = resample_tf(degraded_audio)

        audio_length = degraded_audio.shape[-1]

        if audio_length > sample_size:
            degraded_audio = degraded_audio[:, :sample_size]

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
        if audio_length > sample_size:
            clean_audio = clean_audio[:, :sample_size]
        
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

        # Compute metrics at specified intervals if metrics_every is set
        if metrics_every > 0 and (current_step) % metrics_every == 0:
            LOG.info(f"Computing metrics at step {current_step}")
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1)
            
            # Compute metrics for this step
            step_metrics = compute_metrics(
                denoised, 
                clean_audio[1] if clean_audio is not None else None,
                degraded_audio[1] if degraded_audio is not None else None,
                model=model,
                device=device
            )

            # Create a new dictionary for this step's metrics
            step_data = {
                "metrics": step_metrics,
                "step": current_step,
                "sigma": float(sigma)
            }
            
            # Store in the metrics dictionary with the step number as key
            metrics_dict[f"step_{current_step}"] = step_data
    # Create date-based directory structure
    current_date = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H-%M-%S")
    output_dir = os.path.join("output", current_date)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for this specific generation
    generation_dir = os.path.join(output_dir, f"generation_{current_time}")
    os.makedirs(generation_dir, exist_ok=True)

    # Set output filenames in the generation directory
    output_wav = os.path.join(generation_dir, "output.wav")
    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else:
        filename_extension = "wav"
    output_filename = os.path.join(generation_dir, f"output.{filename_extension}")

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
        "callback": progress_callback if (preview_every is not None or metrics_every > 0) else None,
        "scale_phi": cfg_rescale,
        "rho": rho,
        "clean_audio": clean_audio,
        "output_dir": generation_dir
    }

    # Do the audio generation
    LOG.info("Generating audio")
    
    audio, final_metrics = generate_diffusion_cond(**generate_args)

    # Combine per-step metrics with final metrics
    if final_metrics is not None:
        # Add per-step metrics to the final metrics
        final_metrics["step_metrics"] = metrics_dict
        # Add generation parameters
        final_metrics["generation_params"] = {
            "steps": steps,
            "metrics_every": metrics_every,
            "preview_every": preview_every,
            "sampler_type": sampler_type,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "rho": rho
        }
        # Save metrics to JSON file
        metrics_file = os.path.join(generation_dir, "metrics.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            LOG.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            LOG.error(f"Error saving metrics to file: {e}")

    # Encode the audio to WAV format
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
    
    clean_audio = clean_audio[1]
    max_abs_val = torch.max(torch.abs(clean_audio))
    if max_abs_val > 1e-7:
        clean_audio = (
            clean_audio.to(torch.float32)
            .div(max_abs_val)
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
    else:
        clean_audio = clean_audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Save the WAV file
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
                return (output_wav, preview_images, final_metrics)

    # Generate spectrogram
    try:
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
        # Generate clean audio spectrogram if available
        clean_spectrogram = None
        if clean_audio is not None:
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


def create_metric_plots(metrics_data):
    """Create plots for all metrics over steps."""
    if not metrics_data or "step_metrics" not in metrics_data or not metrics_data["step_metrics"]:
        return None
        
    try:
        # Get all metric names from the first step
        first_step = next(iter(metrics_data["step_metrics"].values()))
        all_metric_names = list(first_step["metrics"].keys())
        
        if not all_metric_names:
            return None
            
        # Separate original metrics from restoration success metrics
        metric_names = [name for name in all_metric_names if not name.startswith('restoration_success_')]
            
        # Create a figure with subplots for each metric plus one for combined restoration
        n_metrics = len(metric_names)
        n_cols = 2
        n_rows = (n_metrics + 2) // 2  # +2 to account for the combined restoration plot
        
        fig = plt.figure(figsize=(15, 5 * n_rows))
        
        # Extract steps and values for each metric
        steps = []
        values = {name: [] for name in metric_names}
        restoration_values = {name: [] for name in metric_names}
        
        for step_data in sorted(metrics_data["step_metrics"].values(), key=lambda x: x["step"]):
            steps.append(step_data["step"])
            for name in metric_names:
                # Get original metric
                values[name].append(step_data["metrics"][name])
                # Get restoration success metric
                restoration_name = f'restoration_success_{name}'
                if restoration_name in step_data["metrics"]:
                    restoration_values[name].append(step_data["metrics"][restoration_name])
        # Add final step with demo metrics if available
        final_step = metrics_data["steps"]
        steps.append(final_step)
        
        # Original metrics
        demo_metrics = {
            'lsd': metrics_data['demo_lsd'],
            'ltas': metrics_data['demo_ltas'],
            'sisdr': metrics_data['demo_sisdr'],
            'snr': metrics_data['demo_snr'],
            'stft': metrics_data['demo_stft'],
            'mel': metrics_data['demo_mel'],
            'latent_mse_loss': metrics_data['latent_mse_loss'],
            'latent_l1_loss': metrics_data['latent_l1_loss'],
            'waveform_mse_loss': metrics_data['waveform_mse_loss'],
            'waveform_l1_loss': metrics_data['waveform_l1_loss']
        }
        
        # Restoration success metrics
        restoration_metrics = {
            'lsd': metrics_data['restoration_success_lsd'],
            'ltas': metrics_data['restoration_success_ltas'],
            'sisdr': metrics_data['restoration_success_sisdr'],
            'snr': metrics_data['restoration_success_snr'],
            'stft': metrics_data['restoration_success_stft'],
            'mel': metrics_data['restoration_success_mel'],
            'latent_mse_loss': metrics_data['restoration_success_latent_mse_loss'],
            'latent_l1_loss': metrics_data['restoration_success_latent_l1_loss'],
            'waveform_mse_loss': metrics_data['restoration_success_waveform_mse_loss'],
            'waveform_l1_loss': metrics_data['restoration_success_waveform_l1_loss']
        }
        
        # Add final values for both original and restoration metrics
        for name in metric_names:
            values[name].append(demo_metrics[name])
            restoration_name = f'restoration_success_{name}'
            restoration_values[name].append(restoration_metrics[name])
    
        # Create subplots for individual metrics
        for i, name in enumerate(metric_names, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.plot(steps, values[name], 'b-', label=f'{name} (raw)')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.title(f'{name} over steps')
            plt.grid(True)

            # Add a marker for the final demo metric if available
            plt.plot(final_step, demo_metrics[name], 'ro', label='Final step')
            plt.legend()
        
        # Create combined restoration success plot
        plt.subplot(n_rows, n_cols, n_metrics + 1)
        for name in metric_names:
            plt.plot(steps, restoration_values[name], label=f'{name}')
        
        plt.xlabel('Step')
        plt.ylabel('Restoration Success')
        plt.title('Combined Restoration Success Metrics')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Convert plot to image and return as numpy array
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Convert to numpy array for Gradio
        from PIL import Image
        img = Image.open(buf)
        return np.array(img)
    except Exception as e:
        LOG.error(f"Error creating metric plots: {str(e)}")
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
    
    # Interface adaptée au type de modèle
    with gr.Row():
        with gr.Column(scale=6):
            # L'entrée texte n'est visible que si le modèle n'est pas de type restauration audio
            prompt_visible = not is_audio_restoration
            prompt = gr.Textbox(show_label=False, placeholder="Prompt", visible=prompt_visible)
            negative_prompt = gr.Textbox(
                show_label=False, placeholder="Negative prompt", visible=prompt_visible
            )
            
            # Information message for audio restoration model
            if is_audio_restoration:
                gr.Markdown("### Audio Restoration Model\nUpload an audio file to restore below")
                
        generate_button = gr.Button("Restore", variant="primary", scale=1)
        
    with gr.Row(equal_height=False):
        with gr.Column():
            # Variables pour suivre si ces conditionnements sont nécessaires
            has_seconds_start = False
            has_seconds_total = False
            
            if model_conditioning_config:
                for config in model_conditioning_config.get("configs", []):
                    if config.get("id") == "seconds_start":
                        has_seconds_start = True
                    if config.get("id") == "seconds_total":
                        has_seconds_total = True
            
            # N'afficher les contrôles de timing que si nécessaire et pas pour la restauration audio
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
                default_steps = 25 if is_audio_restoration else (50 if is_rf else 100)
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
                # Output params
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
                        value=0,
                        label="Compute Metrics Every N Steps",
                        visible=is_audio_restoration,
                    )
            if is_audio_restoration:
                with gr.Accordion("Audio Inputs", open=True):
                    degraded_audio = gr.Audio(label="Degraded audio")
                    clean_audio = gr.Audio(label="Clean reference audio (optional)")
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
                        degraded_audio,
                        clean_audio,
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
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(
                label="Output spectrograms", show_label=True,
                elem_id="spectrogram_gallery",
                columns=2,
                height=400
            )
            
            # Add metrics display section
            with gr.Accordion("Restoration Metrics", open=True, visible=is_audio_restoration):
                metrics_plots = gr.Image(label="Metrics Plots", visible=is_audio_restoration, type="numpy")
            
            # Use different target based on model type
            if is_audio_restoration:
                send_to_init_button = gr.Button("Send to degraded audio", scale=1)
                send_to_init_button.click(
                    fn=lambda audio: audio,
                    inputs=[audio_output],
                    outputs=[degraded_audio],
                )
            else:
                send_to_init_button = gr.Button("Send to init audio", scale=1)
                send_to_init_button.click(
                    fn=lambda audio: audio,
                    inputs=[audio_output],
                    outputs=[init_audio_input],
                )

            if has_inpainting:
                send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
                send_to_inpaint_button.click(
                    fn=lambda audio: audio,
                    inputs=[audio_output],
                    outputs=[inpaint_audio_input],
                )

    def generate_with_plots(*args):
        if is_audio_restoration:
            audio, spectrograms, metrics = generate_cond_restoration(*args)
            plots = create_metric_plots(metrics) if metrics else None
            return audio, spectrograms, plots
        else:
            audio, spectrograms = generate_cond(*args)
            return audio, spectrograms, None

    generate_button.click(
        fn=generate_with_plots,
        inputs=inputs,
        outputs=[audio_output, audio_spectrogram_output, metrics_plots] if is_audio_restoration else [audio_output, audio_spectrogram_output, None],
        api_name="generate",
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
