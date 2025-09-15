import gc
import logging
import re
import matplotlib
import numpy as np
import torch
from einops import rearrange
from torchaudio import transforms as T

from ...inference.generation import (
    generate_diffusion_cond_restoration, generate_diffusion_uncond_restoration
)
from ..aeiou import audio_spectrogram_image

matplotlib.use("Agg")
# Suppress verbose matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = False


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
    t_start: float = 1.0,
    schedule: str = "linear",
    save_metrics: bool = True,
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

    input_sample_size = degraded_audio[1].shape[-1]

    if degraded_audio is not None:
        if not isinstance(degraded_audio, tuple) or len(degraded_audio) != 2:
            raise ValueError(f"Invalid audio format: {degraded_audio}. Expected a tuple of (sample_rate, audio_data).")

        in_sr, degraded_audio = degraded_audio

        if isinstance(degraded_audio, np.ndarray):
            if degraded_audio.dtype == np.float32:
                degraded_audio = torch.from_numpy(degraded_audio)
            elif degraded_audio.dtype == np.int16:
                degraded_audio = torch.from_numpy(degraded_audio).float().div(32767)
            elif degraded_audio.dtype == np.int32:
                degraded_audio = torch.from_numpy(degraded_audio).float().div(2147483647)
            else:
                raise ValueError(f"Unsupported audio data type: {degraded_audio.dtype}")
        elif not isinstance(degraded_audio, torch.Tensor):
            raise ValueError(f"Unsupported audio type: {type(degraded_audio)}")

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
            LOG.info(f"Truncating audio to match lengths: {audio_length} -> {input_sample_size}")
            degraded_audio = degraded_audio[:, :input_sample_size]

        if degraded_audio.shape[0] == 1:
            degraded_audio = degraded_audio.repeat(2, 1)
        
        degraded_audio = (sample_rate, degraded_audio)

    # Handle clean audio if provided
    if clean_audio is not None:
        in_sr, clean_audio = clean_audio
        
        if isinstance(clean_audio, np.ndarray):
            if clean_audio.dtype == np.float32:
                clean_audio = torch.from_numpy(clean_audio)
            elif clean_audio.dtype == np.int16:
                clean_audio = torch.from_numpy(clean_audio).float().div(32767)
            elif clean_audio.dtype == np.int32:
                clean_audio = torch.from_numpy(clean_audio).float().div(2147483647)
            else:
                raise ValueError(f"Unsupported audio data type: {clean_audio.dtype}")
        elif not isinstance(clean_audio, torch.Tensor):
            raise ValueError(f"Unsupported audio type: {type(clean_audio)}")

        if model_half:
            clean_audio = clean_audio.to(torch.float16)

        if clean_audio.dim() == 1:
            clean_audio = clean_audio.unsqueeze(0)
        elif clean_audio.dim() == 2:
            # If shape is (time, channels) transpose to (channels, time). If already (channels, time), keep as is.
            try:
                LOG.debug(f"Clean audio tensor shape before channel/time fix: {tuple(clean_audio.shape)}")
            except Exception:
                pass
            if clean_audio.shape[0] > clean_audio.shape[1]:
                clean_audio = clean_audio.transpose(0, 1)
                try:
                    LOG.debug("Transposed clean audio from (time, channels) to (channels, time)")
                except Exception:
                    pass
            else:
                try:
                    LOG.debug("Clean audio already (channels, time); no transpose")
                except Exception:
                    pass

        if in_sr != sample_rate:
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(clean_audio.device)
                .to(clean_audio.dtype)
            )
            clean_audio = resample_tf(clean_audio)

        audio_length = clean_audio.shape[-1]
        # Truncate clean_audio to match the degraded_audio's length *after* processing
        degraded_audio_len = degraded_audio[1].shape[-1]
        if clean_audio.shape[-1] > degraded_audio_len:
            LOG.warning(f"Truncating clean audio to match degraded audio length: {clean_audio.shape[-1]} -> {degraded_audio_len}")
            clean_audio = clean_audio[:, :degraded_audio_len]
        
        if clean_audio.shape[0] == 1:
            clean_audio = clean_audio.repeat(2, 1)
        
        clean_audio = (sample_rate, clean_audio)

    # Final check to ensure audio lengths match
    if clean_audio is not None and degraded_audio is not None:
        LOG.info(f"Truncating audio to match lengths: {clean_audio[1].shape[-1]} -> {degraded_audio[1].shape[-1]}")
        min_len = min(degraded_audio[1].shape[-1], clean_audio[1].shape[-1])
        degraded_audio = (degraded_audio[0], degraded_audio[1][:, :min_len])
        clean_audio = (clean_audio[0], clean_audio[1][:, :min_len])

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


    # Do the audio generation
    LOG.info("Generating audio")
    
    if model_type in ["diffusion_cond_restoration", "diffusion_cond"]:
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
            "device": device,
            "sampler_type": sampler_type,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "callback": progress_callback if (preview_every is not None) else None,
            "scale_phi": cfg_rescale,
            "rho": rho,
            "clean_audio": clean_audio,
            "metrics_every": metrics_every
        }
        audio, final_metrics = generate_diffusion_cond_restoration(**generate_args)
    elif model_type in ["diffusion_uncond_restoration", "diffusion_uncond"]:
        generate_args = {
            "model": model,
            "model_type": model_type,
            "sample_rate": sample_rate,
            "steps": steps,
            "batch_size": batch_size,
            "sample_size": input_sample_size,
            "device": device,
            "callback": progress_callback if (preview_every is not None) else None,
            "clean_audio": clean_audio,
            "degraded_audio": degraded_audio,
            "effects_list": effects_list,
            "metrics_every": metrics_every,
            "sampler_type": sampler_type,
            "t_start": t_start,
            "schedule": schedule,
        }
        audio, final_metrics = generate_diffusion_uncond_restoration(**generate_args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    


    audio = rearrange(audio, "b d n -> d (b n)")

    # Check if the audio tensor is empty before normalization
    if audio.numel() == 0:
        LOG.warning("Generated audio is empty")
        return (None, sample_rate, preview_images, final_metrics)

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
    return (audio, sample_rate, images_to_show, final_metrics)


