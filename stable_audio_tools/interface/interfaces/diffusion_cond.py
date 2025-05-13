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

    generate_args = {
        "model": model,
        "conditioning": conditioning,
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

    # If inpainting, send mask args
    # This will definitely change in the future
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
    output_filename = "%s.%s" % (basename, filename_extension)
    output_wav = "%s.wav" % basename

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
    except Exception as e:
        print(f"Error saving WAV file {output_wav}: {e}")
        # If saving fails, return None for the path
        return (None, preview_images)

    # If file_format is other than wav, convert to other file format
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
    else:  # wav
        pass
    if cmd:
        cmd += " -loglevel error"  # make output less verbose in the cmd window
        subprocess.run(cmd, shell=True, check=True)

    # Let's look at a nice spectrogram too
    try:
        # Assuming audio_spectrogram_image can handle int16 tensor
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
    except Exception as e:
        print(f"Warning: Could not generate spectrogram: {e}")
        audio_spectrogram = None  # Set to None if generation fails

    # Asynchronously delete the files after returning the output file, so as to prevent clutter in the directory
    if file_naming in ["verbose", "prompt"]:
        delete_files_async([output_wav, output_filename], 30)

    return (output_filename, [audio_spectrogram, *preview_images])


def generate_cond_restoration(
    steps=250,
    preview_every=None,
    seed=-1,
    sampler_type="dpmpp-3m-sde",
    sigma_min=0.03,
    sigma_max=1000,
    rho=1.0,
    cfg_rescale=0.0,
    file_format="wav",
    file_naming="verbose",
    init_audio=None,
    degraded_audio=None,
    batch_size=1,
):
    LOG.debug("Starting generate_cond_restoration with parameters:")
    LOG.debug(f"  steps: {steps}, preview_every: {preview_every}, seed: {seed}")
    LOG.debug(f"  sampler_type: {sampler_type}, sigma_min: {sigma_min}, sigma_max: {sigma_max}")
    LOG.debug(f"  rho: {rho}, cfg_rescale: {cfg_rescale}, file_format: {file_format}")
    LOG.debug(f"  file_naming: {file_naming}, batch_size: {batch_size}")
    LOG.debug(f"  degraded_audio provided: {degraded_audio is not None}")
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

    input_sample_size = sample_size

    if degraded_audio is not None:
        LOG.debug("Degraded audio provided for conditioning")
        in_sr, degraded_audio = degraded_audio
        LOG.debug(f"Degraded audio - SR: {in_sr}, shape: {degraded_audio.shape if hasattr(degraded_audio, 'shape') else 'unknown'}, type: {type(degraded_audio)}, dtype: {degraded_audio.dtype if hasattr(degraded_audio, 'dtype') else 'unknown'}")

        if degraded_audio.dtype == np.float32:
            LOG.debug("Converting float32 numpy array to torch tensor")
            degraded_audio = torch.from_numpy(degraded_audio)
        elif degraded_audio.dtype == np.int16:
            LOG.debug("Converting int16 numpy array to normalized float torch tensor")
            degraded_audio = torch.from_numpy(degraded_audio).float().div(32767)
        elif degraded_audio.dtype == np.int32:
            LOG.debug("Converting int32 numpy array to normalized float torch tensor")
            degraded_audio = torch.from_numpy(degraded_audio).float().div(2147483647)
        else:
            LOG.debug(f"Unsupported audio data type: {degraded_audio.dtype}")
            raise ValueError(f"Unsupported audio data type: {degraded_audio.dtype}")

        if model_half:
            LOG.debug("Converting audio to float16 to match model precision")
            degraded_audio = degraded_audio.to(torch.float16)

        if degraded_audio.dim() == 1:
            LOG.debug("Reshaping 1D audio to [1, n]")
            degraded_audio = degraded_audio.unsqueeze(0)  # [1, n]
        elif degraded_audio.dim() == 2:
            LOG.debug("Transposing 2D audio from [n, 2] to [2, n]")
            degraded_audio = degraded_audio.transpose(0, 1)  # [n, 2] -> [2, n]
        LOG.debug(f"Reshaped degraded_audio tensor shape: {degraded_audio.shape}")

        if in_sr != sample_rate:
            LOG.debug(f"Resampling audio from {in_sr}Hz to {sample_rate}Hz")
            resample_tf = (
                T.Resample(in_sr, sample_rate)
                .to(degraded_audio.device)
                .to(degraded_audio.dtype)
            )
            degraded_audio = resample_tf(degraded_audio)
            LOG.debug(f"Resampled degraded_audio shape: {degraded_audio.shape}")

        audio_length = degraded_audio.shape[-1]
        LOG.debug(f"Audio length: {audio_length}, sample_size: {sample_size}")

        if audio_length > sample_size:
            LOG.debug(f"Audio too long ({audio_length} > {sample_size}), truncating to sample_size")
            # input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            degraded_audio = degraded_audio[:, :sample_size]
            LOG.debug(f"Truncated degraded_audio shape: {degraded_audio.shape}")

        degraded_audio = (sample_rate, degraded_audio)
        LOG.debug(f"Final degraded_audio tuple created: SR={sample_rate}, tensor.shape={degraded_audio[1].shape}")

    conditioning_dict = {
        "degraded_audio": degraded_audio[1],
    }

    conditioning = [conditioning_dict] * batch_size

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
        "init_audio": init_audio,
        # "degraded_audio": degraded_audio,   
        "callback": progress_callback if preview_every is not None else None,
        "scale_phi": cfg_rescale,
        "rho": rho,
    }
    LOG.debug("Prepared generate_args dictionary with keys:")
    for key, value in generate_args.items():
        if key not in ["model", "callback", "init_audio", "degraded_audio"]:
            LOG.debug(f"  {key}: {value}")
        else:
            LOG.debug(f"  {key}: {'<function>' if key == 'callback' else '<model object>' if key == 'model' else 'present' if value is not None else 'None'}")

    # Do the audio generation
    LOG.debug("Calling generate_diffusion_cond with prepared arguments")
    audio = generate_diffusion_cond(**generate_args)
    LOG.debug(f"Received audio from generate_diffusion_cond with shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}, type: {type(audio)}")

    # simple e.g. "output.wav"
    basename = "output"
    LOG.debug(f"Using base filename: {basename}")

    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else:
        filename_extension = "wav"
    LOG.debug(f"File extension: {filename_extension}")
    output_filename = "%s.%s" % (basename, filename_extension)
    output_wav = "%s.wav" % basename
    LOG.debug(f"Output filenames - wav: {output_wav}, final: {output_filename}")

    # Encode the audio to WAV format
    LOG.debug(f"Rearranging audio with shape {audio.shape}")
    audio = rearrange(audio, "b d n -> d (b n)")
    LOG.debug(f"Rearranged audio shape: {audio.shape}")

    # Check if the audio tensor is empty before normalization
    if audio.numel() == 0:
        LOG.debug("Audio tensor is empty, returning None for audio path")
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
    except Exception as e:
        print(f"Error saving WAV file {output_wav}: {e}")
        # If saving fails, return None for the path
        return (None, preview_images)

    # If file_format is other than wav, convert to other file format
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
    else:  # wav
        pass
    if cmd:
        cmd += " -loglevel error"  # make output less verbose in the cmd window
        subprocess.run(cmd, shell=True, check=True)

    # Let's look at a nice spectrogram too
    try:
        # Assuming audio_spectrogram_image can handle int16 tensor
        audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
    except Exception as e:
        print(f"Warning: Could not generate spectrogram: {e}")
        audio_spectrogram = None  # Set to None if generation fails

    # Asynchronously delete the files after returning the output file, so as to prevent clutter in the directory
    if file_naming in ["verbose", "prompt"]:
        delete_files_async([output_wav, output_filename], 30)

    return (output_filename, [audio_spectrogram, *preview_images])

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes.
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file

    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start()


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
                # CFG scale
                default_cfg = 1.0 if is_audio_restoration else 7.0
                cfg_scale_slider = gr.Slider(
                    minimum=0.0, maximum=25.0, step=0.1, value=default_cfg, label="CFG scale"
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
            if is_audio_restoration:
                with gr.Accordion("Degraded audio", open=False):
                    init_audio_input = gr.Audio(label="Init audio", visible=False)
                    degraded_audio = gr.Audio(label="Degraded audio")
                    min_noise_level = 0.01 if is_rf else 0.1
                    max_noise_level = 1.0 if is_rf else 100.0
                    init_noise_level_slider = gr.Slider(
                        minimum=min_noise_level,
                        maximum=max_noise_level,
                        step=0.01,
                        value=0.1,
                        label="Init noise level",
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
                LOG.debug("Audio restoration mode enabled")
                inputs = [
                        steps_slider,
                        preview_every_slider,
                        seed_textbox,
                        sampler_type_dropdown,
                        sigma_min_slider,
                        sigma_max_slider,
                        rho_slider,
                        cfg_rescale_slider,
                        file_format_dropdown,
                        file_naming_dropdown,
                        init_audio_input,
                        degraded_audio,
                    ]
            else:
                LOG.debug("Default generation mode enabled")
                inputs = [
                    prompt,
                    negative_prompt,
                    seconds_start_slider,
                    seconds_total_slider,
                    cfg_scale_slider,
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
                label="Output spectrogram", show_label=False
            )
            
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

    generate_button.click(
        fn=generate_cond_restoration if is_audio_restoration else generate_cond,
        inputs=inputs,
        outputs=[audio_output, audio_spectrogram_output],
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
