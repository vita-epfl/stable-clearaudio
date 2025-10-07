import gc
import numpy as np
import gradio as gr
import json 
import re
import subprocess
import torch
import torchaudio
import warnings
import io
from PIL import Image

# Supprimer les avertissements
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values")

# Modifier à la fois la fonction audio_spectrogram_image et appliquer un monkey patch direct à Gradio
from stable_audio_tools.interface.aeiou import audio_spectrogram_image as original_audio_spectrogram_image

# Définir une nouvelle fonction qui enveloppe l'originale pour garantir les images RGB
def wrapped_audio_spectrogram_image(*args, **kwargs):
    img = original_audio_spectrogram_image(*args, **kwargs)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img

# Remplacer la fonction originale par notre wrapper
from stable_audio_tools.interface import aeiou
aeiou.audio_spectrogram_image = wrapped_audio_spectrogram_image

# Monkey patch direct sur les fonctions de Gradio
import gradio.processing_utils
original_encode_pil = gradio.processing_utils.encode_pil_to_bytes

# Remplacer la fonction encode_pil_to_bytes pour empêcher l'usage de WEBP
def safe_encode_pil_to_bytes(pil_image, format="JPEG", **params):
    # Toujours utiliser JPEG au lieu de WEBP
    if format.upper() == "WEBP":
        format = "JPEG"
    # Convertir RGBA en RGB si on utilise JPEG
    if format.upper() == "JPEG" and pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    # Appeler la fonction originale avec les modifications
    return original_encode_pil(pil_image, format=format, **params)

# Appliquer le monkey patch
gradio.processing_utils.encode_pil_to_bytes = safe_encode_pil_to_bytes

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..interface.aeiou import audio_spectrogram_image
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

from .interfaces.restoration_sampling_ui import create_restoration_ui

model = None
model_type = None
sample_rate = 32000
sample_size = 1920000

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size, model_type
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        # Load checkpoint with EMA fallback
        state_dict = load_ckpt_state_dict(model_ckpt_path)
        # If checkpoint comes from a Lightning training run, it may contain EMA weights under
        # the prefix 'diffusion_ema.ema_model.'. Remap those keys to match the inference
        # model's expected 'model.' prefix so that EMA parameters are used for sampling.
        if any(k.startswith("diffusion_ema.ema_model.") for k in state_dict.keys()):
            print("Found EMA weights in checkpoint – remapping to model.* keys")
            remapped_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("diffusion_ema.ema_model."):
                    print("Found EMA weights in checkpoint – remapping to model.* keys")
                    # Map 'diffusion_ema.ema_model.xxx' -> 'model.xxx'
                    new_key = k.replace("diffusion_ema.ema_model.", "model.")
                    remapped_state_dict[new_key] = v
                elif k.startswith("diffusion.model."):
                    print("Found EMA weights in checkpoint – remapping to model.* keys")
                    # Map 'diffusion.model.xxx' -> 'model.xxx' (non-EMA online weights fallback)
                    new_key = k.replace("diffusion.", "")
                    remapped_state_dict[new_key] = v
                elif k.startswith("diffusion."):
                    print("Found EMA weights in checkpoint – remapping to model.* keys")
                    # Strip leading 'diffusion.' for other sub-modules (e.g. pretransform)
                    new_key = k[len("diffusion."):]
                    remapped_state_dict[new_key] = v
                else:
                    print("Found EMA weights in checkpoint – remapping to model.* keys")
                    remapped_state_dict[k] = v
            state_dict = remapped_state_dict
        copy_state_dict(model, state_dict)


    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config

def generate_lm(
        temperature=1.0,
        top_p=0.95,
        top_k=0,    
        batch_size=1,
        ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    audio = model.generate_audio(
        batch_size=batch_size,
        max_gen_len = sample_size//model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])

def autoencoder_process(audio, latent_noise, n_quantizers):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    audio = audio.to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_autoencoder_ui(model_config):

    is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

    if is_dac_rvq:
        n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
    else:
        n_quantizers = 0

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        n_quantizers_slider = gr.Slider(minimum=1, maximum=n_quantizers, step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
        latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=autoencoder_process, inputs=[input_audio, latent_noise_slider, n_quantizers_slider], outputs=output_audio, api_name="process")

    return ui

def diffusion_prior_process(audio, steps, sampler_type, sigma_min, sigma_max):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.unsqueeze(0)

    audio = model.stereoize(audio, in_sr, steps, sampler_kwargs={"sampler_type": sampler_type, "sigma_min": sigma_min, "sigma_max": sigma_max})

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_diffusion_prior_ui(model_config):
    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        # Sampler params
        with gr.Row():
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=diffusion_prior_process, inputs=[input_audio, steps_slider, sampler_type_dropdown, sigma_min_slider, sigma_max_slider], outputs=output_audio, api_name="process")    

    return ui

def create_lm_ui(model_config):
    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_lm, 
            inputs=[
                temperature_slider, 
                top_p_slider, 
                top_k_slider
            ], 
            outputs=[output_audio, audio_spectrogram_output],
            api_name="generate"
        )

    return ui

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=device)
    
    # Set global variables from model config
    import stable_audio_tools.interface.interfaces.restoration as restoration
    restoration.sample_size = model_config["sample_size"]
    restoration.sample_rate = model_config["sample_rate"]
    
    if model_type in ["diffusion_cond", "diffusion_uncond", "diffusion_inpaint", "diffusion_uncond_restoration", "diffusion_cond_restoration"] :
        ui = create_restoration_ui(model_config, model, in_model_half=model_half)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(model_config)
    elif model_type == "lm":
        ui = create_lm_ui(model_config)
        
    return ui