import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T
from torch.nn.functional import interpolate
from einops import rearrange

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop

import logging
import gc
import json
import os
import time
import torchaudio

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.DEBUG)


def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None

    # Inpainting mask
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    # Now the generative AI part:

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, device=device)
    elif diff_objective == "rectified_flow":
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, device=device)

    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        clean_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        return_latents = False,
        **sampler_kwargs
        ) -> tp.Tuple[torch.Tensor, tp.Optional[dict]]: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        clean_audio: A tuple of (sample_rate, audio) containing the clean reference audio for metrics.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    LOG.info("Starting audio generation")
    
    # Initialize metrics dictionary
    metrics_dict = {}
    
    # The length of the output in audio samples
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        LOG.info(f"Using latent diffusion, adjusted sample_size to {sample_size}")

    # Seed
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    LOG.info(f"Using seed: {seed}")
    torch.manual_seed(seed)

    # Define the initial noise
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, (
        "Must provide either conditioning or conditioning_tensors"
    )
    if conditioning_tensors is None:
        if isinstance(conditioning, list) and len(conditioning) > 0 and 'degraded_audio' in conditioning[0]:
            degraded_audio = conditioning[0]['degraded_audio']
            
            if hasattr(model, 'input_concat_ids') and 'degraded_audio' in model.input_concat_ids and hasattr(degraded_audio, 'shape'):
                expected_latent_size = sample_size
                
                if degraded_audio.shape[-1] == expected_latent_size:
                    conditioning_tensors = {'degraded_audio': [degraded_audio, torch.ones(degraded_audio.shape[0], degraded_audio.shape[-1]).to(device)]}
                else:
                    conditioning_tensors = model.conditioner(conditioning, device)
            else:
                conditioning_tensors = model.conditioner(conditioning, device)
        else:
            conditioning_tensors = model.conditioner(conditioning, device)

    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    conditioning_inputs["input_concat_cond"] = conditioning_inputs["input_concat_cond"][:, :, :sample_size]

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        in_sr, init_audio = init_audio
        io_channels = model.io_channels

        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
        sampler_kwargs["sigma_max"] = init_noise_level

    # Convert to model dtype
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)

    # Generate audio
    diff_objective = model.diffusion_objective
    LOG.info(f"Using diffusion objective: {diff_objective}")

    if diff_objective == "v":    
        sampled = sample(model.model, noise, steps, 0, **conditioning_inputs, cfg_scale=cfg_scale, dist_shift=model.dist_shift, batch_cfg=True)
    elif diff_objective == "rectified_flow":
        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]
        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # Cleanup
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()

    # Decode latents if needed
    if model.pretransform is not None and not return_latents:
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Calculate metrics if clean audio is provided
    if clean_audio is not None and not return_latents:
        LOG.info("Calculating metrics between generated and clean audio")
    
        from ..training.losses.metrics import (
            LogSpectralDistance,
            LTASDistance,
            SISDRMetric,
            SNRMetric,
            STFTDistance,
            MelDistance
        )
        
        metrics = {
            'lsd': LogSpectralDistance().to(device),
            'ltas': LTASDistance().to(device),
            'sisdr': SISDRMetric().to(device),
            'snr': SNRMetric().to(device),
            'stft': STFTDistance().to(device),
            'mel': MelDistance(sample_rate=model.sample_rate).to(device)
        }
        
        clean_audio_tensor = clean_audio[1].unsqueeze(0).to(device)
        clean_audio_latent = model.pretransform.encode(clean_audio_tensor)
        clean_audio_tensor = model.pretransform.decode(clean_audio_latent)
        clean_audio_tensor = rearrange(clean_audio_tensor, 'b d n -> d (b n)')
        sampled_metrics = rearrange(sampled, 'b d n -> d (b n)')

        # Calculate metrics between generated and clean audio
        for name, metric in metrics.items():
            metrics_dict[f'demo_{name}'] = metric(sampled_metrics, clean_audio_tensor).item()
            LOG.info(f"Metric {name} (generated vs clean): {metrics_dict[f'demo_{name}']}")

        # Calculate metrics between degraded and clean audio if degraded audio is available
        LOG.info("Calculating metrics between degraded and clean audio")
        degraded_audio_tensor = degraded_audio.unsqueeze(0).to(device)
        degraded_audio_latent = model.pretransform.encode(degraded_audio_tensor)
        degraded_audio_tensor = model.pretransform.decode(degraded_audio_latent)
        degraded_audio_tensor = rearrange(degraded_audio_tensor, 'b d n -> d (b n)')
        for name, metric in metrics.items():
            metrics_dict[f'degraded_{name}'] = metric(degraded_audio_tensor, clean_audio_tensor).item()
            LOG.info(f"Metric {name} (degraded vs clean): {metrics_dict[f'degraded_{name}']}")
        
        # Save metrics and audio files
        try:
            # Create date-based directory structure
            current_date = time.strftime("%Y-%m-%d")
            current_time = time.strftime("%H-%M-%S")
            output_dir = os.path.join("stable_audio_tools\output", current_date)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a subdirectory for this specific generation
            generation_dir = os.path.join(output_dir, f"generation_{current_time}")
            os.makedirs(generation_dir, exist_ok=True)
            
            # Add additional metadata
            metrics_dict.update({
                "timestamp": f"{current_date}_{current_time}",
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sample_rate": model.sample_rate,
                "sample_size": sample_size
            })
            
            # Save metrics to JSON file
            metrics_filename = os.path.join(generation_dir, "metrics.json")
            with open(metrics_filename, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            
            LOG.info(f"Metrics saved to {metrics_filename}")

            # Save clean audio¨
            if clean_audio is not None:
                clean_audio_filename = os.path.join(generation_dir, "clean_audio.wav")
                clean_audio_save = clean_audio_tensor.to(torch.float32)
                if torch.max(torch.abs(clean_audio_save)) > 1e-7:
                    clean_audio_save = clean_audio_save.div(torch.max(torch.abs(clean_audio_save)))
                    clean_audio_save = clean_audio_save.mul(32767).to(torch.int16).cpu()
                    torchaudio.save(clean_audio_filename, clean_audio_save, model.sample_rate)
                    LOG.info(f"Clean audio saved to {clean_audio_filename}")

            degraded_audio_filename = os.path.join(generation_dir, "degraded_audio.wav")
            #degraded_audio_tensor = degraded_audio[1]
            degraded_audio_save = degraded_audio_tensor.to(torch.float32)
            if torch.max(torch.abs(degraded_audio_save)) > 1e-7:
                degraded_audio_save = degraded_audio_save.div(torch.max(torch.abs(degraded_audio_save)))
            degraded_audio_save = degraded_audio_save.mul(32767).to(torch.int16).cpu()
            torchaudio.save(degraded_audio_filename, degraded_audio_save, model.sample_rate)
            LOG.info(f"Degraded audio saved to {degraded_audio_filename}")

            # Save generated audio
            generated_audio_filename = os.path.join(generation_dir, "generated_audio.wav")
            generated_audio_save = sampled_metrics.to(torch.float32)
            if torch.max(torch.abs(generated_audio_save)) > 1e-7:
                generated_audio_save = generated_audio_save.div(torch.max(torch.abs(generated_audio_save)))
            generated_audio_save = generated_audio_save.mul(32767).to(torch.int16).cpu()
            torchaudio.save(generated_audio_filename, generated_audio_save, model.sample_rate)
            LOG.info(f"Generated audio saved to {generated_audio_filename}")

        except Exception as e:
            LOG.warning(f"Failed to save metrics or audio files: {e}")

     # Add additional metadata
        metrics_dict.update({
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sample_rate": model.sample_rate,
            "sample_size": sample_size
        })

    # Return audio and metrics
    return sampled, metrics_dict if clean_audio is not None else None

def generate_diffusion_cond_inpaint(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask = None,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion inpainting model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        inpaint_mask: A mask to use for inpainting. Shape should be [batch_size, sample_size]
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)

        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        inpaint_sr, inpaint_audio = inpaint_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)

        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
       
        if inpaint_mask is not None:
            # interpolate inpaint mask to the sample size
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)  
    else:
        mask = inpaint_mask.unsqueeze(1)

    # Inpainting mask
    mask = mask.to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    
    if init_audio is not None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective
    LOG.debug(f"Diffusion objective: {diff_objective}")
    LOG.debug(f"Sampler kwargs: {sampler_kwargs}")

    # Log input_concat_ids to understand what gets concatenated in the model
    if hasattr(model, 'input_concat_ids'):
        LOG.debug(f"Model input_concat_ids: {model.input_concat_ids}")

    # Debug conditioning inputs before sampling
    for k, v in conditioning_inputs.items():
        if hasattr(v, 'shape'):
            LOG.debug(f"Conditioning input '{k}' shape: {v.shape}, dtype: {v.dtype}, min: {v.min().item():.4f}, max: {v.max().item():.4f}, mean: {v.mean().item():.4f}")
        else:
            LOG.debug(f"Conditioning input '{k}' is not a tensor")

    if diff_objective == "v":    
        LOG.debug("Starting k-diffusion sampling process (sample_k)")
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
        LOG.debug(f"sample_k completed - output shape: {sampled.shape}, min: {sampled.min().item():.4f}, max: {sampled.max().item():.4f}, mean: {sampled.mean().item():.4f}")
    elif diff_objective == "rectified_flow":
        LOG.debug("Starting rectified flow sampling process (sample_rf)")

        if "sigma_min" in sampler_kwargs:
            LOG.debug(f"Removing sigma_min={sampler_kwargs['sigma_min']} from sampler_kwargs for RF sampling")
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            LOG.debug(f"Removing rho={sampler_kwargs['rho']} from sampler_kwargs for RF sampling")
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
        LOG.debug(f"sample_rf completed - output shape: {sampled.shape}, min: {sampled.min().item():.4f}, max: {sampled.max().item():.4f}, mean: {sampled.mean().item():.4f}")

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    LOG.debug("Sampling completed, cleaning up")
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        LOG.debug("Decoding latents to audio with pretransform")
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        LOG.debug(f"Latents before decoding - shape: {sampled.shape}, dtype: {sampled.dtype}, min: {sampled.min().item():.4f}, max: {sampled.max().item():.4f}, mean: {sampled.mean().item():.4f}")
        sampled = model.pretransform.decode(sampled)
        LOG.debug(f"Decoded audio - shape: {sampled.shape}, min: {sampled.min().item():.4f}, max: {sampled.max().item():.4f}, mean: {sampled.mean().item():.4f}")

    # Return audio
    LOG.debug(f"Final output - shape: {sampled.shape}, min: {sampled.min().item():.4f}, max: {sampled.max().item():.4f}, mean: {sampled.mean().item():.4f}")
    LOG.debug("========== GENERATE_DIFFUSION_COND COMPLETED ==========")
    return sampled


# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask