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
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.DEBUG)

def _calculate_metrics(
    sampled_waveform,
    sampled_latent,
    clean_audio_tensor,
    clean_audio_latent,
    degraded_audio_tensor,
    degraded_audio_latent,
    model,
    device,
    steps
):
    LOG.info(f"Calculating metrics for {steps} steps")
    
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

    metrics_dict = {}
    
    clean_audio_tensor_rearranged = rearrange(clean_audio_tensor, 'b d n -> d (b n)')
    degraded_audio_tensor_rearranged = rearrange(degraded_audio_tensor, 'b d n -> d (b n)')
    
    sampled_metrics_waveform = rearrange(sampled_waveform, 'b d n -> d (b n)')

    # Calculate metrics between generated and clean audio
    for name, metric in metrics.items():
        # Calculate demo metric (generated vs clean)
        demo_metric = metric(sampled_metrics_waveform, clean_audio_tensor_rearranged).item()
        metrics_dict[f'demo_{name}'] = demo_metric
        
        # Calculate degraded metric (degraded vs clean)
        degraded_metric = metric(degraded_audio_tensor_rearranged, clean_audio_tensor_rearranged).item()
        # Store degraded metrics explicitly so they can be saved separately
        metrics_dict[f'degraded_{name}'] = degraded_metric
        
        # Calculate clean metric (clean vs clean)
        clean_metric = metric(clean_audio_tensor_rearranged, model.pretransform.decode(model.pretransform.encode(clean_audio_tensor)).squeeze()).item()
        
        # Calculate restoration success metric
        numerator = degraded_metric - demo_metric
        denominator = degraded_metric - clean_metric
        restoration_success = numerator / denominator if abs(denominator) > 1e-9 else 0.0
        metrics_dict[f'restoration_success_{name}'] = restoration_success

    # Latent space losses
    if sampled_latent is not None and clean_audio_latent is not None and degraded_audio_latent is not None:
        # MSE Loss
        demo_mse = F.mse_loss(sampled_latent, clean_audio_latent)
        degraded_mse = F.mse_loss(degraded_audio_latent, clean_audio_latent)
        clean_mse = F.mse_loss(clean_audio_latent, clean_audio_latent)
        metrics_dict['latent_mse_loss'] = demo_mse.item()
        metrics_dict['degraded_latent_mse_loss'] = degraded_mse.item()
        
        # Calculate restoration success for latent MSE
        numerator = degraded_mse.item() - demo_mse.item()
        denominator = degraded_mse.item() - clean_mse.item()
        restoration_success_mse = numerator / denominator if abs(denominator) > 1e-9 else 0.0
        metrics_dict['restoration_success_latent_mse_loss'] = restoration_success_mse

        # L1 Loss
        demo_l1 = F.l1_loss(sampled_latent, clean_audio_latent)
        degraded_l1 = F.l1_loss(degraded_audio_latent, clean_audio_latent)
        clean_l1 = F.l1_loss(clean_audio_latent, clean_audio_latent)
        metrics_dict['latent_l1_loss'] = demo_l1.item()
        metrics_dict['degraded_latent_l1_loss'] = degraded_l1.item()
        
        # Calculate restoration success for latent L1
        numerator = degraded_l1.item() - demo_l1.item()
        denominator = degraded_l1.item() - clean_l1.item()
        restoration_success_l1 = numerator / denominator if abs(denominator) > 1e-9 else 0.0
        metrics_dict['restoration_success_latent_l1_loss'] = restoration_success_l1

    # Waveform domain losses
    # MSE Loss
    demo_waveform_mse = F.mse_loss(sampled_metrics_waveform, clean_audio_tensor_rearranged)
    degraded_waveform_mse = F.mse_loss(degraded_audio_tensor_rearranged, clean_audio_tensor_rearranged)
    clean_waveform_mse = F.mse_loss(clean_audio_tensor_rearranged, clean_audio_tensor_rearranged)
    metrics_dict['waveform_mse_loss'] = demo_waveform_mse.item()
    metrics_dict['degraded_waveform_mse_loss'] = degraded_waveform_mse.item()
    
    # Calculate restoration success for waveform MSE
    numerator = degraded_waveform_mse.item() - demo_waveform_mse.item()
    denominator = degraded_waveform_mse.item() - clean_waveform_mse.item()
    restoration_success_waveform_mse = numerator / denominator if abs(denominator) > 1e-9 else 0.0
    metrics_dict['restoration_success_waveform_mse_loss'] = restoration_success_waveform_mse

    # L1 Loss
    demo_waveform_l1 = F.l1_loss(sampled_metrics_waveform, clean_audio_tensor_rearranged)
    degraded_waveform_l1 = F.l1_loss(degraded_audio_tensor_rearranged, clean_audio_tensor_rearranged)
    clean_waveform_l1 = F.l1_loss(clean_audio_tensor_rearranged, clean_audio_tensor_rearranged)
    metrics_dict['waveform_l1_loss'] = demo_waveform_l1.item()
    metrics_dict['degraded_waveform_l1_loss'] = degraded_waveform_l1.item()
    
    # Calculate restoration success for waveform L1
    numerator = degraded_waveform_l1.item() - demo_waveform_l1.item()
    denominator = degraded_waveform_l1.item() - clean_waveform_l1.item()
    restoration_success_waveform_l1 = numerator / denominator if abs(denominator) > 1e-9 else 0.0
    metrics_dict['restoration_success_waveform_l1_loss'] = restoration_success_waveform_l1

    # Add additional metadata
    metrics_dict.update({
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "steps": steps,
        "sample_rate": model.sample_rate,
        "sample_size": sampled_waveform.shape[-1]
    })
    
    return metrics_dict

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
        init_noise_level: float = 1.0,
        clean_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        return_latents = False,
        callback = None,
        output_dir = None,
        metrics_every: int = 0,
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
        init_noise_level: The noise level to use when generating from an initial audio sample.
        clean_audio: A tuple of (sample_rate, audio) containing the clean reference audio for metrics.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        callback: A callback function to call at each step of the diffusion process.
        output_dir: The directory to save the generated audio and metrics to.
        metrics_every: The number of steps between each metric calculation. If 0, metrics are only calculated at the end.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    LOG.info("Starting audio generation")
    
    # Initialize metrics dictionary
    metrics_dict = {}
    init_audio = None
    
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
            
            # Use degraded audio as init_audio for restoration
            if model.pretransform is not None:
                init_audio = model.pretransform.encode(degraded_audio.unsqueeze(0).to(device))
            else:
                init_audio = degraded_audio.unsqueeze(0).to(device)

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
        # For restoration, init_audio is already prepared
        if 'degraded_audio' not in conditioning[0]:
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

    # Optionally run metrics loop
    if clean_audio is not None and metrics_every > 0:

        # Check if we can actually calculate restoration metrics
        can_calculate_metrics = isinstance(conditioning, list) and len(conditioning) > 0 and 'degraded_audio' in conditioning[0]

        if not can_calculate_metrics:
            LOG.warning("Cannot calculate restoration metrics without degraded_audio in conditioning. Skipping metrics loop.")
        
        else:
            all_metrics = {}
            step_milestones = sorted(list(set(list(range(metrics_every, steps, metrics_every)) + [steps])))
            
            # Prepare clean and degraded tensors once
            degraded_audio_tensor_pre = conditioning[0]['degraded_audio']
            clean_audio_tensor = clean_audio[1].unsqueeze(0).to(device)
            degraded_audio_tensor = degraded_audio_tensor_pre.unsqueeze(0).to(device)

            if model.pretransform:
                clean_audio_latent = model.pretransform.encode(clean_audio_tensor)
                clean_audio_tensor = model.pretransform.decode(clean_audio_latent)
                degraded_audio_latent = model.pretransform.encode(degraded_audio_tensor)
                degraded_audio_tensor = model.pretransform.decode(degraded_audio_latent)
            else:
                clean_audio_latent = None
                degraded_audio_latent = None

            for i, steps_i in enumerate(step_milestones):
                LOG.info(f"[{i+1}/{len(step_milestones)}] Running diffusion for {steps_i} steps to calculate metrics")
                
                # Run sampler
                if diff_objective == "v":
                    all_args = {**sampler_kwargs, **conditioning_inputs, "cfg_scale": cfg_scale, "batch_cfg": True, "rescale_cfg": True}
                    sampled_latent = sample_k(model.model, noise, init_data=init_audio, steps=steps_i, device=device, callback=callback, **all_args)
                elif diff_objective == "rectified_flow":
                    if "sigma_min" in sampler_kwargs: del sampler_kwargs["sigma_min"]
                    if "rho" in sampler_kwargs: del sampler_kwargs["rho"]
                    all_args = {**sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, "cfg_scale": cfg_scale, "batch_cfg": True, "rescale_cfg": True}
                    sampled_latent = sample_rf(model.model, noise, init_data=init_audio, steps=steps_i, device=device, callback=callback, **all_args)
                
                # Decode if needed
                if model.pretransform is not None:
                    sampled_waveform = sampled_latent.to(next(model.pretransform.parameters()).dtype)
                    sampled_waveform = model.pretransform.decode(sampled_waveform)
                else:
                    sampled_waveform = sampled_latent
                
                # Calculate metrics for this step
                metrics_at_step = _calculate_metrics(
                    sampled_waveform,
                    sampled_latent if model.pretransform else None,
                    clean_audio_tensor,
                    clean_audio_latent,
                    degraded_audio_tensor,
                    degraded_audio_latent,
                    model,
                    device,
                    steps_i
                )
                
                for k, v in metrics_at_step.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
            
            # Final sampled audio is from the last iteration
            sampled = sampled_waveform
            if return_latents:
                sampled = sampled_latent

            # Cleanup and return
            del noise
            del conditioning_tensors
            del conditioning_inputs
            torch.cuda.empty_cache()

            return sampled, all_metrics
        
    # Generate audio without metrics loop
    LOG.info(f"Running diffusion for {steps} steps")
    if diff_objective == "v":    
        all_args = {**sampler_kwargs, **conditioning_inputs, "cfg_scale": cfg_scale, "batch_cfg": True, "rescale_cfg": True}
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, device=device, callback=callback, **all_args)
    elif diff_objective == "rectified_flow":
        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]
        all_args = {**sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, "cfg_scale": cfg_scale, "batch_cfg": True, "rescale_cfg": True}
        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, device=device, callback=callback, **all_args)

    # Cleanup
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()

    # Decode latents if needed
    if model.pretransform is not None and not return_latents:
        sampled_latent = sampled.clone().detach()
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Calculate metrics if clean audio is provided
    if clean_audio is not None and not return_latents:
        LOG.info("Calculating metrics between generated and clean audio")

        # Check if we can actually calculate restoration metrics
        can_calculate_metrics = isinstance(conditioning, list) and len(conditioning) > 0 and 'degraded_audio' in conditioning[0]
        
        if not can_calculate_metrics:
            LOG.warning("Cannot calculate restoration metrics without degraded_audio in conditioning. Skipping metric calculation.")
            return sampled, None

        degraded_audio_tensor_pre = conditioning[0]['degraded_audio']
        clean_audio_tensor = clean_audio[1].unsqueeze(0).to(device)
        degraded_audio_tensor = degraded_audio_tensor_pre.unsqueeze(0).to(device)

        if model.pretransform:
            clean_audio_latent = model.pretransform.encode(clean_audio_tensor)
            clean_audio_tensor = model.pretransform.decode(clean_audio_latent)
            degraded_audio_latent = model.pretransform.encode(degraded_audio_tensor)
            degraded_audio_tensor = model.pretransform.decode(degraded_audio_latent)
        else:
            clean_audio_latent = None
            degraded_audio_latent = None
        
        metrics_dict = _calculate_metrics(
            sampled,
            sampled_latent if model.pretransform else None,
            clean_audio_tensor,
            clean_audio_latent,
            degraded_audio_tensor,
            degraded_audio_latent,
            model,
            device,
            steps
        )

        if output_dir:
            try:
                 # Save clean audio
                if clean_audio is not None:
                    clean_audio_filename = os.path.join(output_dir, "clean_audio.wav")
                    clean_audio_save = rearrange(clean_audio_tensor, 'b d n -> d (b n)').to(torch.float32)
                    if torch.max(torch.abs(clean_audio_save)) > 1e-7:
                        clean_audio_save = clean_audio_save.div(torch.max(torch.abs(clean_audio_save)))
                    clean_audio_save = clean_audio_save.mul(32767).to(torch.int16).cpu()
                    torchaudio.save(clean_audio_filename, clean_audio_save, model.sample_rate)
                    LOG.info(f"Clean audio saved to {clean_audio_filename}")

                # Save degraded audio
                degraded_audio_filename = os.path.join(output_dir, "degraded_audio.wav")
                degraded_audio_save = rearrange(degraded_audio_tensor, 'b d n -> d (b n)').to(torch.float32)
                if torch.max(torch.abs(degraded_audio_save)) > 1e-7:
                    degraded_audio_save = degraded_audio_save.div(torch.max(torch.abs(degraded_audio_save)))
                degraded_audio_save = degraded_audio_save.mul(32767).to(torch.int16).cpu()
                torchaudio.save(degraded_audio_filename, degraded_audio_save, model.sample_rate)
                LOG.info(f"Degraded audio saved to {degraded_audio_filename}")
            except Exception as e:
                LOG.warning(f"Failed to save audio files: {e}")

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



    if diff_objective == "v":    
        # k-diffusion denoising process go!
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective == "rectified_flow":

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    LOG.info("Sampling completed, cleaning up")
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

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