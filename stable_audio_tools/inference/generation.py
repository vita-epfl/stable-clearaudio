import numpy as np
import torch 
import typing as tp
import math 
from torch.nn.functional import interpolate
from einops import rearrange

import os
from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf, sample_cold, sample_cold_waveform, sample_rectified_flow, sample_rectified_flow_waveform

import logging
import os
import time
import torchaudio
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

def _calculate_best_step(all_metrics: dict, steps_list: list) -> dict:
    """
    Calculate the best step based on a scoring of various metrics.

    Args:
        all_metrics (dict): A dictionary where keys are metric names and values are lists of metric values over steps.
        steps_list (list): A list of the steps at which metrics were calculated.

    Returns:
        dict: A dictionary containing the best step, its index, and the scores for all steps.
    """
    # All restoration_success metrics are positive (higher is better)
    restoration_metrics = {
        key: 1.0 for key in all_metrics.keys() if key.startswith('restoration_success_')
    }

    # Define raw metrics and their polarity (1 for positive, -1 for negative)
    raw_metrics = {
        'demo_lsd': -1.0,
        'demo_ltas': -1.0,
        'demo_sisdr': 1.0,
        'demo_snr': 1.0,
        'demo_stft': -1.0,
        'demo_mel': -1.0,
        'latent_mse_loss': -1.0,
        'latent_l1_loss': -1.0,
        'waveform_mse_loss': -1.0,
        'waveform_l1_loss': -1.0
    }

    # Combine all metrics to be used for scoring
    scoring_metrics = {**restoration_metrics, **raw_metrics}

    if not scoring_metrics or not steps_list:
        return {}

    num_steps = len(steps_list)
    scores = np.zeros(num_steps)

    for metric_name, polarity in scoring_metrics.items():
        if metric_name not in all_metrics:
            continue

        metric_values = np.array(all_metrics[metric_name])
        
        # Normalize the metric values to a 0-1 range
        min_val = np.min(metric_values)
        max_val = np.max(metric_values)
        
        if max_val - min_val > 1e-9:
            normalized_values = (metric_values - min_val) / (max_val - min_val)
        else:
            normalized_values = np.zeros_like(metric_values)

        # Apply polarity
        if polarity == -1.0:
            # For negative metrics, lower is better, so we invert the score
            scores += (1.0 - normalized_values)
        else:
            scores += normalized_values

    best_step_index = np.argmax(scores)
    best_step = steps_list[best_step_index]

    return {
        'best_step': best_step,
        'best_step_index': int(best_step_index),
        'scores': scores.tolist(),
        'steps_list': steps_list
    }

def _calculate_metrics(
    sampled_waveform,
    sampled_latent,
    clean_audio_tensor,
    clean_audio_latent,
    degraded_audio_tensor,
    degraded_audio_latent,
    model,
    device,
    steps,
    clean_metrics=None
):
    """Calculate metrics between sampled, clean, and degraded audio.
    
    Note: The latent parameters (sampled_latent, clean_audio_latent, degraded_audio_latent)
    are only used when model.pretransform is not None. When model.pretransform is None,
    these parameters can be set to None.
    """
    LOG.debug(f"Calculating metrics for {steps} steps")
    
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
    
    # Handle tensor shape variations for clean audio
    if len(clean_audio_tensor.shape) == 3:  # Shape is [batch, channels, samples]
        clean_audio_tensor_rearranged = rearrange(clean_audio_tensor, 'b d n -> d (b n)')
    elif len(clean_audio_tensor.shape) == 2:  # Shape is [channels, samples]
        clean_audio_tensor_rearranged = clean_audio_tensor
    else:
        raise ValueError(f"Unexpected shape for clean_audio_tensor: {clean_audio_tensor.shape}")
    
    # Handle tensor shape variations for degraded audio
    if len(degraded_audio_tensor.shape) == 3:  # Shape is [batch, channels, samples]
        degraded_audio_tensor_rearranged = rearrange(degraded_audio_tensor, 'b d n -> d (b n)')
    elif len(degraded_audio_tensor.shape) == 2:  # Shape is [channels, samples]
        degraded_audio_tensor_rearranged = degraded_audio_tensor
    else:
        raise ValueError(f"Unexpected shape for degraded_audio_tensor: {degraded_audio_tensor.shape}")
    
    # Handle tensor shape variations for sampled waveform
    if len(sampled_waveform.shape) == 3:  # Shape is [batch, channels, samples]
        sampled_metrics_waveform = rearrange(sampled_waveform, 'b d n -> d (b n)')
    elif len(sampled_waveform.shape) == 2:  # Shape is [channels, samples]
        sampled_metrics_waveform = sampled_waveform
    else:
        raise ValueError(f"Unexpected shape for sampled_waveform: {sampled_waveform.shape}")

    # Final check to ensure audio lengths match before metric calculation
    min_len = min(sampled_metrics_waveform.shape[-1], clean_audio_tensor_rearranged.shape[-1], degraded_audio_tensor_rearranged.shape[-1])
    sampled_metrics_waveform = sampled_metrics_waveform[..., :min_len]
    clean_audio_tensor_rearranged = clean_audio_tensor_rearranged[..., :min_len]
    degraded_audio_tensor_rearranged = degraded_audio_tensor_rearranged[..., :min_len]

    # Calculate metrics between generated and clean audio
    for name, metric in metrics.items():
        # Calculate demo metric (generated vs clean)
        demo_metric = metric(sampled_metrics_waveform, clean_audio_tensor_rearranged).item()
        metrics_dict[f'demo_{name}'] = demo_metric
        
        # Calculate degraded metric (degraded vs clean)
        degraded_metric = metric(degraded_audio_tensor_rearranged, clean_audio_tensor_rearranged).item()
        # Store degraded metrics explicitly so they can be saved separately
        metrics_dict[f'degraded_{name}'] = degraded_metric
        
        # Use pre-calculated clean metrics if provided, otherwise calculate them
        if clean_metrics and f'clean_{name}' in clean_metrics:
            clean_metric = clean_metrics[f'clean_{name}']
            # Store clean metrics for graph display
            metrics_dict[f'clean_{name}'] = clean_metric
        else:
            # Fall back to calculating clean metrics if not provided
            # This code path shouldn't be used if clean_metrics are passed correctly
            LOG.debug("Calculating clean metrics - this should only happen at step 0")
            
            if model.pretransform is not None:
                encoded_clean = model.pretransform.encode(clean_audio_tensor)
                decoded_clean = model.pretransform.decode(encoded_clean).squeeze()
                
                # Handle different tensor shapes depending on batch dimension
                if len(decoded_clean.shape) == 3:  # Shape is [batch, channels, samples]
                    # Then rearrange and truncate to match dimensions
                    decoded_clean_rearranged = rearrange(decoded_clean, 'b d n -> d (b n)')
                elif len(decoded_clean.shape) == 2:  # Shape is [channels, samples]
                    # Already in the right format, no need to rearrange
                    decoded_clean_rearranged = decoded_clean
                else:
                    raise ValueError(f"Unexpected shape for decoded_clean: {decoded_clean.shape}")
                    
                # Make sure sizes match before computing metric
                min_len_clean = min(clean_audio_tensor_rearranged.shape[-1], decoded_clean_rearranged.shape[-1])
                clean_metric = metric(clean_audio_tensor_rearranged[..., :min_len_clean], 
                                    decoded_clean_rearranged[..., :min_len_clean]).item()
            else:
                # If there's no pretransform, use the clean audio directly
                clean_metric = metric(clean_audio_tensor_rearranged, clean_audio_tensor_rearranged).item()
            # Store clean metrics explicitly so they can be added to graphs
            metrics_dict[f'clean_{name}'] = clean_metric
        
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

def generate_diffusion_uncond_restoration(
        model,
        model_type,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 44100,
        device: str = "cuda",
        clean_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        degraded_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        effects_list: tp.Optional[list] = None,
        callback = None,
        t_start: float = 1.0,
        schedule: str = "linear",
        output_dir = None,
        metrics_every: int = 0,
        use_ema: bool = True,
        **sampler_kwargs
) -> tp.Tuple[torch.Tensor, tp.Optional[dict]]:
    """
    Generate audio from a prompt using a diffusion model.

    Args:
        model: The diffusion model to use for generation. Must have `degradation_ops` attribute.
        steps (int): The number of sampling steps.
        batch_size (int): The number of samples to generate.
        sample_size (int): The length of the generated audio in samples.
        sample_rate (int): The sample rate of the generated audio.
        device (str): The device to use for generation.
        degraded_audio (tp.Optional[tp.Tuple[int, torch.Tensor]]): A tuple of (sample_rate, audio_tensor)
            for the audio to be restored. If None, generation will start from noise.
        t_start (float): The starting time for the reverse process. Should correspond to the degradation
            level of the input audio. Defaults to 1.0 (fully degraded).
        schedule (str): Time schedule type passed to `sample_cold` ("linear", "sqrt", "cosine").
        output_dir (str): The directory to save the generated audio and metrics to.
        metrics_every (int): The number of steps between each metric calculation. If 0, metrics are only calculated at the end.
        **sampler_kwargs: Additional arguments for the sampler.
    """
    LOG.info("Starting audio generation with cold diffusion unconditional restoration")

    # Set model to eval mode for consistent results
    model.eval()
    
    # Use EMA model if available and requested
    inference_model = model
    if use_ema and hasattr(model, 'diffusion_ema'):
        inference_model = model.diffusion_ema
        LOG.debug("Using EMA model for inference")
    elif use_ema and hasattr(model, 'model_ema'):
        inference_model = model.model_ema
        LOG.debug("Using EMA model for inference")
    else:
        inference_model = model.model
        LOG.debug("Using base model for inference")

    # The length of the output in audio samples
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        LOG.debug(f"Using latent diffusion, adjusted sample_size to {sample_size}")

    # Set up torch backend for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    
    # Prepare initial tensor x
    if degraded_audio is not None:
        LOG.debug("Using provided degraded audio as input.")
        in_sr, x_audio = degraded_audio
        x_audio = prepare_audio(
            x_audio,
            in_sr=in_sr,
            target_sr=sample_rate,
            target_length=audio_sample_size,
            target_channels=model.io_channels,
            device=device
        )
        if x_audio.shape[0] < batch_size:
            x_audio = x_audio.repeat(batch_size, 1, 1)
    else:
        raise ValueError("No degraded audio provided, generation from noise is not supported for cold diffusion.")

    # Encode to latent space if needed
    if model.pretransform is not None:
        x = model.pretransform.encode(x_audio)
    else:
        # If no pretransform, use the audio directly
        x = x_audio

    # Convert to model dtype
    model_dtype = next(model.model.parameters()).dtype
    x = x.to(model_dtype)

    # Prepare for metrics calculation
    all_metrics = {}
    callback = None
    clean_audio_latent = None
    degraded_audio_latent = None

    if clean_audio is not None:
        LOG.debug("Preparing audio for metrics calculation.")
        in_sr, clean_audio_tensor = clean_audio
        clean_audio_tensor = prepare_audio(
            clean_audio_tensor,
            in_sr=in_sr,
            target_sr=sample_rate,
            target_length=audio_sample_size,
            target_channels=model.io_channels,
            device=device
        )

        degraded_audio_tensor = x_audio.clone() # x_audio is the prepared degraded audio

        # Initialize latents based on whether pretransform exists
        if model.pretransform is not None:
            clean_audio_latent = model.pretransform.encode(clean_audio_tensor)
            degraded_audio_latent = model.pretransform.encode(degraded_audio_tensor)

        LOG.debug(f"Calculating metrics at step 0/{steps} (clean audio)")
        metrics_at_step_0 = _calculate_metrics(
            clean_audio_tensor,
            clean_audio_latent,
            clean_audio_tensor,
            clean_audio_latent,
            degraded_audio_tensor,
            degraded_audio_latent,
            model,
            device,
            0
        )
        all_metrics["step_0"] = metrics_at_step_0

        # Calculate clean metrics once here - these won't change during the restoration process
        clean_metrics = {}
        
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
        
        # Handle different tensor shapes depending on batch dimension
        if len(clean_audio_tensor.shape) == 3:  # Shape is [batch, channels, samples]
            clean_audio_tensor_rearranged = rearrange(clean_audio_tensor, 'b d n -> d (b n)')
        elif len(clean_audio_tensor.shape) == 2:  # Shape is [channels, samples]
            clean_audio_tensor_rearranged = clean_audio_tensor
        else:
            raise ValueError(f"Unexpected shape for clean_audio_tensor: {clean_audio_tensor.shape}")
        
        if model.pretransform is not None:
            # First encode and decode clean audio through the pretransform to get clean metrics
            encoded_clean = model.pretransform.encode(clean_audio_tensor)
            decoded_clean = model.pretransform.decode(encoded_clean).squeeze()
            
            if len(decoded_clean.shape) == 3:  # Shape is [batch, channels, samples]
                decoded_clean_rearranged = rearrange(decoded_clean, 'b d n -> d (b n)')
            elif len(decoded_clean.shape) == 2:  # Shape is [channels, samples]
                decoded_clean_rearranged = decoded_clean
            else:
                raise ValueError(f"Unexpected shape for decoded_clean: {decoded_clean.shape}")
                
            # Make sure sizes match before computing metric
            min_len_clean = min(clean_audio_tensor_rearranged.shape[-1], decoded_clean_rearranged.shape[-1])
            
            # Calculate clean metrics (clean vs clean) once
            for name, metric in metrics.items():
                clean_metric = metric(clean_audio_tensor_rearranged[..., :min_len_clean], 
                                      decoded_clean_rearranged[..., :min_len_clean]).item()
                clean_metrics[f'clean_{name}'] = clean_metric
        else:
            # When there's no pretransform, use the clean audio directly for metrics
            for name, metric in metrics.items():
                clean_metric = metric(clean_audio_tensor_rearranged, clean_audio_tensor_rearranged).item()
                clean_metrics[f'clean_{name}'] = clean_metric
            
        if metrics_every > 0:
            def metrics_callback(callback_args):
                i = callback_args['i']
                x_0_hat_latent = callback_args['denoised']
                if (i + 1) % metrics_every == 0 or i == steps - 1:
                    LOG.debug(f"Calculating metrics at step {i+1}/{steps}")
                    metrics_at_step = _calculate_metrics(
                        model.pretransform.decode(x_0_hat_latent) if model.pretransform else x_0_hat_latent,
                        x_0_hat_latent if model.pretransform else None,
                        clean_audio_tensor,
                        clean_audio_latent,
                        degraded_audio_tensor,
                        degraded_audio_latent,
                        model,
                        device,
                        i + 1,
                        clean_metrics=clean_metrics  # Pass pre-calculated clean metrics
                    )
                    for k, v in metrics_at_step.items():
                        if k not in all_metrics:
                            all_metrics[k] = []
                        # Don't append clean metrics to the per-step arrays
                        if not k.startswith('clean_'):
                            all_metrics[k].append(v)
            callback = metrics_callback
    else:
        # No clean audio provided, no metrics callback needed
        callback = None

    # Use autocast for mixed precision consistency
    with torch.amp.autocast("cuda"):
        if model_type == "rectified_flow_uncond_restoration":
            # Generate audio using rectified flow sampling
            LOG.debug(f"Starting rectified flow sampling for {steps} steps from t_start={t_start}.")

            if model.pretransform is not None:
            
                fake_latent = sample_rectified_flow(
                    inference_model,
                    x,
                    steps,
                    t_start=t_start,
                    callback=callback,
                    **sampler_kwargs
                )
                LOG.debug("Decoding final latents to audio.")
                fakes = model.pretransform.decode(fake_latent)
            else:

                fakes = sample_rectified_flow_waveform(
                    inference_model,
                    x,
                    steps,
                    t_start=t_start,
                    callback=callback,
                    **sampler_kwargs
                )
        elif model_type == "cold_diffusion_uncond_restoration":
            # Generate audio using sample_cold
            LOG.debug(f"Starting cold sampling for {steps} steps from t_start={t_start}.")

            if model.pretransform is not None:
            
                fake_latent = sample_cold(
                    inference_model,
                    x,
                    steps,
                    t_start=t_start,
                    schedule=schedule,
                    callback=callback,
                    **sampler_kwargs
                )
                LOG.debug("Decoding final latents to audio.")
                fakes = model.pretransform.decode(fake_latent)
            else:

                fakes = sample_cold_waveform(
                    inference_model,
                    x,
                    steps,
                    t_start=t_start,
                    schedule=schedule,
                    callback=callback,
                    **sampler_kwargs
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Calculate final metrics if not already done by callback
    if clean_audio is not None and (metrics_every == 0 or steps % metrics_every != 0):
        # Only store the final metrics
        metrics_at_step = _calculate_metrics(
            fakes,
            fake_latent if model.pretransform else None,
            clean_audio_tensor,
            clean_audio_latent,
            degraded_audio_tensor,
            degraded_audio_latent,
            model,
            device,
            steps,
            clean_metrics=clean_metrics  # Use pre-calculated clean metrics
        )
        for k, v in metrics_at_step.items():
            if k not in all_metrics:
                all_metrics[k] = []
            # Don't append clean metrics to the per-step arrays
            if not k.startswith('clean_'):
                all_metrics[k].append(v)

    if output_dir and clean_audio is not None:
        try:
            # Save clean audio
            clean_audio_filename = os.path.join(output_dir, "clean_audio.wav")
            clean_audio_save = rearrange(clean_audio_tensor, 'b d n -> d (b n)').to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(clean_audio_filename, clean_audio_save, sample_rate)
            LOG.debug(f"Clean audio saved to {clean_audio_filename}")

            # Save degraded audio
            degraded_audio_filename = os.path.join(output_dir, "degraded_audio.wav")
            degraded_audio_save = rearrange(degraded_audio_tensor, 'b d n -> d (b n)').to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(degraded_audio_filename, degraded_audio_save, sample_rate)
            LOG.debug(f"Degraded audio saved to {degraded_audio_filename}")
        except Exception as e:
            LOG.warning(f"Failed to save audio files: {e}")


    # Log the keys and types in all_metrics before returning
    LOG.debug(f"Final metrics keys: {list(all_metrics.keys())}")
    for k, v in all_metrics.items():
        if k.startswith('clean_'):
            LOG.debug(f"Clean metric {k} type: {type(v)}, value: {v}")
            
    # Copy clean metrics from clean_metrics dictionary to all_metrics dictionary
    for k, v in clean_metrics.items():
        all_metrics[k] = v

    # After all metrics are collected, calculate the best step
    if clean_audio is not None and metrics_every > 0:
        steps_list = sorted(list(set(all_metrics.get('steps', []))))
        if steps_list:
            best_step_info = _calculate_best_step(all_metrics, steps_list)
            all_metrics['best_step_recommendation'] = best_step_info
    
    # Add clean metrics to all_metrics as scalar values (not lists) for plotting as reference lines
    if clean_metrics:
        for clean_key, clean_value in clean_metrics.items():
            all_metrics[clean_key] = clean_value
            LOG.debug(f"Added clean metric {clean_key} = {clean_value}")

    return fakes, all_metrics

def generate_diffusion_cond_restoration(
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
        LOG.debug(f"Using latent diffusion, adjusted sample_size to {sample_size}")

    # Seed
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    LOG.info(f"Using seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    LOG.debug(f"Using diffusion objective: {diff_objective}")

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
                LOG.debug(f"[{i+1}/{len(step_milestones)}] Running diffusion for {steps_i} steps to calculate metrics")
                
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
    if model.pretransform is not None:
        sampled_latent = sampled.clone().detach()
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Calculate metrics if clean audio is provided
    if clean_audio is not None:
        LOG.debug("Calculating metrics between generated and clean audio")

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
                    LOG.debug(f"Clean audio saved to {clean_audio_filename}")

                # Save degraded audio
                degraded_audio_filename = os.path.join(output_dir, "degraded_audio.wav")
                degraded_audio_save = rearrange(degraded_audio_tensor, 'b d n -> d (b n)').to(torch.float32)
                if torch.max(torch.abs(degraded_audio_save)) > 1e-7:
                    degraded_audio_save = degraded_audio_save.div(torch.max(torch.abs(degraded_audio_save)))
                degraded_audio_save = degraded_audio_save.mul(32767).to(torch.int16).cpu()
                torchaudio.save(degraded_audio_filename, degraded_audio_save, model.sample_rate)
                LOG.debug(f"Degraded audio saved to {degraded_audio_filename}")
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
    LOG.debug("Sampling completed, cleaning up")
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