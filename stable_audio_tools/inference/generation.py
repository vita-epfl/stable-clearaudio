import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T
from torch.nn.functional import interpolate

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop

import logging
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
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
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
        degraded_audio_path: Path to a degraded audio file to use as the initial input for restoration.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    LOG.debug("========== GENERATE_DIFFUSION_COND CALLED ==========")
    LOG.debug(f"Parameters: steps={steps}, cfg_scale={cfg_scale}, batch_size={batch_size}, sample_size={sample_size}")
    LOG.debug(f"init_audio provided: {init_audio is not None}, init_noise_level: {init_noise_level}")
    LOG.debug(f"conditioning provided: {conditioning is not None}, conditioning_tensors provided: {conditioning_tensors is not None}")
    LOG.debug(f"negative_conditioning provided: {negative_conditioning is not None}, negative_conditioning_tensors provided: {negative_conditioning_tensors is not None}")
    
    # Vérifier si le modèle a des attributs input_concat_ids et diffusion
    LOG.debug(f"Model has attribute input_concat_ids: {hasattr(model, 'input_concat_ids')}")
    if hasattr(model, 'input_concat_ids'):
        LOG.debug(f"Model input_concat_ids: {model.input_concat_ids}")
    LOG.debug(f"Model has attribute diffusion: {hasattr(model, 'diffusion')}")
    if hasattr(model, 'diffusion'):
        LOG.debug(f"Model diffusion has input_concat_ids: {hasattr(model.diffusion, 'input_concat_ids')}")
        if hasattr(model.diffusion, 'input_concat_ids'):
            LOG.debug(f"Model diffusion input_concat_ids: {model.diffusion.input_concat_ids}")
    
    if conditioning is not None:
        LOG.debug(f"Conditioning keys: {list(conditioning[0].keys()) if isinstance(conditioning, list) else list(conditioning.keys())}")
        if isinstance(conditioning, list) and 'degraded_audio' in conditioning[0]:
            degraded = conditioning[0]['degraded_audio']
            LOG.debug(f"Degraded audio in conditioning - shape: {degraded.shape if hasattr(degraded, 'shape') else 'unknown'}, type: {type(degraded)}, dtype: {degraded.dtype if hasattr(degraded, 'dtype') else 'unknown'}")
            if hasattr(degraded, 'shape'):
                LOG.debug(f"Degraded audio stats - min: {degraded.min().item() if degraded.numel() > 0 else 'N/A'}, max: {degraded.max().item() if degraded.numel() > 0 else 'N/A'}, mean: {degraded.mean().item() if degraded.numel() > 0 else 'N/A'}, std: {degraded.std().item() if degraded.numel() > 0 else 'N/A'}")
            
            # S'assurer que l'audio dégradé est bien un tensor et non un tuple
            if isinstance(degraded, tuple) and len(degraded) == 2:
                LOG.debug(f"Degraded audio is a tuple, extracting tensor part")
                # Extraire le tensor de l'audio du tuple (sample_rate, audio_tensor)
                conditioning[0]['degraded_audio'] = degraded[1]

    # The length of the output in audio samples
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        LOG.debug(f"Using latent diffusion, adjusted sample_size to {sample_size} (downsampling ratio: {model.pretransform.downsampling_ratio})")

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    LOG.debug(f"Using seed: {seed}")
    print(seed)
    torch.manual_seed(seed)

    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
    LOG.debug(f"Generated initial noise - shape: {noise.shape}, device: {noise.device}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, (
        "Must provide either conditioning or conditioning_tensors"
    )
    if conditioning_tensors is None:
        LOG.debug("Conditioning tensors not provided, computing them now")
        
        # Vérifier si nous avons déjà des latents encodés pour degraded_audio
        if isinstance(conditioning, list) and len(conditioning) > 0 and 'degraded_audio' in conditioning[0]:
            degraded_audio = conditioning[0]['degraded_audio']
            
            # Si l'audio dégradé a déjà la taille attendue des latents, c'est probablement déjà encodé
            if hasattr(model, 'input_concat_ids') and 'degraded_audio' in model.input_concat_ids and hasattr(degraded_audio, 'shape'):
                expected_latent_size = sample_size
                LOG.debug(f"Checking if degraded_audio is already encoded: shape={degraded_audio.shape[-1]}, expected={expected_latent_size}")
                
                if degraded_audio.shape[-1] == expected_latent_size:
                    LOG.debug("Degraded audio appears to be already encoded, using directly")
                    # Créer directement le tensor de conditionnement
                    conditioning_tensors = {'degraded_audio': [degraded_audio, torch.ones(degraded_audio.shape[0], degraded_audio.shape[-1]).to(device)]}
                else:
                    LOG.debug("Degraded audio needs encoding by conditioner")
                    conditioning_tensors = model.conditioner(conditioning, device)
            else:
                conditioning_tensors = model.conditioner(conditioning, device)
        else:
            conditioning_tensors = model.conditioner(conditioning, device)
            
        LOG.debug(f"Computed conditioning_tensors - keys: {list(conditioning_tensors.keys())}, sizes: {[(k, [t.shape for t in v] if isinstance(v, list) else v.shape) for k, v in conditioning_tensors.items()]}")


    LOG.debug("Getting conditioning inputs from tensors")
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    conditioning_inputs["input_concat_cond"] = conditioning_inputs["input_concat_cond"][:, :, :sample_size]
    LOG.debug(f"Conditioning inputs - keys: {list(conditioning_inputs.keys())}, sizes: {[(k, v.shape if hasattr(v, 'shape') else 'N/A') for k, v in conditioning_inputs.items()]}")

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        LOG.debug("Processing negative conditioning")
        if negative_conditioning_tensors is None:
            LOG.debug("Computing negative conditioning tensors")
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            LOG.debug(f"Computed negative_conditioning_tensors - keys: {list(negative_conditioning_tensors.keys())}")
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
        LOG.debug(f"Negative conditioning inputs - keys: {list(negative_conditioning_tensors.keys())}")
    else:
        LOG.debug("No negative conditioning provided")
        negative_conditioning_tensors = {}

    if init_audio is not None:
        LOG.debug("Processing init_audio")
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio
        LOG.debug(f"Initial audio - SR: {in_sr}, shape: {init_audio.shape if hasattr(init_audio, 'shape') else 'unknown'}, type: {type(init_audio)}, dtype: {init_audio.dtype if hasattr(init_audio, 'dtype') else 'unknown'}")

        io_channels = model.io_channels
        LOG.debug(f"Model io_channels: {io_channels}")

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels
            LOG.debug(f"Using pretransform io_channels: {io_channels}")

        # Prepare the initial audio for use by the model
        LOG.debug("Preparing initial audio")
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        LOG.debug(f"Prepared init_audio - shape: {init_audio.shape}, min: {init_audio.min().item():.4f}, max: {init_audio.max().item():.4f}, mean: {init_audio.mean().item():.4f}")

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            LOG.debug("Encoding initial audio with pretransform")
            init_audio = model.pretransform.encode(init_audio)
            LOG.debug(f"Encoded init_audio - shape: {init_audio.shape}, min: {init_audio.min().item():.4f}, max: {init_audio.max().item():.4f}, mean: {init_audio.mean().item():.4f}")

        init_audio = init_audio.repeat(batch_size, 1, 1)
        LOG.debug(f"Final init_audio after batch repeat - shape: {init_audio.shape}")

        sampler_kwargs["sigma_max"] = init_noise_level
        LOG.debug(f"Set sigma_max to init_noise_level: {init_noise_level}")

    # Convert to model dtype
    model_dtype = next(model.model.parameters()).dtype
    LOG.debug(f"Model dtype: {model_dtype}")
    noise = noise.type(model_dtype)
    LOG.debug("Converting conditioning inputs to model dtype")
    # conditioning_inputs = {k: v.type(model_dtype) if v is not None and hasattr(v, 'type') else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        LOG.debug("Using v-diffusion")
        # k-diffusion denoising process go!
        sampled = sample(model.model, noise, steps, 0, **conditioning_inputs, cfg_scale=cfg_scale, dist_shift=model.dist_shift, batch_cfg=True)
    elif diff_objective == "rectified_flow":
        LOG.debug("Using rectified flow")
        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
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

    # Return audio
    return sampled

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