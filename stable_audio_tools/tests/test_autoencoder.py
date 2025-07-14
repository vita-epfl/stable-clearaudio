import torch
import torchaudio
import json
import os
import sys
from pathlib import Path
# Ensure local project root is first in Python path so tests use local sources
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import argparse
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict

def main():
    parser = argparse.ArgumentParser(description="Test Autoencoder Forward Pass")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config JSON file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input clean audio file (.wav)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the reconstructed audio file (.wav)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional path to the pretransform checkpoint (.ckpt)")
    parser.add_argument("--pretrained_ckpt_path", type=str, default=None, help="Optional path to the pretrained checkpoint (.ckpt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        model_config = json.load(f)

    sample_rate = model_config.get("sample_rate", 44100)
    sample_size = model_config.get("sample_size", None) # Expecting sample size in config
    audio_channels = model_config.get("audio_channels", 2)

    if sample_size is None:
        raise ValueError("Config must contain 'sample_size'")

    # Create model and extract pretransform (autoencoder)
    print("Creating model...")
    model = create_model_from_config(model_config)
    if not hasattr(model, 'pretransform') or model.pretransform is None:
        raise AttributeError("Model created from config does not have a 'pretransform' attribute.")
    
    ae = model.pretransform
    ae = ae.to(args.device)
    ae.eval()

    # Load checkpoint if provided
    if args.ckpt_path:
        print(f"Loading pretransform checkpoint from {args.ckpt_path}")
        copy_state_dict(ae, load_ckpt_state_dict(args.ckpt_path))
    else:
        print("No pretransform checkpoint provided, using initialized weights.")

    # Load pretrained checkpoint if provided
    if args.pretrained_ckpt_path:
        print(f"Loading pretrained checkpoint from {args.pretrained_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
    else:
        print("No pretrained checkpoint provided, using initialized weights.")

    # Load audio
    print(f"Loading audio from {args.input}")
    try:
        audio, sr = torchaudio.load(args.input)
    except Exception as e:
        print(f"Error loading audio file {args.input}: {e}")
        return

    # Resample if necessary
    if sr != sample_rate:
        print(f"Resampling audio from {sr} Hz to {sample_rate} Hz")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)

    # Ensure correct channels
    if audio.shape[0] != audio_channels:
        if audio.shape[0] == 1 and audio_channels == 2:
            print("Converting mono audio to stereo by duplicating channel.")
            audio = torch.cat([audio, audio], dim=0)
        elif audio.shape[0] == 2 and audio_channels == 1:
             print("Converting stereo audio to mono by averaging channels.")
             audio = torch.mean(audio, dim=0, keepdim=True)
        else:
             print(f"Warning: Audio has {audio.shape[0]} channels, expected {audio_channels}. Trying to proceed...")
             # You might want to add more sophisticated channel handling here

    # Pad or trim to sample_size
    current_length = audio.shape[-1]
    if current_length < sample_size:
        print(f"Padding audio from {current_length} to {sample_size} samples.")
        padding = sample_size - current_length
        audio = torch.nn.functional.pad(audio, (0, padding))
    elif current_length > sample_size:
        print(f"Trimming audio from {current_length} to {sample_size} samples.")
        audio = audio[..., :sample_size]

    # Prepare batch and move to device
    batch = audio.unsqueeze(0).to(args.device)
    print(f"Input batch shape: {batch.shape}")

    # Run forward pass
    print("Running autoencoder forward pass (encode + decode)...")
    with torch.no_grad():
        if hasattr(ae, 'encode') and hasattr(ae, 'decode'):
            latents = ae.encode(batch)
            if isinstance(latents, tuple): # Handle VAE case where encode might return mean and logvar
                latents = latents[0] # Assuming the first element is the mean/sample
            reconstructed_batch = ae.decode(latents)
        elif hasattr(ae, 'forward'):
            # Fallback if AE has a single forward method
             reconstructed_batch = ae(batch)
        else:
            raise AttributeError("Autoencoder object does not have required 'encode'/'decode' or 'forward' methods.")
        
    print(f"Output batch shape: {reconstructed_batch.shape}")

    # Process output
    reconstructed_audio = reconstructed_batch.squeeze(0).cpu()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save output
    print(f"Saving reconstructed audio to {args.output}")
    try:
        torchaudio.save(args.output, reconstructed_audio, sample_rate)
        print("Done.")
    except Exception as e:
        print(f"Error saving audio file {args.output}: {e}")

if __name__ == '__main__':
    main()
