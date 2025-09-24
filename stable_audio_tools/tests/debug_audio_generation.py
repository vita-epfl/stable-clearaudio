#!/usr/bin/env python3
"""
Debug script to test generate_diffusion_cond_restoration with different parameters
"""
import torch
import torchaudio
import numpy as np
from stable_audio_tools.interface.interfaces.restoration import generate_restoration

def test_generation_params():
    """Test generation with different parameters to isolate the issue"""

    # Load a test degraded audio file
    try:
        degraded_audio, sr = torchaudio.load("path/to/test/degraded_audio.wav")
        degraded_audio = (sr, degraded_audio)
    except:
        print("Please provide a path to a test degraded audio file")
        return

    test_configs = [
        {"cfg_rescale": 0.0, "name": "cfg_0"},
        {"cfg_rescale": 0.1, "name": "cfg_0.1"},
        {"cfg_rescale": 0.5, "name": "cfg_0.5"},
        {"steps": 50, "name": "steps_50"},
        {"steps": 100, "name": "steps_100"},
        {"sampler_type": "euler", "name": "euler_sampler"},
    ]

    for config in test_configs:
        print(f"Testing config: {config['name']}")
        try:
            result = generate_restoration(
                degraded_audio=degraded_audio,
                cfg_rescale=config.get("cfg_rescale", 0.0),
                steps=config.get("steps", 30),
                sampler_type=config.get("sampler_type", "euler"),
                # Add other parameters as needed
            )
            print(f"Success: {config['name']}")
        except Exception as e:
            print(f"Error with {config['name']}: {e}")

if __name__ == "__main__":
    test_generation_params()
