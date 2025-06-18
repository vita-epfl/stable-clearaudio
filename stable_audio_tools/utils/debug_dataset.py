"""
Debug script to understand why the validation DataLoader is empty in distributed mode.
"""

import os
import sys
import json
import torch
import torch.distributed as dist

# Add the project path to sys.path
sys.path.append('/mnt/vita/scratch/vita-staff/users/alefevre/programs/stable-clearaudio')

from stable_audio_tools.data.dataset import create_dataloader_from_config

def debug_validation_dataset():
    print("Debugging validation dataset...")
    
    # Load the validation dataset config
    val_config_path = "stable_audio_tools/configs/dataset_configs/maestro_RCP_intense_eq_valid.json"
    with open(val_config_path) as f:
        val_dataset_config = json.load(f)
    
    # Load the model config to get parameters
    model_config_path = "stable_audio_tools/configs/model_configs/audio_restoration/stable_clearaudio_rcp.json"
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    print(f"Validation dataset config: {val_dataset_config}")
    print(f"Model sample rate: {model_config['sample_rate']}")
    print(f"Model sample size: {model_config['sample_size']}")
    
    try:
        # Create the validation dataloader
        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=32,  # Use smaller batch size for debugging
            num_workers=2,   # Use fewer workers for debugging
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False
        )
        
        print(f"Validation DataLoader created successfully!")
        print(f"Length of validation DataLoader: {len(val_dl)}")
        print(f"Dataset size: {len(val_dl.dataset)}")
        
        # Try to get one batch
        if len(val_dl) > 0:
            first_batch = next(iter(val_dl))
            print(f"First batch shape: {first_batch[0].shape if hasattr(first_batch[0], 'shape') else 'N/A'}")
        else:
            print("DataLoader is empty!")
            
    except Exception as e:
        print(f"Error creating validation DataLoader: {e}")
        import traceback
        traceback.print_exc()

def debug_distributed_sampling():
    print("\nDebugging distributed sampling simulation...")
    
    # Simulate what happens with DistributedSampler
    val_config_path = "stable_audio_tools/configs/dataset_configs/maestro_RCP_intense_eq_valid.json"
    with open(val_config_path) as f:
        val_dataset_config = json.load(f)
    
    model_config_path = "stable_audio_tools/configs/model_configs/audio_restoration/stable_clearaudio_rcp.json"
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    # Create dataset without DataLoader first
    from stable_audio_tools.data.dataset import AudioDirDataset
    
    try:
        # Extract dataset configuration
        dataset_configs = val_dataset_config["datasets"]
        audio_dir_config = dataset_configs[0]
        
        print(f"Audio directory path: {audio_dir_config['path']}")
        print(f"Checking if path exists: {os.path.exists(audio_dir_config['path'])}")
        
        if os.path.exists(audio_dir_config['path']):
            files = os.listdir(audio_dir_config['path'])
            print(f"Files in directory: {len(files)}")
            if len(files) > 0:
                print(f"First few files: {files[:5]}")
        
        # Create the dataset directly
        dataset = AudioDirDataset(
            audio_dirs=[audio_dir_config["path"]],
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            random_crop=val_dataset_config.get("random_crop", True),
            force_channels=None,
            custom_metadata_args=val_dataset_config.get("custom_metadata_args", None)
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Simulate distributed sampling
        world_size = 4
        for rank in range(world_size):
            from torch.utils.data.distributed import DistributedSampler
            
            # This will fail if distributed is not initialized, but let's see the error
            try:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
                print(f"Rank {rank}: DistributedSampler would have {len(sampler)} samples")
            except Exception as e:
                print(f"Rank {rank}: DistributedSampler failed: {e}")
                # Calculate manually what the sampler would give
                total_size = len(dataset)
                samples_per_rank = total_size // world_size
                remainder = total_size % world_size
                
                if rank < remainder:
                    samples_for_this_rank = samples_per_rank + 1
                else:
                    samples_for_this_rank = samples_per_rank
                    
                print(f"Rank {rank}: Would get approximately {samples_for_this_rank} samples")
    
    except Exception as e:
        print(f"Error in distributed sampling debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting dataset debugging...")
    debug_validation_dataset()
    debug_distributed_sampling()
