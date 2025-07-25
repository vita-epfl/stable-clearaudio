import os
import argparse
import random
import shutil
from glob import glob
from tqdm import tqdm

def find_audio_files(directory):
    """Recursively finds all audio files in a directory, excluding target folders."""
    extensions = ('*.wav', '*.flac', '*.mp3')
    audio_files = []
    
    # Construct paths to exclude
    train_dir_path = os.path.join(directory, 'maestro_full_train')
    valid_dir_path = os.path.join(directory, 'maestro_full_valid')

    print(f"Searching for audio files in '{directory}'...")
    all_files = []
    for ext in extensions:
        # Search recursively
        all_files.extend(glob(os.path.join(directory, '**', ext), recursive=True))

    # Filter out files that are already in the target directories
    for f in all_files:
        if not f.startswith(train_dir_path) and not f.startswith(valid_dir_path):
            audio_files.append(f)
            
    return audio_files

def split_dataset(source_dir, train_ratio=0.95):
    """
    Splits audio files from a source directory into training and validation sets.

    Args:
        source_dir (str): The directory containing the audio files.
        train_ratio (float): The proportion of files to be used for training.
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    train_dir = os.path.join(source_dir, 'maestro_full_train')
    valid_dir = os.path.join(source_dir, 'maestro_full_valid')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    audio_files = find_audio_files(source_dir)
    
    if not audio_files:
        print("No audio files found to move.")
        return

    print(f"Found {len(audio_files)} audio files to process.")
    
    random.shuffle(audio_files)

    split_index = int(len(audio_files) * train_ratio)
    train_files = audio_files[:split_index]
    valid_files = audio_files[split_index:]

    print(f"Moving {len(train_files)} files to '{train_dir}'...")
    for f in tqdm(train_files, desc="Moving train files"):
        try:
            shutil.move(f, os.path.join(train_dir, os.path.basename(f)))
        except Exception as e:
            print(f"Could not move file {f}: {e}")


    print(f"Moving {len(valid_files)} files to '{valid_dir}'...")
    for f in tqdm(valid_files, desc="Moving validation files"):
        try:
            shutil.move(f, os.path.join(valid_dir, os.path.basename(f)))
        except Exception as e:
            print(f"Could not move file {f}: {e}")

    print("\nDataset splitting complete.")
    print(f"Training set size: {len(os.listdir(train_dir))}")
    print(f"Validation set size: {len(os.listdir(valid_dir))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split an audio dataset into training and validation sets.')
    parser.add_argument('--source_dir', type=str, required=True, 
                        help='Path to the directory with audio files (e.g., /mnt/vita/scratch/datasets/maestro-v3.0.0).')
    parser.add_argument('--train_ratio', type=float, required=True, 
                        help='Ratio of files to be used for training (e.g., 0.95).')
    
    args = parser.parse_args()

    split_dataset(args.source_dir, args.train_ratio)