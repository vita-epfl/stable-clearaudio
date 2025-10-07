# Dataset Builder Script: `build_degraded_dataset.py`

This document provides instructions on how to use the `build_degraded_dataset.py` script to generate datasets of clean and degraded audio files for training and evaluation of audio restoration models.

## Overview

The script processes a source directory of audio files to create a new dataset. Its main functions are:

1.  **Trimming**: Extracts a random segment of a specified duration from each source file.
2.  **Degradation**: Applies a series of audio degradation effects to the trimmed audio to create a 'degraded' version.
3.  **Output Generation**: Saves the clean (trimmed) and degraded audio files to separate directories.
4.  **Visualization**: Generates and saves frequency spectrum visualizations for the clean, degraded, and comparative (clean vs. degraded) audio to help in analyzing the effects of the degradation.

## Configuration

The script is controlled by a JSON configuration file. You must provide the path to this file using the `--config` command-line argument.

### Configuration Parameters

-   `dataset_path` (str): Path to the source dataset. This can be a directory containing audio files (`.wav`, `.mp3`, `.flac`) or a path to a single audio file.
-   `degraded_output_dir` (str): The directory where the generated degraded audio files will be saved.
-   `clean_output_dir` (str): The directory where the corresponding clean (trimmed) audio files will be saved.
-   `low_quality_effects_dir` (str): Path to a directory containing impulse responses or other files needed for certain degradation presets.
-   `build_clean_dataset` (bool): If `true`, the script will save the clean, trimmed audio segments alongside the degraded ones.
-   `duration` (float): The duration (in seconds) of the audio segments to be extracted from the source files.
-   `num_files` (int): The maximum number of audio files to process from the source dataset. Set to `-1` to process all files.
-   `degradation_presets` (List[str]): A list of strings specifying the degradation effects to apply. These presets are used by the `get_metadata_on_the_fly` function.

### Example Configuration (`build_degraded_config_workstation.json`)

```json
{
    "dataset_path": "/path/to/your/source/audio",
    "degraded_output_dir": "./data/degraded",
    "clean_output_dir": "./data/clean",
    "low_quality_effects_dir": "./assets/low_quality_effects",
    "build_clean_dataset": true,
    "duration": 30.0,
    "num_files": 100,
    "degradation_presets": ["telephone", "radio"]
}
```

## How to Run

To avoid `ModuleNotFoundError`, it is recommended to run the script as a module from the root directory of the `stable-clearaudio` project.

1.  Navigate to the project's root directory:
    ```bash
    cd /path/to/stable-clearaudio
    ```

2.  Execute the script using the following command structure:
    ```bash
    python stable_audio_tools.configs.dataset_configs.build_dataset.build_degraded_dataset --config [PATH_TO_CONFIG_JSON]
    ```

    **Example (workstation):**
    ```bash
    python stable_audio_tools.configs.dataset_configs.build_dataset.build_degraded_dataset --config stable_audio_tools/configs/dataset_configs/build_dataset/build_degraded_config_workstation.json
    ```

    **Example (RCP):**
    ```bash
    python stable_audio_tools.configs.dataset_configs.build_dataset.build_degraded_dataset --config stable_audio_tools/configs/dataset_configs/build_dataset/build_degraded_config_RCP.json
    ```

## Outputs

The script will create the output directories specified in the configuration if they don't exist. For each processed source file, the following files will be generated:

-   **Degraded Audio**: A `.wav` file in `degraded_output_dir`.
    -   Example: `source_file_degraded_telephone_radio.wav`
-   **Degraded Audio Visualization**: A `.png` frequency analysis plot in `degraded_output_dir`.
    -   Example: `source_file_degraded_freq_analysis.png`

If `build_clean_dataset` is `true`:

-   **Clean Audio**: A `.wav` file in `clean_output_dir`.
    -   Example: `source_file_clean.wav`
-   **Clean Audio Visualization**: A `.png` frequency analysis plot in `clean_output_dir`.
    -   Example: `source_file_freq_analysis.png`
-   **Comparative Visualization**: A `.png` plot comparing the frequency spectrums of the clean and degraded audio, saved in `degraded_output_dir`.
    -   Example: `source_file_freq_comparison.png`