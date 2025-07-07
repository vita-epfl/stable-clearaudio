import numpy as np
import os
import gc
import glob
import json
import shutil
import gradio as gr
import logging
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")

import logging

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = True

from .diffusion_cond import generate_restoration

def create_restoration_sampling_ui():
    diffusion_objective = model.diffusion_objective

    is_rf = diffusion_objective == "rectified_flow"
    
    with gr.Row():                
        generate_button = gr.Button("Generate", variant="primary", scale=1)
        
    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                # Steps slider
                default_steps = 30
                steps_slider = gr.Slider(
                    minimum=1, maximum=500, step=1, value=default_steps, label="Steps"
                )

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    # Seed
                    seed_textbox = gr.Textbox(
                        label="Seed (set to -1 for random seed)", value="-1"
                    )

                with gr.Row():
                    cfg_rescale_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.01,
                        value=0.0,
                        label="CFG rescale amount",
                    )

                with gr.Row():
                    # Sampler params
                    if is_rf:
                        sampler_types = ["euler", "rk4", "dpmpp"]
                        default_sampler_type = "euler"
                    else:
                        sampler_types = [
                            "dpmpp-2m-sde",
                            "dpmpp-3m-sde",
                            "dpmpp-2m",
                            "k-heun",
                            "k-lms",
                            "k-dpmpp-2s-ancestral",
                            "k-dpm-2",
                            "k-dpm-adaptive",
                            "k-dpm-fast",
                            "v-ddim",
                            "v-ddim-cfgpp",
                        ]
                        default_sampler_type = "dpmpp-3m-sde"
                    sampler_type_dropdown = gr.Dropdown(
                        sampler_types, label="Sampler type", value=default_sampler_type
                    )
                    sigma_min_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=0.01,
                        label="Sigma min",
                        visible=not is_rf,
                    )
                    sigma_max_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1000.0,
                        step=0.1,
                        value=100,
                        label="Sigma max",
                        visible=not is_rf,
                    )
                    rho_slider = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        step=0.01,
                        value=1.0,
                        label="Sigma curve strength",
                        visible=not is_rf,
                    )

            with gr.Accordion("Output params", open=False):
                with gr.Row():
                    file_format_dropdown = gr.Dropdown(
                        [
                            "wav",
                            "flac",
                            "mp3 320k",
                            "mp3 v0",
                            "mp3 128k",
                            "m4a aac_he_v2 64k",
                            "m4a aac_he_v2 32k",
                        ],
                        label="File format",
                        value="wav",
                    )
                    preview_every_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                        label="Spec Preview Every N Steps",
                    )
                    metrics_every_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=30,
                        label="Compute Metrics Every N Steps"
                    )
            with gr.Accordion("Audio Inputs", open=True):
                with gr.Row():
                    degraded_audio_files = gr.File(label="Degraded audio files", file_count="multiple")
                    clean_audio = gr.Audio(label="Clean reference audio (optional)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        degraded_audio_dropdown = gr.Dropdown(label="Select degraded audio to play", interactive=True)
                    with gr.Column(scale=1):
                        degraded_audio_player = gr.Audio(label="Degraded audio player")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Processing Options")
                    
                with gr.Row():
                    process_folder_path = gr.Textbox(
                        label="Audio Folder",
                        placeholder="Path to folder where the algorithm should search for audio files (e.g. audio/degraded/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_04_Track04_wav)",
                    )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        model_name = gr.Textbox(
                            label="Model name",
                            placeholder="Name of the model for results file (e.g. intense_equalizer)",
                        )
                    with gr.Column(scale=1):
                        effects_list = gr.Textbox(
                            label="Effects list",
                            placeholder="Comma-separated list of effects (e.g. equalizer,bass,overdrive)",
                        )

            inputs = [
                steps_slider,
                preview_every_slider,
                metrics_every_slider,
                seed_textbox,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                rho_slider,
                cfg_rescale_slider,
                file_format_dropdown,
                degraded_audio_files,
                clean_audio,
                process_folder_path,
                model_name,
                effects_list,
            ]            

        with gr.Column():
            output_audio_dropdown = gr.Dropdown(label="Select generated audio")
            output_audio_player = gr.Audio(label="Audio player")
            audio_spectrogram_output = gr.Gallery(label="Output spectrograms", show_label=True, elem_id="spectrogram_gallery", columns=2, height=400)
            
            # Add metrics display section
            with gr.Accordion("Restoration Metrics", open=True):
                metrics_plots = gr.Image(label="Metrics Plots", type="numpy")
            
            # Use different target based on model type
            send_to_init_button = gr.Button("Send to degraded audio", scale=1)
            send_to_init_button.click(
                fn=lambda audio: [audio] if audio else None,
                inputs=[output_audio_player],
                outputs=[degraded_audio_files]
                )

    def process_folder_files(process_folder_path, model_name, effects_list):
        
        LOG.info(f"Processing folder: {process_folder_path}")
            
        # Process effects list if provided
        if model_name and effects_list:
            effects = [effect.strip() for effect in effects_list.split(",")]
            LOG.info(f"Looking for effects: {effects}")
            
            # Initialize the consolidated results JSON file with the expected structure
            consolidated_results_file = os.path.join(output_dir, f"results_{model_name}.json")
            
            # If the file already exists, load it to keep existing data
            if os.path.exists(consolidated_results_file):
                try:
                    with open(consolidated_results_file, 'r') as f:
                        consolidated_results = json.load(f)
                        
                        # Reformat existing data to ensure correct structure
                        # Fix audio paths and losses format for all existing entries
                        if "data" in consolidated_results:
                            for audio_name, audio_data in list(consolidated_results["data"].items()):
                                if audio_name not in ["light", "standard", "strong", "unknown"]:
                                    for duration, duration_data in list(audio_data.items()):
                                        for intensity, intensity_data in list(duration_data.items()):
                                            if "effects" in intensity_data:
                                                for i, effect in enumerate(intensity_data["effects"]):
                                                    # Fix degraded audio path
                                                    if "degraded" in effect and "audio" in effect["degraded"]:
                                                        old_path = effect["degraded"]["audio"]
                                                        if "/" in old_path.replace("audio/degraded/", ""):
                                                            # Extract parts from path
                                                            parts = old_path.split("/")
                                                            audio_base = parts[2]  # After audio/degraded/
                                                            
                                                            # Find effect name (last part of path without extension)
                                                            effect_name = os.path.splitext(parts[-1])[0]
                                                            
                                                            # Get duration and intensity from middle parts
                                                            dur_part = next((p for p in parts[3:-1] if p.endswith('s') and p[:-1].isdigit()), "5s")
                                                            int_part = next((p for p in parts[3:-1] if p in ["light", "standard", "strong"]), "standard")
                                                            
                                                            # Create new path format
                                                            new_path = f"audio/degraded/{audio_base}_{dur_part}_{int_part}_{effect_name}.wav"
                                                            effect["degraded"]["audio"] = new_path
                                                            LOG.info(f"Fixed degraded audio path: {old_path} -> {new_path}")
                                                    
                                                    # Fix losses format for degraded
                                                    if "degraded" in effect and "losses" in effect["degraded"] and isinstance(effect["degraded"]["losses"], dict):
                                                        for loss_type in ["l1", "l2", "snr"]:
                                                            if loss_type in effect["degraded"]["losses"] and isinstance(effect["degraded"]["losses"][loss_type], list):
                                                                # Convert array to single value (use first value or average)
                                                                values = effect["degraded"]["losses"][loss_type]
                                                                if values:
                                                                    effect["degraded"]["losses"][loss_type] = values[0] if len(values) > 0 else 0
                                                    
                                                    # Fix losses format for restored
                                                    if "restored" in effect and "losses" in effect["restored"] and isinstance(effect["restored"]["losses"], dict):
                                                        for loss_type in ["l1", "l2", "snr"]:
                                                            if loss_type in effect["restored"]["losses"] and isinstance(effect["restored"]["losses"][loss_type], list):
                                                                # Convert array to single value (use first value or average)
                                                                values = effect["restored"]["losses"][loss_type]
                                                                if values:
                                                                    effect["restored"]["losses"][loss_type] = values[0] if len(values) > 0 else 0
                        
                        LOG.info("Restructured existing results JSON to fix formats")
                except Exception as e:
                    LOG.warning(f"Could not load existing results file: {e}")
                    consolidated_results = {
                        "model": model_name,
                        "data": {}
                    }
            else:
                consolidated_results = {
                    "model": model_name,
                    "data": {}
                }
            
            # Delete any intensity entries that might be at the wrong level
            # Also delete any entries with empty or 'unknown' durations
            keys_to_delete = []
            for key in consolidated_results["data"].keys():
                # Remove intensity keys at top level (they should be deeper in the hierarchy)
                if key in ["light", "standard", "strong", "unknown"]:
                    keys_to_delete.append(key)
                # Remove any audio_name entries that contain "unknown" duration key
                elif isinstance(consolidated_results["data"][key], dict):
                    duration_keys_to_delete = []
                    for duration_key in consolidated_results["data"][key].keys():
                        if duration_key == "unknown":
                            duration_keys_to_delete.append(duration_key)
                        # Also check for empty intensity structures
                        elif duration_key in consolidated_results["data"][key]:
                            intensity_keys_to_delete = []
                            for intensity_key in consolidated_results["data"][key][duration_key].keys():
                                if intensity_key in ["light", "standard", "strong"]:
                                    intensity_struct = consolidated_results["data"][key][duration_key][intensity_key]
                                    # Delete empty intensity structures (only containing empty effects list)
                                    if isinstance(intensity_struct, dict) and "effects" in intensity_struct and len(intensity_struct["effects"]) == 0:
                                        intensity_keys_to_delete.append(intensity_key)
                            # Delete empty intensity structures
                            for intensity_key in intensity_keys_to_delete:
                                del consolidated_results["data"][key][duration_key][intensity_key]
                    # Delete empty duration structures
                    for duration_key in duration_keys_to_delete:
                        del consolidated_results["data"][key][duration_key]
            
            # Delete top-level invalid entries
            for key in keys_to_delete:
                del consolidated_results["data"][key]
            
            # Ensure model_name is sanitized for filenames
            model_name = model_name.replace(" ", "_")
            
            # Find all audio files that match the effects in subfolders
            for root, dirs, files in os.walk(process_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_name = os.path.basename(file_path)
                    file_name_without_ext, ext = os.path.splitext(file_name)
                    
                    if ext.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                        # Use STRICT matching of effect names
                        # Only match if it's the exact filename or an exact part separated by underscores
                        for effect in effects:
                            # Option 1: Exact filename match (without extension)
                            exact_match = (effect == file_name_without_ext)
                            
                            if exact_match:
                                if effect not in effects_files:
                                    effects_files[effect] = []
                                effects_files[effect].append(file_path)
                                
                            # Extra check: prevent ANY partial/substring matches
                            # This is a safety check to ensure only EXACT matches are accepted
                            if not (exact_match):
                                # Remove from effects_files if somehow it got added
                                if effect in effects_files and file_path in effects_files[effect]:
                                    effects_files[effect].remove(file_path)
            
            # Create a list of files to process from the effects files
            for effect, files in effects_files.items():
                folder_files.extend(files)
            
            if not folder_files:
                raise gr.Error(f"No audio files matching effects {effects} found in folder {process_folder_path}")
            
            LOG.info(f"Found {len(folder_files)} audio files to process")
            
            # Utilisation du output_dir déjà défini
            LOG.info(f"Results will be saved in: {output_dir}")
        else:
            # Standard folder processing without effects list
            extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
            for ext in extensions:
                folder_files.extend(glob.glob(os.path.join(process_folder_path, f"*{ext}")))
            
            if not folder_files:
                raise gr.Error(f"No audio files found in folder {process_folder_path}")
            
            LOG.info(f"Found {len(folder_files)} audio files to process")
            
            # Using the centralized output directory already defined
            LOG.info(f"Results will be saved in: {output_dir}")

    def generate_multiple_with_plots(steps, preview_every, metrics_every, seed, sampler_type, sigma_min, sigma_max, rho, cfg_rescale, file_format, degraded_audio_files, clean_audio, process_folder_path=None, model_name=None, effects_list=None):
        """
        Generate multiple audio files from a folder of degraded audio files.

        Args:
            steps (int): Number of steps to generate audio.
            preview_every (int): Preview every N steps.
            metrics_every (int): Compute metrics every N steps.
            seed (int): Seed for random number generator.
            sampler_type (str): Type of sampler to use.
            sigma_min (float): Minimum sigma value.
            sigma_max (float): Maximum sigma value.
            rho (float): Rho value.
            cfg_rescale (float): CFG rescale amount.
            file_format (str): File format to save audio files.
            file_naming (str): File naming format.
            degraded_audio_files (list): List of degraded audio files.
            clean_audio (gr.Audio): Clean reference audio.
            process_folder_path (str): Path to folder to process.
            model_name (str): Name of the model to use.
            effects_list (str): List of effects to apply.
        
        Returns:
            audios (list): List of generated audio files.
            spectrograms (list): List of spectrograms for each audio file.
            plots (list): List of plots for each audio file.
        """
        
        # Check if we have a folder path to process
        folder_files = []
        effects_files = {}
        consolidated_results = None
        
        # Define output directory for JSON results - always in output/batch_processing
        output_dir = os.path.join("output", "batch_processing")
        os.makedirs(output_dir, exist_ok=True)
        
        # Define directories for restored and degraded audio files
        restored_audio_dir = os.path.join("output", "audio", "restored")
        degraded_audio_dir = os.path.join("output", "audio", "degraded")
        os.makedirs(restored_audio_dir, exist_ok=True)
        os.makedirs(degraded_audio_dir, exist_ok=True)
        
        if process_folder_path and os.path.isdir(process_folder_path):
            folder_files = process_folder_files(process_folder_path, model_name, effects_list)
            
        elif not degraded_audio_files:
            raise gr.Error("No degraded audio files provided.")

        labels = []
        
        # If we're processing from a folder
        if process_folder_path and folder_files:
            for f in folder_files:
                filename = os.path.basename(f)
                filename_no_ext = os.path.splitext(filename)[0]
                labels.append(filename_no_ext)
                
                # If using effects list, extract effect name from filename
                if effects_list:
                    # Find which effect this file corresponds to
                    current_effect = None
                    for effect in effects_files.keys():
                        # Check for exact match
                        exact_match = (effect == filename_no_ext)
                        
                        if exact_match:
                            LOG.info(f"Found exact effect match for {effect} in {f}")
                            current_effect = effect
                            break
                        else:
                            LOG.info(f"Skipping non-exact match for {effect} in {f}")
                    
                    if current_effect:
                        # This pre-initialization logic is faulty and creates unwanted JSON structures.
                        # The main processing loop handles the JSON creation correctly.
                        # Therefore, this block is being removed.
                        pass
        # If we're uploading individual files
        if degraded_audio_files and not process_folder_path:
            for f in degraded_audio_files:
                filename = os.path.basename(f.name)
                filename_no_ext = os.path.splitext(filename)[0]
                labels.append(filename_no_ext)

        all_metrics = []
        output_audios_list = []
        output_spectrograms_list = []
        labels = []
        # Configure environment variables for batch processing
        # Disable creation of temporary directories
        os.environ["STABLE_AUDIO_NO_DATE_FOLDER"] = "1"
        os.environ["STABLE_AUDIO_BATCH_PROCESSING"] = "1"
        
        # Use the specified output directory for all outputs
        # Don't create degradation_processing subfolder
        batch_processing_dir = output_dir
        os.makedirs(batch_processing_dir, exist_ok=True)
        # Override the default temp dir to control where files are saved
        os.environ["STABLE_AUDIO_CUSTOM_TMP_DIR"] = batch_processing_dir
        
        # Determine which files to process - either from folder or uploaded files
        files_to_process = folder_files if process_folder_path and folder_files else [f.name for f in degraded_audio_files]
            
        for i, degraded_audio_path in enumerate(files_to_process):
            LOG.info(f"Processing file {i+1}/{len(files_to_process)}: {os.path.basename(degraded_audio_path)}")
            
            try:
                audio_data, sr = torchaudio.load(degraded_audio_path)
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.repeat(2,1)
                degraded_audio_input = (sr, audio_data.numpy())
            except Exception as e:
                LOG.error(f"Error loading audio file {degraded_audio_path}: {str(e)}")
                continue

            audio, spectrograms, metrics = generate_restoration(
                steps=steps,
                preview_every=preview_every,
                metrics_every=metrics_every,
                seed=seed,
                sampler_type=sampler_type,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                cfg_rescale=cfg_rescale,
                file_format=file_format,
                degraded_audio=degraded_audio_input,
                clean_audio=clean_audio,
                batch_size=1,
                degraded_audio_filename=degraded_audio_path  # Pass the filename here
            )
            
            # Process based on whether we're using effects list or standard folder processing
            if model_name and effects_list and consolidated_results:
                original_filename = os.path.basename(degraded_audio_path)
                original_filename_without_ext = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(audio)[1]
                
                # Find the effect this file corresponds to
                current_effect = None
                
                # Sort effects by length (descending) to avoid partial matches
                # This ensures 'intense_equalizer' is checked before 'equalizer'
                sorted_effects = sorted(effects_files.keys(), key=len, reverse=True)
                
                for effect in sorted_effects:
                    # STRICT EXACT EFFECT MATCHING - only exact match allowed
                    if effect == original_filename_without_ext:
                        current_effect = effect
                        LOG.info(f"✓ Exact match: effect '{effect}' equals file name '{original_filename_without_ext}'")
                        break
                    
                    # NO PARTIAL MATCHES - STRICTER RULE: effect must be exactly equal to the filename
                    # If we're looking for 'equalizer', don't match 'intense_equalizer'
                    LOG.info(f"✗ Effect '{effect}' does NOT match file '{original_filename_without_ext}' - skipping")
                    continue
                
                if current_effect:
                    # Extract path components for structured data using a single approach
                    # Convert backslashes to forward slashes for consistent processing
                    normalized_path = degraded_audio_path.replace('\\', '/')
                    path_parts = normalized_path.split('/')
                    
                    # Initialize default values
                    audio_name = "unknown"
                    duration = "5s"
                    intensity = "standard"
                    effect_name = current_effect  # Default to the matched effect
                    
                    # Extract the original audio name, duration, intensity, and effect from the path
                    # Expected format: audio/degraded/AUDIO_NAME/DURATION/INTENSITY/EFFECT.wav
                    # Also check for pattern in filename: AUDIO_NAME_DURATION_INTENSITY_EFFECT.wav
                    
                    # First, try to extract from directory structure
                    for i, part in enumerate(path_parts):
                        if part == "degraded" and i < len(path_parts) - 1:
                            # Check if this is the new format (flat) or old format (nested dirs)
                            next_part = path_parts[i+1]
                            
                            if next_part.endswith(".wav") or next_part.endswith(".mp3"):
                                # This is the flat format with filename pattern AUDIO_NAME_DURATION_INTENSITY_EFFECT.wav
                                filename_without_ext = os.path.splitext(next_part)[0]
                                # Try to parse the filename parts
                                filename_parts = filename_without_ext.split('_')
                                
                                if len(filename_parts) >= 4:  # At least audio_name, duration, intensity, effect
                                    # Effect is the last part
                                    effect_name = filename_parts[-1]
                                    # Intensity is second to last
                                    intensity = filename_parts[-2]
                                    # Duration is third to last
                                    duration = filename_parts[-3]
                                    # Audio name is everything else joined
                                    audio_name = '_'.join(filename_parts[:-3])
                            else:
                                # This is the nested directory format
                                # The audio name is just after 'degraded'
                                audio_name = path_parts[i+1]
                                
                                # The duration is two levels after 'degraded', if available
                                if i + 2 < len(path_parts):
                                    duration = path_parts[i+2]
                                    
                                # The intensity is three levels after 'degraded', if available
                                if i + 3 < len(path_parts):
                                    intensity = path_parts[i+3]
                                    
                                # The effect is four levels after 'degraded', if available
                                if i + 4 < len(path_parts):
                                    # Remove extension if present
                                    effect_name = os.path.splitext(path_parts[i+4])[0]
                            break
                    
                    LOG.info(f"Extracted: Audio={audio_name}, Duration={duration}, Intensity={intensity}, Effect={effect_name}")
                    
                    # Use extracted values or default values
                    audio_name = audio_name or os.path.splitext(os.path.basename(degraded_audio_path))[0].split('_')[0]
                    duration = duration or "5s"
                    intensity = intensity or "standard"
                    effect_name = effect_name or current_effect or "unknown_effect"
                    
                    # Format the filename with audio name, duration, intensity, effect, and restored suffix
                    # Example: MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_04_Track04_wav_5s_light_bass_restored.wav
                    restored_basename = f"{audio_name}_{duration}_{intensity}_{effect_name}_restored{ext}"
                    
                    # Copy the audio file to restored folder
                    new_filepath = os.path.join(restored_audio_dir, restored_basename)
                    LOG.info(f"Saving restored audio to: {new_filepath}")
                    shutil.copy2(audio, new_filepath)
                    
                    # FIXED JSON STRUCTURE: data should be organized as data[audio_name][duration][intensity]
                    # Reset audio_name to prevent wrong nesting
                    # We need to make sure we're not using intensity as audio_name or similar confusion
                    
                    # Debug the values we're using for nesting
                    LOG.info(f"JSON Structure: audio_name='{audio_name}', duration='{duration}', intensity='{intensity}'")
                    
                    # Make sure we have valid values for each level and they're not getting confused
                    if audio_name in ["light", "standard", "strong"]:
                        LOG.warning(f"audio_name '{audio_name}' looks like an intensity - using proper filename instead")
                        # Try to extract a better audio_name
                        audio_name = os.path.splitext(os.path.basename(degraded_audio_path))[0].split('_')[0]
                        if not audio_name or audio_name in ["light", "standard", "strong"]:
                            audio_name = "MIDI-Unprocessed"
                    
                    # Make sure duration has an 's' suffix
                    if not str(duration).endswith('s') and str(duration).isdigit():
                        duration = f"{duration}s"
                    
                    # Ensure intensity is one of the valid values 
                    if intensity not in ["light", "standard", "strong"]:
                        LOG.warning(f"Invalid intensity '{intensity}' - using 'standard' instead")
                        intensity = "standard"
                    
                    # Now create the proper structure
                    LOG.info(f"Creating JSON structure: data[{audio_name}][{duration}][{intensity}]")
                    if audio_name not in consolidated_results["data"]:
                        consolidated_results["data"][audio_name] = {}
                    
                    if duration not in consolidated_results["data"][audio_name]:
                        consolidated_results["data"][audio_name][duration] = {}
                        
                    if intensity not in consolidated_results["data"][audio_name][duration]:
                        consolidated_results["data"][audio_name][duration][intensity] = {}
                    
                    if "effects" not in consolidated_results["data"][audio_name][duration][intensity]:
                        consolidated_results["data"][audio_name][duration][intensity]["effects"] = []
                    
                    # Check if this effect entry already exists
                    effect_exists = False
                    for existing_effect in consolidated_results["data"][audio_name][duration][intensity]["effects"]:
                        # Update comparison to check just the effect name, not the full path
                        if existing_effect["name"] == current_effect:
                            effect_exists = True
                            LOG.info(f"Effect {current_effect} already processed for this audio, updating it instead")
                            
                            # We already have metrics from generate_restoration call at line 1615
                            # No need to reload them from file - just use what we have
                            LOG.info(f"Using metrics from generate_restoration for effect {current_effect}")
                            # Log available metrics types
                            if metrics and isinstance(metrics, dict):
                                LOG.info(f"Available metrics keys: {list(metrics.keys())}")
                                if "degraded_metrics" in metrics and isinstance(metrics["degraded_metrics"], dict):
                                    LOG.info(f"Degraded metrics: {list(metrics['degraded_metrics'].keys())}")
                                if "restored_metrics" in metrics and isinstance(metrics["restored_metrics"], dict):
                                    LOG.info(f"Restored metrics: {list(metrics['restored_metrics'].keys())}")
                            else:
                                LOG.warning(f"No metrics available from generate_restoration for effect {current_effect}")
                            
                            # Restructure metrics if needed - the metrics from generate_restoration may be flat, not nested
                            degraded_metrics = {}
                            restored_metrics = {}
                            
                            # Check if metrics come in the expected nested format
                            if "degraded_metrics" in metrics and isinstance(metrics["degraded_metrics"], dict):
                                # Already in expected format
                                degraded_metrics = metrics["degraded_metrics"]
                                LOG.info("Using nested degraded_metrics structure")
                            elif isinstance(metrics, dict):
                                # Metrics are in a flat structure, need to sort them into degraded and restored
                                LOG.info("Restructuring flat metrics into degraded/restored categories")
                                
                                # Process metrics with 'degraded_' prefix or other indicators that they are degraded
                                for key, value in metrics.items():
                                    # Skip generation_params and timestamp but add some fallbacks
                                    if key in ['generation_params', 'timestamp', 'steps', 'sample_rate', 'sample_size']:
                                        continue
                                        
                                    # Make sure we have at least some metrics to display
                                    if not degraded_metrics and not key.startswith("degraded_") and not key.startswith("demo_") and not key.startswith("restoration_"):
                                        LOG.info(f"Adding fallback metric {key} to both degraded and restored")
                                        degraded_metrics[key] = value
                                        restored_metrics[key] = value
                                        
                                    # Put degraded_ prefixed metrics into degraded_metrics
                                    if key.startswith('degraded_'):
                                        # Remove the "degraded_" prefix for cleaner naming
                                        clean_key = key.replace('degraded_', '')
                                        degraded_metrics[clean_key] = value
                                        LOG.info(f"Added degraded metric: {clean_key} = {value}")
                                    # Put restoration success metrics into restored metrics
                                    elif key.startswith('restoration_success_'):
                                        clean_key = key.replace('restoration_success_', '')
                                        restored_metrics[clean_key] = value
                                        LOG.info(f"Added restoration metric: {clean_key} = {value}")
                                    # Any demo_ metrics go to restored (these are the model outputs)
                                    elif key.startswith('demo_'):
                                        # Remove the "demo_" prefix for cleaner naming
                                        clean_key = key.replace('demo_', '')
                                        restored_metrics[clean_key] = value
                                        LOG.info(f"Added demo metric as restored: {clean_key} = {value}")
                                    # Handle losses
                                    elif key.endswith('_loss'):
                                        if 'degraded_' + key in metrics:
                                            restored_metrics[key] = value
                                            LOG.info(f"Added loss metric to restored: {key} = {value}")
                                        else:
                                            # If we don't have a degraded version, this might be a shared metric
                                            degraded_metrics[key] = value
                                            restored_metrics[key] = value
                                            LOG.info(f"Added loss metric to both: {key} = {value}")
                                    
                                # Copy basic parameters to both metrics
                                for key in ['timestamp', 'steps', 'sample_rate', 'sample_size']:
                                    if key in metrics:
                                        degraded_metrics[key] = metrics[key]
                                        restored_metrics[key] = metrics[key]
                                        
                            LOG.info(f"Restructured metrics - degraded: {len(degraded_metrics)} keys, restored: {len(restored_metrics)} keys")
                            if len(degraded_metrics) > 0:
                                LOG.info(f"Degraded metrics keys: {list(degraded_metrics.keys())[:5]}...")
                            if len(restored_metrics) > 0:
                                LOG.info(f"Restored metrics keys: {list(restored_metrics.keys())[:5]}...")
                            
                            # Update degraded path and losses
                            degraded_basename = f"{audio_name}_{duration}_{intensity}_{effect_name}{ext}"
                            formatted_degraded_path = f"audio/degraded/{degraded_basename}"
                            
                            # Update the existing entry with correct format
                            existing_effect["degraded"]["audio"] = formatted_degraded_path
                            
                            # Preserve all metrics from _calculate_metrics and ensure they are scalars, not arrays
                            degraded_losses = {}
                            for metric_key, metric_value in degraded_metrics.items():
                                # Convert array values to scalars
                                if isinstance(metric_value, list) and len(metric_value) > 0:
                                    degraded_losses[metric_key] = metric_value[0]
                                else:
                                    degraded_losses[metric_key] = metric_value
                            
                            existing_effect["degraded"]["losses"] = degraded_losses
                            
                            # Copy the degraded audio file to the degraded folder
                            degraded_filepath = os.path.join(degraded_audio_dir, degraded_basename)
                            LOG.info(f"Saving degraded audio to: {degraded_filepath}")
                            try:
                                if os.path.exists(degraded_audio_path):
                                    shutil.copy2(degraded_audio_path, degraded_filepath)
                                else:
                                    LOG.warning(f"Source degraded audio not found: {degraded_audio_path}")
                            except Exception as e:
                                LOG.error(f"Error copying degraded audio file: {str(e)}")
                            
                            # Update restored entry too
                            json_restored_path = f"audio/restored/{audio_name}_{duration}_{intensity}_{effect_name}_restored{ext}"
                            existing_effect["restored"]["audio"] = json_restored_path
                            
                            # Preserve all metrics from _calculate_metrics and ensure they are scalars, not arrays
                            restored_losses = {}
                            for metric_key, metric_value in restored_metrics.items():
                                # Convert array values to scalars
                                if isinstance(metric_value, list) and len(metric_value) > 0:
                                    restored_losses[metric_key] = metric_value[0]
                                else:
                                    restored_losses[metric_key] = metric_value
                                    
                            existing_effect["restored"]["losses"] = restored_losses
                            break
                    
                    if not effect_exists:
                        # Initialize detailed metrics with empty dictionaries
                        detailed_metrics = {
                            "degraded_metrics": {},
                            "restored_metrics": {}
                        }
                        
                        # Get metrics from the model via metrics (from final_metrics)
                        if metrics and isinstance(metrics, dict):
                            LOG.info(f"Available metrics keys from generate_restoration: {list(metrics.keys())}")
                            
                            # Check if metrics already have a nested structure
                            if "degraded_metrics" in metrics and "restored_metrics" in metrics:
                                detailed_metrics["degraded_metrics"] = metrics["degraded_metrics"]
                                detailed_metrics["restored_metrics"] = metrics["restored_metrics"]
                                LOG.info(f"Using nested structure from metrics directly")
                            # Otherwise, check if the metrics have a flat structure with losses
                            elif "degraded" in metrics and "restored" in metrics:
                                # Extract from nested structure with losses
                                if "losses" in metrics["degraded"]:
                                    detailed_metrics["degraded_metrics"] = metrics["degraded"]["losses"]
                                    LOG.info(f"Extracted {len(detailed_metrics['degraded_metrics'])} degraded metrics from losses")
                                if "losses" in metrics["restored"]:
                                    detailed_metrics["restored_metrics"] = metrics["restored"]["losses"]
                                    LOG.info(f"Extracted {len(detailed_metrics['restored_metrics'])} restored metrics from losses")
                            else:
                                # Fallback: try to extract flat metrics
                                LOG.info("Metrics don't have expected structure, trying to extract manually")
                                for key, value in metrics.items():
                                    if key.startswith('degraded_') or key in ['lsd', 'ltas', 'sisdr', 'snr', 'stft', 'mel', 'latent_mse_loss', 'latent_l1_loss']:
                                        clean_key = key.replace('degraded_', '')
                                        detailed_metrics["degraded_metrics"][clean_key] = value
                                    elif key.startswith('restored_') or key.startswith('restoration_') or key.startswith('demo_'):
                                        clean_key = key.replace('restored_', '').replace('restoration_', '').replace('demo_', '')
                                        detailed_metrics["restored_metrics"][clean_key] = value
                        else:
                            LOG.warning("No metrics available from generate_restoration!")
                            
                        # Load metrics from the JSON file as a fallback
                        metrics_file = os.path.join(batch_processing_dir, "equalizer_metrics.json")
                        if os.path.exists(metrics_file) and (not detailed_metrics["degraded_metrics"] or not detailed_metrics["restored_metrics"]):
                            try:
                                with open(metrics_file, 'r') as f:
                                    file_metrics = json.load(f)
                                    if "degraded" in file_metrics and "losses" in file_metrics["degraded"]:
                                        detailed_metrics["degraded_metrics"] = file_metrics["degraded"]["losses"]
                                        LOG.info(f"Loaded {len(detailed_metrics['degraded_metrics'])} degraded metrics from file")
                                    if "restored" in file_metrics and "losses" in file_metrics["restored"]:
                                        detailed_metrics["restored_metrics"] = file_metrics["restored"]["losses"]
                                        LOG.info(f"Loaded {len(detailed_metrics['restored_metrics'])} restored metrics from file")
                            except Exception as e:
                                LOG.error(f"Failed to load metrics from file: {str(e)}")
                        
                        # Log what we have
                        LOG.info(f"Final detailed_metrics: degraded={list(detailed_metrics['degraded_metrics'].keys())}, restored={list(detailed_metrics['restored_metrics'].keys())}")
                        
                        # Build paths with the exact format from model.json
                        # Format degraded: audio/degraded/AUDIO_NAME_DURATION_INTENSITY_EFFECT.wav
                        degraded_basename = f"{audio_name}_{duration}_{intensity}_{effect_name}{ext}"
                        formatted_degraded_path = f"audio/degraded/{degraded_basename}"
                        
                        # Format restored: audio/restored/AUDIO_NAME_DURATION_INTENSITY_EFFECT_restored.wav
                        json_restored_path = f"audio/restored/{audio_name}_{duration}_{intensity}_{effect_name}_restored{ext}"
                        
                        # Copy the degraded audio file to the degraded folder
                        degraded_filepath = os.path.join(degraded_audio_dir, degraded_basename)
                        LOG.info(f"Saving degraded audio to: {degraded_filepath}")
                        try:
                            if os.path.exists(degraded_audio_path):
                                shutil.copy2(degraded_audio_path, degraded_filepath)
                            else:
                                LOG.warning(f"Source degraded audio not found: {degraded_audio_path}")
                        except Exception as e:
                            LOG.error(f"Error copying degraded audio file: {str(e)}")
                        
                        # Apply the same metrics restructuring logic for new entries
                        degraded_metrics = {}
                        restored_metrics = {}
                        
                        # First check if we have nested metrics structure from detailed_metrics
                        if "degraded_metrics" in detailed_metrics and isinstance(detailed_metrics["degraded_metrics"], dict):
                            # Extract all metrics and ensure they are scalar values
                            for key, value in detailed_metrics["degraded_metrics"].items():
                                if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                                    degraded_metrics[key] = float(value[0])
                                else:
                                    try:
                                        # Try to convert to Python scalar for JSON serialization
                                        degraded_metrics[key] = float(value) if isinstance(value, (int, float, np.number)) else value
                                    except (TypeError, ValueError):
                                        degraded_metrics[key] = value
                            
                            if "restored_metrics" in detailed_metrics and isinstance(detailed_metrics["restored_metrics"], dict):
                                # Same for restored metrics
                                for key, value in detailed_metrics["restored_metrics"].items():
                                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                                        restored_metrics[key] = float(value[0])
                                    else:
                                        try:
                                            # Try to convert to Python scalar for JSON serialization
                                            restored_metrics[key] = float(value) if isinstance(value, (int, float, np.number)) else value
                                        except (TypeError, ValueError):
                                            restored_metrics[key] = value
                            
                            LOG.info(f"Extracted metrics - degraded: {list(degraded_metrics.keys())}, restored: {list(restored_metrics.keys())}")
                                        
                            # Copy basic parameters to both metrics
                            for key in ['timestamp', 'steps', 'sample_rate', 'sample_size']:
                                if key in metrics:
                                    degraded_metrics[key] = metrics[key]
                                    restored_metrics[key] = metrics[key]
                        
                        LOG.info(f"Final metrics for new entry - degraded: {len(degraded_metrics)} keys, restored: {len(restored_metrics)} keys")
                        
                        # Format losses in the correct structure - all metrics, ensuring they are single values
                        # Process degraded metrics - preserve all metrics but ensure they are scalar values
                        degraded_losses = {}
                        for metric_key, metric_value in degraded_metrics.items():
                            # Convert array values to scalars
                            if isinstance(metric_value, list) and len(metric_value) > 0:
                                degraded_losses[metric_key] = metric_value[0]
                            else:
                                degraded_losses[metric_key] = metric_value
                        
                        # Process restored metrics - preserve all metrics but ensure they are scalar values
                        restored_losses = {}
                        for metric_key, metric_value in restored_metrics.items():
                            # Convert array values to scalars
                            if isinstance(metric_value, list) and len(metric_value) > 0:
                                restored_losses[metric_key] = metric_value[0]
                            else:
                                restored_losses[metric_key] = metric_value
                        
                        effect_entry = {
                            "name": current_effect,
                            "degraded": {
                                "losses": degraded_losses,  # Include ALL metrics
                                "audio": formatted_degraded_path  # Formatted path to degraded audio
                            },
                            "restored": {
                                "losses": restored_losses,  # Include ALL metrics
                                "audio": json_restored_path  # Format restored audio path
                            }
                        }
                        
                        LOG.info(f"Created entry for {current_effect} with {len(degraded_losses)} degraded metrics and {len(restored_losses)} restored metrics")
                        
                        # Add to the corresponding effects array
                        consolidated_results["data"][audio_name][duration][intensity]["effects"].append(effect_entry)
                    
                    # Clean up the JSON structure before saving
                    def remove_empty_structures(data):
                        if not isinstance(data, dict):
                            return data
                        
                        # First clean up any nested dictionaries
                        for key in list(data.keys()):
                            if isinstance(data[key], dict):
                                # Clean up nested dict
                                data[key] = remove_empty_structures(data[key])
                                # If it became empty after cleanup, remove it
                                if not data[key]:
                                    del data[key]
                            elif isinstance(data[key], list) and key == "effects":
                                # Don't delete empty effects lists, they're valid
                                pass
                            elif data[key] == {}:
                                # Remove empty dictionaries
                                del data[key]
                        return data
                    
                    # Clean up the consolidated_results structure
                    if "data" in consolidated_results:
                        consolidated_results["data"] = remove_empty_structures(consolidated_results["data"])
                        
                    # Save the consolidated results after each file to avoid loss if interrupted
                    results_filename = f"results_{model_name}.json"
                    results_path = os.path.join(output_dir, results_filename)
                    with open(results_path, "w") as f:
                        json.dump(consolidated_results, f, indent=4)
                    
                    LOG.info(f"Updated consolidated results in {results_path}")
                    audio = new_filepath
                else:
                    # Standard handling if effect not identified
                    LOG.warning(f"Could not identify effect for file {degraded_audio_path}")
                    # Use basename to avoid paths in filenames
                    audio_basename = os.path.basename(os.path.dirname(degraded_audio_path))
                    # Create a fallback name with identifiable components
                    fallback_name = f"{audio_basename}_{original_filename_without_ext}_restored{ext}"
                    new_filepath = os.path.join(restored_audio_dir, fallback_name)
                    LOG.info(f"Saving restored audio to: {new_filepath} (effect not identified)")
                    shutil.copy2(audio, new_filepath)
                    audio = new_filepath
            elif process_folder_path and output_dir:
                # Standard folder processing (without effects list)
                original_filename = os.path.basename(degraded_audio_path)
                original_filename_without_ext = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(audio)[1]
                restored_filename = f"{original_filename_without_ext}_restored{ext}"
                
                # Copy the audio file to our custom output directory
                new_filepath = os.path.join(output_dir, restored_filename)
                LOG.info(f"Copying restored audio from {audio} to {new_filepath}")
                shutil.copy2(audio, new_filepath)
                
                # Copy the metrics file to our custom output directory
                src_metrics_path = os.path.join(os.path.dirname(audio), f"{original_filename_without_ext}_metrics.json")
                dest_metrics_path = os.path.join(output_dir, f"{original_filename_without_ext}_metrics.json")
                if os.path.exists(src_metrics_path):
                    LOG.info(f"Copying metrics from {src_metrics_path} to {dest_metrics_path}")
                    shutil.copy2(src_metrics_path, dest_metrics_path)
                
                audio = new_filepath
            else:
                # Default behavior for uploaded files
                original_filename = os.path.basename(degraded_audio_path)
                original_filename_without_ext = os.path.splitext(original_filename)[0]
                ext = os.path.splitext(audio)[1]
                restored_filename = f"{original_filename_without_ext}_restored{ext}"
                
                output_dir = os.path.dirname(audio)
                new_filepath = os.path.join(output_dir, restored_filename)
                
                os.rename(audio, new_filepath)
                audio = new_filepath
            
            all_metrics.append(metrics)
            output_audios_list.append(new_filepath)
            output_spectrograms_list.extend(spectrograms)
        
        # Disable plots for batch processing
        if model_name and effects_list:
            plots = None
        elif clean_audio is not None:
            plots = create_metric_plots(all_metrics, labels) if all_metrics else None
        else:
            plots = None
        
        return output_audios_list, output_spectrograms_list, plots

    def generate_with_plots(*args):
        audios, spectrograms, plots = generate_multiple_with_plots(*args)
        first_audio = audios[0] if audios else None
        return gr.update(choices=audios, value=first_audio), first_audio, spectrograms, plots

    generate_button.click(
        fn=generate_with_plots,
        inputs=inputs,
        outputs=[output_audio_dropdown, output_audio_player, audio_spectrogram_output, metrics_plots],
        api_name="generate",
    )

    def select_audio(audio_file):
        return audio_file
    
    output_audio_dropdown.change(fn=select_audio, inputs=output_audio_dropdown, outputs=output_audio_player)

    def update_degraded_dropdown(files):
        if not files:
            return gr.update(choices=[], value=None), None
        
        choices = [(os.path.basename(f.name), f.name) for f in files]
        first_filepath = choices[0][1] if choices else None

        return gr.update(choices=choices, value=first_filepath), first_filepath

    degraded_audio_files.upload(
        fn=update_degraded_dropdown,
        inputs=[degraded_audio_files],
        outputs=[degraded_audio_dropdown, degraded_audio_player]
    )
    
    degraded_audio_dropdown.change(
        fn=lambda x: x, 
        inputs=[degraded_audio_dropdown], 
        outputs=[degraded_audio_player]
    )

def create_restoration_ui(model_config, in_model, in_model_half=True):
    global model, sample_size, sample_rate, model_type, model_half

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_type = model_config["model_type"]

    model_half = in_model_half

    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_restoration_sampling_ui(model_config)
    return ui
