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

from .restoration import generate_restoration
from .batch_processing import get_metrics_spectrograms_batch_processing


def update_degraded_dropdown(files):
        if not files:
            return gr.update(choices=[], value=None), None
        
        choices = [(os.path.basename(f.name), f.name) for f in files]
        first_filepath = choices[0][1] if choices else None

        return gr.update(choices=choices, value=first_filepath), first_filepath

def select_audio(audio_file):
    return audio_file

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


def create_metric_plots(metrics_data_list, labels):
    try:
        if not metrics_data_list:
            LOG.warning("No metrics data provided to plot.")
            return None

        # Determine the set of all metric keys across all files
        all_metric_keys = set()
        for metrics_data in metrics_data_list:
            if metrics_data:
                all_metric_keys.update(k for k in metrics_data.keys() if k not in ["steps", "timestamp", "sample_rate", "sample_size", "generation_params"])

        # Exclude all degraded_ metrics from plotting
        filtered_metric_keys = [k for k in all_metric_keys if not k.startswith('degraded_')]
        
        # Now separate the remaining metrics into restoration and main categories
        restoration_metrics_keys = sorted([k for k in filtered_metric_keys if k.startswith('restoration_success_')])
        main_metrics_keys = sorted([k for k in filtered_metric_keys if k not in restoration_metrics_keys])

        if not main_metrics_keys and not restoration_metrics_keys:
            LOG.warning("No metrics found to plot.")
            return None

        n_plots = len(main_metrics_keys)
        num_restoration_plots = 0
        if restoration_metrics_keys:
            for metrics_data in metrics_data_list:
                if metrics_data and any(k in metrics_data for k in restoration_metrics_keys):
                    num_restoration_plots += 1
        
        n_plots += num_restoration_plots

        if n_plots == 0:
            return None
        
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create figure with extra space for legend
        fig = plt.figure(figsize=(15, 4 * n_rows + 1))
        gs = plt.GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.2])
        axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(n_plots)]

        plot_idx = 0

        # Plot main metrics
        for metric_name in main_metrics_keys:
            ax = axes[plot_idx]
            for i, metrics_data in enumerate(metrics_data_list):
                if metrics_data and metric_name in metrics_data and "steps" in metrics_data and metrics_data["steps"]:
                    steps = metrics_data["steps"]
                    # Vérifier si l'index i est valide dans la liste labels
                    label = labels[i] if i < len(labels) else f'Audio {i+1}'
                    ax.plot(steps, metrics_data[metric_name], '-o', label=label, markersize=4)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.set_title(f'{metric_name} over steps')
            ax.grid(True)
            ax.legend(loc='best')
            plot_idx += 1
        
        # Plot restoration success metrics, one plot per file
        if restoration_metrics_keys:
            # Create a single legend for all restoration plots
            legend_handles = []
            legend_labels = []
            
            for i, metrics_data in enumerate(metrics_data_list):
                file_restoration_metrics = [k for k in restoration_metrics_keys if k in metrics_data] if metrics_data else []

                if file_restoration_metrics and "steps" in metrics_data and metrics_data["steps"]:
                    ax = axes[plot_idx]
                    steps = metrics_data["steps"]
                    for metric_name in file_restoration_metrics:
                        label = metric_name.replace('restoration_success_', '')
                        line = ax.plot(steps, metrics_data[metric_name], '-o', label=label, markersize=4)[0]
                        if i == 0:  # Only collect handles and labels from first plot
                            legend_handles.append(line)
                            legend_labels.append(label)
                    
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Value')
                    ax.set_title(f'NRS for {labels[i]}')
                    ax.grid(True)
                    plot_idx += 1

            # Add single legend at the bottom
            if legend_handles:
                legend_ax = fig.add_subplot(gs[-1, :])
                legend_ax.axis('off')
                legend_ax.set_title('Legend for Normalized Restoration Success (NRS)')
                legend_ax.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_handles))

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(pad=3.0)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        
        from PIL import Image
        img = Image.open(buf)
        return np.array(img)
    except Exception as e:
        LOG.error(f"Error creating metric plots: {str(e)}", exc_info=True)
        return None

def generate_multiple_with_plots(steps, preview_every, metrics_every, seed, sampler_type, 
degraded_audio_files, clean_audio, process_folder_path=None, model_name=None, 
effects_list=None, sigma_min=None, sigma_max=None, rho=None, cfg_rescale=None, file_format=None):
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
    
    # If we are processing from a folder
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
            effects_list=effects_list,
            batch_size=1,
            degraded_audio_filename=degraded_audio_path  # Pass the filename here
        )
        
        # Process based on whether we're using effects list or standard folder processing
        if model_name and effects_list and consolidated_results:
            metrics, new_filepath, spectrograms = get_metrics_spectrograms_batch_processing()(
                audio,
                degraded_audio_path,
                effects_files,
                model_name,
                output_dir,
                degraded_audio_dir,
                restored_audio_dir,
                batch_processing_dir,
                consolidated_results,
                metrics
            )
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

def create_uncond_restoration_sampling_ui():
    global model, sample_rate, model_type, model_half
    
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

                # New row: preset selector for low-quality degradations
                with gr.Row():
                    # Dynamically list available preset YAMLs
                    import os
                    from pathlib import Path
                    _presets_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "dataset_configs" / "low_quality_effect"
                    if _presets_dir.is_dir():
                        _preset_files = sorted([p.stem for p in _presets_dir.glob("*.yaml")])
                    else:
                        _preset_files = []
                    preset_selector = gr.CheckboxGroup(
                        choices=_preset_files,
                        label="Low-quality effect presets (mandatory for cold diffusion models)",
                    )

                    # Whenever presets change, update effects_list textbox as comma-separated string
                    def _update_effects_list(selected):
                        return ",".join(selected) if selected else ""
                    preset_selector.change(_update_effects_list, inputs=[preset_selector], outputs=[effects_list])

            inputs = [
                steps_slider,
                preview_every_slider,
                metrics_every_slider,
                seed_textbox,
                sampler_type_dropdown,
                degraded_audio_files,
                clean_audio,
                process_folder_path,
                model_name,
                effects_list,
                file_format_dropdown,
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

    generate_button.click(
        fn=generate_with_plots,
        inputs=inputs,
        outputs=[output_audio_dropdown, output_audio_player, audio_spectrogram_output, metrics_plots],
        api_name="generate",
    )
    
    output_audio_dropdown.change(fn=select_audio, inputs=output_audio_dropdown, outputs=output_audio_player)

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


def create_cond_restoration_sampling_ui():
    global model, sample_rate, model_type, model_half

    diffusion_objective = getattr(model, 'diffusion_objective', None)

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
                degraded_audio_files,
                clean_audio,
                process_folder_path,
                model_name,
                effects_list,
                sigma_min_slider,
                sigma_max_slider,
                rho_slider,
                cfg_rescale_slider,
                file_format_dropdown,
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

    generate_button.click(
        fn=generate_with_plots,
        inputs=inputs,
        outputs=[output_audio_dropdown, output_audio_player, audio_spectrogram_output, metrics_plots],
        api_name="generate",
    )
    
    output_audio_dropdown.change(fn=select_audio, inputs=output_audio_dropdown, outputs=output_audio_player)

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

    from . import restoration as _restoration_module
    _restoration_module.model = in_model
    _restoration_module.sample_size = sample_size
    _restoration_module.sample_rate = sample_rate
    _restoration_module.model_type = model_type if 'model_type' in globals() else None
    _restoration_module.model_half = in_model_half

    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            if model_type == "diffusion_cond_restoration":
                create_cond_restoration_sampling_ui()
            elif model_type == "cold_diffusion_uncond_restoration":
                create_uncond_restoration_sampling_ui()
    return ui
