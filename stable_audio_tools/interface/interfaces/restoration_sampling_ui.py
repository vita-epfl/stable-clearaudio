import numpy as np
import os
import json
import gradio as gr
import logging
import torchaudio
import tempfile
import subprocess
from .restoration import generate_restoration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

LOG = logging.getLogger(__name__)
# handler
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = False


def update_degraded_dropdown(files):
        if not files:
            return gr.update(choices=[], value=None), None
        
        choices = [(os.path.basename(f.name), f.name) for f in files]
        first_filepath = choices[0][1] if choices else None

        return gr.update(choices=choices, value=first_filepath), first_filepath

def select_audio(audio_file):
    return audio_file

def get_model_name():
    # In a real application, you would fetch the model's name from your model management system
    # For now, we'll just return a placeholder
    return "placeholder_model_name"

def process_folder_files(
    input_folder,
    model_name,
    steps,
    sampler_type,
    sigma_min,
    sigma_max,
    rho,
    cfg_rescale,
    preview_every,
    file_format,
    batch_size,
    t_start,
    schedule,
    metrics_every,
    effects_list=None,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Processes all audio files in a folder based on JSON metadata files.

    For each JSON file ending in '_data.json', this function will:
    1. Load the degraded audio file.
    2. Run the restoration process on it.
    3. Save the restored audio in the same folder.
    4. Calculate metrics comparing:
        - Degraded audio vs. clean audio
        - Restored audio vs. clean audio
    5. Save a new JSON file with these metrics.
    """
    # Check if the folder exists
    if not os.path.isdir(input_folder):
        return f"Error: Folder '{input_folder}' not found."

    # Find all JSON files in the folder that end with "_data.json"
    json_files = [f for f in os.listdir(input_folder) if f.endswith("_data.json")]

    if not json_files:
        return "No JSON files ending with '_data.json' found in the folder."

    # Loop through each JSON file found
    for json_filename in progress.tqdm(json_files, desc="Processing files"):
        json_path = os.path.join(input_folder, json_filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        clean_audio_name = data.get("clean_audio_name")
        degraded_audio_name = data.get("degraded_audio_name")
        restored_audio_name = data.get("restored_audio_name")
        restoration_metrics_name = data.get("restoration_metrics_name")

        # Construct full paths for audio files
        clean_audio_path = os.path.join(input_folder, clean_audio_name)
        degraded_audio_path = os.path.join(input_folder, degraded_audio_name)

        # Check if audio files exist
        if not os.path.exists(clean_audio_path) or not os.path.exists(degraded_audio_path):
            print(f"Warning: Audio files for {json_filename} not found. Skipping.")
            continue

        # Load degraded audio
        try:
            degraded_audio, sr = torchaudio.load(degraded_audio_path)
            # If mono, convert to stereo
            if degraded_audio.shape[0] == 1:
                degraded_audio = degraded_audio.repeat(2, 1)
        except Exception as e:
            print(f"Error loading degraded audio {degraded_audio_path}: {e}")
            continue

        # Load clean audio
        try:
            clean_audio, sr = torchaudio.load(clean_audio_path)
        except Exception as e:
            print(f"Error loading clean audio {clean_audio_path}: {e}")
            continue

        print(f"Processing: {degraded_audio_name}")

        # Run the restoration process
        restored_audio_data, restored_sr, images_to_show, final_metrics = generate_restoration(
            degraded_audio=(sr, degraded_audio),
            steps=steps,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            cfg_rescale=cfg_rescale,
            preview_every=preview_every,
            metrics_every=metrics_every,
            file_format=file_format,
            clean_audio=(sr, clean_audio),
            effects_list=effects_list,
            batch_size=batch_size,
            degraded_audio_filename=degraded_audio_name,
            t_start=t_start,
            schedule=schedule,
            save_metrics=False
        )

        # Separate metrics for degraded and restored audio
        degraded_audio_metrics = {}
        restored_audio_metrics = {}

        for key, value in final_metrics.items():
            if key.startswith('degraded_'):
                degraded_audio_metrics[key.replace('degraded_', '')] = value[0] if isinstance(value, list) else value
            elif key.startswith('demo_'):
                restored_audio_metrics[key.replace('demo_', '')] = value[0] if isinstance(value, list) else value

        # Create the new JSON structure
        output_metrics = {
            "model_name": model_name,
            "degraded_audio_metrics": degraded_audio_metrics,
            "restored_audio_metrics": restored_audio_metrics
        }

        # Save the new metrics JSON file
        metrics_output_path = os.path.join(input_folder, restoration_metrics_name)
        with open(metrics_output_path, 'w') as f:
            json.dump(output_metrics, f, indent=4)

        print(f"Metrics saved to {metrics_output_path}")
        
        # Save the restored audio to the process folder
        if restored_audio_data is not None:
            restored_output_path = os.path.join(input_folder, restored_audio_name)
            try:
                torchaudio.save(restored_output_path, restored_audio_data, restored_sr)
                print(f"Restored audio saved to {restored_output_path}")
            except Exception as e:
                print(f"Error saving restored audio: {e}")

    return "Processing complete. Metrics and restored audio saved in the folder."


def create_metric_plots(metrics_data_list, labels):
    try:
        if not metrics_data_list:
            LOG.warning("No metrics data provided to plot.")
            return []

        # Determine the set of all metric keys across all files
        EXCLUDED_KEYS = ["steps", "timestamp", "sample_rate", "sample_size", "generation_params"]
        
        # Debug: Print all available metrics
        LOG.debug(f"Available metrics in data: {[k for metrics_data in metrics_data_list for k in metrics_data.keys()]}")
        
        # Fix: Don't require metrics to be a list, just exclude the specific non-plottable keys
        plottable_metrics = sorted(list(set(
            k for metrics_data in metrics_data_list 
            for k in metrics_data.keys() 
            if k not in EXCLUDED_KEYS
        )))
        
        # Exclude all degraded_ metrics from plotting
        filtered_metric_keys = [k for k in plottable_metrics if not k.startswith('degraded_')]
        
        # Now separate the remaining metrics into restoration and main categories
        restoration_metrics_keys = sorted([k for k in filtered_metric_keys if k.startswith('restoration_success_')])
        # Also filter out clean_* metrics which are now single values, not arrays
        # Additionally filter out the step_0 graph as requested
        main_metrics_keys = sorted([k for k in filtered_metric_keys 
                               if k not in restoration_metrics_keys and 
                               not k.startswith('clean_') and 
                               k != 'step_0'])

        if not main_metrics_keys and not restoration_metrics_keys:
            LOG.warning("No metrics found to plot.")
            return []

        # Create individual plots instead of one combined plot
        individual_plots = []
        
        # Helper function to create a single metric plot
        def create_single_plot(metric_name, is_restoration=False):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if is_restoration:
                # Plot restoration success metrics as bar chart
                for i, metrics_data in enumerate(metrics_data_list):
                    if metrics_data and metric_name in metrics_data:
                        success_rate = metrics_data[metric_name]
                        label = labels[i] if i < len(labels) else f'Audio {i+1}'
                        ax.bar(label, success_rate, alpha=0.7)
                ax.set_ylabel('Success Rate')
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.set_ylim(0, 1)
            else:
                # Plot regular metrics as line plots
                for i, metrics_data in enumerate(metrics_data_list):
                    if metrics_data and metric_name in metrics_data and "steps" in metrics_data and metrics_data["steps"]:
                        steps = metrics_data["steps"]
                        
                        if isinstance(metrics_data[metric_name], list):
                            values = metrics_data[metric_name]
                            
                            # Check if we have step_0 data to include
                            if "step_0" in metrics_data and metric_name in metrics_data["step_0"]:
                                if 0 not in steps:
                                    steps = [0] + steps
                                    values = [metrics_data["step_0"][metric_name]] + values
                            
                            label = labels[i] if i < len(labels) else f'Audio {i+1}'
                            ax.plot(steps, values, '-o', label=label, markersize=4)
                            
                            # Add clean reference line if available
                            base_metric = metric_name
                            if base_metric.startswith('demo_'):
                                base_metric = base_metric[5:]
                            
                            if len(values) > 0:
                                clean_metric_key = f"clean_{base_metric}"
                                if clean_metric_key in metrics_data:
                                    try:
                                        clean_value = metrics_data[clean_metric_key]
                                        if isinstance(clean_value, (int, float)):
                                            ax.axhline(y=clean_value, color='red', linestyle='--', alpha=0.7, label=f'Clean Reference ({clean_value:.3f})')
                                    except Exception as e:
                                        LOG.debug(f"Could not plot clean reference for {clean_metric_key}: {e}")
                
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to numpy array
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)
            
            from PIL import Image
            img = Image.open(buf)
            return np.array(img)

        # Create plots for main metrics only
        for metric_name in main_metrics_keys:
            individual_plots.append(create_single_plot(metric_name))
        
        return individual_plots
    except Exception as e:
        LOG.error(f"Error creating metric plots: {str(e)}", exc_info=True)
        return []

def generate_multiple_with_plots(steps, t_start, schedule, preview_every, metrics_every, sampler_type, 
   degraded_audio_files, clean_audio, process_folder_path=None, model_name=None, 
   effects_list=None, sigma_min=None, sigma_max=None, rho=None, cfg_rescale=None, file_format=None, batch_size=1, progress=gr.Progress(track_tqdm=True)):
    """
    Generate multiple audio files from a folder of degraded audio files.

    Args:
        steps (int): Number of steps to generate audio.
        preview_every (int): Preview every N steps.
        metrics_every (int): Compute metrics every N steps.
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
    
    if process_folder_path and os.path.isdir(process_folder_path):
        process_folder_files(
            process_folder_path, model_name, steps, sampler_type, sigma_min, sigma_max, rho, cfg_rescale,
            preview_every, file_format, batch_size, t_start, schedule, metrics_every=steps, effects_list=effects_list,
            progress=progress
        )
        # Return empty lists instead of None to allow proper unpacking
        return [], [], []
        
    elif not degraded_audio_files:
        raise gr.Error("No degraded audio files provided.")

    labels = []
    
    # If we're uploading individual files
    if degraded_audio_files and not process_folder_path:
        for f in degraded_audio_files:
            filename = os.path.basename(f.name)
            filename_no_ext = os.path.splitext(filename)[0]
            labels.append(filename_no_ext)

    # Create a temporary directory for the output files
    temp_dir = tempfile.mkdtemp()

    all_metrics = []
    output_audios_list = []
    output_spectrograms_list = []
    labels = []
    # Configure environment variables for batch processing
    # Disable creation of temporary directories
    os.environ["STABLE_AUDIO_NO_DATE_FOLDER"] = "1"
    os.environ["STABLE_AUDIO_BATCH_PROCESSING"] = "1"
    
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

        audio_data, sr, spectrograms, metrics = generate_restoration(
            steps=steps,
            preview_every=preview_every,
            metrics_every=metrics_every,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            cfg_rescale=cfg_rescale,
            file_format=file_format,
            degraded_audio=degraded_audio_input,
            clean_audio=clean_audio,
            effects_list=effects_list,
            batch_size=batch_size,
            degraded_audio_filename=degraded_audio_path,
            t_start=t_start,
            schedule=schedule,
        )
        
        # Define output filenames
        base_filename = os.path.splitext(os.path.basename(degraded_audio_path))[0]
        output_wav = os.path.join(temp_dir, f"{base_filename}_restored.wav")
        filename_extension = file_format.split(" ")[0].lower() if file_format else "wav"
        output_filename = os.path.join(temp_dir, f"{base_filename}_restored.{filename_extension}")

        # Save the restored audio
        if audio_data is not None:
            try:
                torchaudio.save(output_wav, audio_data, sr)
                LOG.debug(f"Saved WAV file to {output_wav}")
            except Exception as e:
                LOG.error(f"Error saving WAV file {output_wav}: {e}")
                continue

            # If file_format is other than wav, convert to other file format
            if file_format and file_format != "wav":
                cmd = ""
                if file_format == "m4a aac_he_v2 32k":
                    cmd = f'ffmpeg -i "{output_wav}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y "{output_filename}"'
                elif file_format == "m4a aac_he_v2 64k":
                    cmd = f'ffmpeg -i "{output_wav}" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y "{output_filename}"'
                elif file_format == "flac":
                    cmd = f'ffmpeg -i "{output_wav}" -y "{output_filename}"'
                elif file_format == "mp3 320k":
                    cmd = f'ffmpeg -i "{output_wav}" -b:a 320k -y "{output_filename}"'
                elif file_format == "mp3 v0":
                    cmd = f'ffmpeg -i "{output_wav}" -q:a 0 -y "{output_filename}"'
                elif file_format == "mp3 128k":
                    cmd = f'ffmpeg -i "{output_wav}" -b:a 128k -y "{output_filename}"'
                
                if cmd:
                    cmd += " -loglevel error"  # make output less verbose in the cmd window
                    try:
                        subprocess.run(cmd, shell=True, check=True)
                        LOG.debug(f"Converted to {file_format} format: {output_filename}")
                    except Exception as e:
                        LOG.error(f"Error converting to {file_format}: {e}")
                        new_filepath = output_wav # Fallback to wav
                new_filepath = output_filename
            else:
                new_filepath = output_wav
        else:
            new_filepath = None
        
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

def generate_with_plots(*args, progress=gr.Progress(track_tqdm=True)):
    audios, spectrograms, plots = generate_multiple_with_plots(*args, progress=progress)
    first_audio = audios[0] if audios else None
    # Return all plots for Gallery navigation
    return gr.update(choices=audios, value=first_audio), first_audio, spectrograms, plots, first_audio

def navigate_plots(plots_list, current_index, direction):
    """Navigate between metric plots"""
    if not plots_list or len(plots_list) == 0:
        return None, "0 / 0", 0
    
    if direction == "prev":
        new_index = (current_index - 1) % len(plots_list)
    else:  # direction == "next"
        new_index = (current_index + 1) % len(plots_list)
    
    plot_info = f"{new_index + 1} / {len(plots_list)}"
    return plots_list[new_index], plot_info, new_index

def create_uncond_restoration_sampling_ui():
    global model, sample_rate, model_type, model_half

    diffusion_objective = model.diffusion_objective

    is_rf = diffusion_objective == "rectified_flow"
    if is_rf:
        LOG.info("Rectified flow model detected")
    else:
        LOG.info("Non rectified flow model detected")
    
    with gr.Row():                
        generate_button = gr.Button("Generate", variant="primary", scale=1)
        
    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2/3):
                    # Steps slider
                    default_steps = 30
                    steps_slider = gr.Slider(
                        minimum=0, maximum=500, step=1, value=default_steps, label="Steps"
                    )

            
                with gr.Column(scale=1/3):
                    batch_size_slider = gr.Slider(
                        minimum=1, maximum=16, step=1, value=1, label="Batch size"
                    )

            with gr.Accordion("Sampler params", open=False):
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
                        sampler_types = ["euler", "rk4"]
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
                        value=10,
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

            effects_list = gr.Textbox(visible=False)

            open_cold_diffusion = model.diffusion_objective == "cold_diffusion"
                
            with gr.Accordion("Cold diffusion parameters", open=open_cold_diffusion):

                with gr.Row():
                    with gr.Column(scale=1):
                        t_start_slider = gr.Slider(
                            label="T start",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=1.0,
                        )
                    with gr.Column(scale=1):
                        schedule_dropdown = gr.Dropdown(
                            [
                                "linear",
                                "cosine",
                            "squaredcos_cap0",
                            "squaredcos_cap1",
                            "sigmoid",
                        ],
                        label="Schedule",
                        value="linear",
                    )

                from pathlib import Path
                _presets_dir = Path(__file__).resolve().parent.parent.parent / "configs" / "dataset_configs" / "low_quality_effect"
                if _presets_dir.is_dir():
                    _preset_files = sorted([p.stem for p in _presets_dir.glob("*.yaml")])
                else:
                    _preset_files = []
                preset_selector = gr.CheckboxGroup(
                    choices=_preset_files,
                    label="Low-quality effect presets (mandatory for cold diffusion models or for batch processing. Should be the same than the ones used for training)",
                )

                # Whenever presets change, update effects_list textbox as comma-separated string
                preset_selector.change(lambda s: ",".join(s) if s else "", inputs=[preset_selector], outputs=[effects_list])
            
            with gr.Accordion("Batch Processing Options", open=False):                    
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

            inputs = [
                steps_slider,
                t_start_slider,
                schedule_dropdown,
                preview_every_slider,
                metrics_every_slider,
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
                batch_size_slider,
            ]            

        with gr.Column():
            output_audio_dropdown = gr.Dropdown(label="Select generated audio")
            output_audio_player = gr.Audio(label="Audio player")
            download_audio_file = gr.File(label="Download output audio")
            audio_spectrogram_output = gr.Gallery(label="Output spectrograms", show_label=True, elem_id="spectrogram_gallery", columns=2, height=400)
            
            # Add metrics display section
            with gr.Accordion("Restoration Metrics", open=True):
                metrics_plots = gr.Gallery(label="Metrics Plots", show_label=True, elem_id="metrics_gallery", columns=1, rows=1, height=600)
            
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
        outputs=[output_audio_dropdown, output_audio_player, audio_spectrogram_output, metrics_plots, download_audio_file],
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
    print("Creating cond restoration sampling ui")
    global model, sample_rate, model_type, model_half

    diffusion_objective = getattr(model, 'diffusion_objective', None)

    is_rf = diffusion_objective == "rectified_flow"
    
    with gr.Row():                
        generate_button = gr.Button("Generate", variant="primary", scale=1)
        
    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2/3):
                    # Steps slider
                    default_steps = 30
                    steps_slider = gr.Slider(
                        minimum=1, maximum=500, step=1, value=default_steps, label="Steps"
                    )

                
                with gr.Column(scale=1/3):
                    batch_size_slider = gr.Slider(
                        minimum=1, maximum=16, step=1, value=1, label="Batch size"
                    )

            with gr.Accordion("Sampler params", open=False):
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
                        value=10,
                        label="Compute Metrics Every N Steps"
                    )
            
            # Hidden cold diffusion parameters for conditional models (not used but needed for unified inputs)
            t_start_slider = gr.Slider(
                label="T start",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=1.0,
                visible=False,
            )
            schedule_dropdown = gr.Dropdown(
                [
                    "linear",
                    "cosine",
                    "squaredcos_cap0",
                    "squaredcos_cap1",
                    "sigmoid",
                ],
                label="Schedule",
                value="linear",
                visible=False,
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
                t_start_slider,
                schedule_dropdown,
                preview_every_slider,
                metrics_every_slider,
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
                batch_size_slider,
            ]            

        with gr.Column():
            output_audio_dropdown = gr.Dropdown(label="Select generated audio")
            output_audio_player = gr.Audio(label="Audio player")
            download_audio_file = gr.File(label="Download output audio")
            audio_spectrogram_output = gr.Gallery(label="Output spectrograms", show_label=True, elem_id="spectrogram_gallery", columns=2, height=400)
            
            # Add metrics display section
            with gr.Accordion("Restoration Metrics", open=True):
                metrics_plots = gr.Gallery(label="Metrics Plots", show_label=True, elem_id="metrics_gallery", columns=1, rows=1, height=600)
            
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
        outputs=[output_audio_dropdown, output_audio_player, audio_spectrogram_output, metrics_plots, download_audio_file],
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
            if model_type in ["diffusion_cond_restoration", "diffusion_cond"]:
                create_cond_restoration_sampling_ui()
            elif model_type in ["diffusion_uncond_restoration", "diffusion_uncond"]:
                create_uncond_restoration_sampling_ui()
    return ui
