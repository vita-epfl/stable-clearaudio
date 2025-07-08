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

from .restoration import generate_restoration


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

def get_metrics_spectrograms_batch_processing()(audio, degraded_audio_path, effects_files, model_name, output_dir, degraded_audio_dir, restored_audio_dir, batch_processing_dir, consolidated_results, metrics):
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

    return audio, metrics, new_filepath