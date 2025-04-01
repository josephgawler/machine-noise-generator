import os
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse

def trim_middle_5_seconds(audio_path, output_path):
    """Load audio, trim to middle 5 seconds, and save"""
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate start and end points for middle 5 seconds
    total_samples = len(y)
    mid_point = total_samples // 2
    samples_per_second = sr
    offset = int(2.5 * samples_per_second)  # 2.5 seconds on each side of midpoint
    
    # Extract the middle 5 seconds
    start_idx = mid_point - offset
    end_idx = mid_point + offset
    
    # Handle cases where audio might be shorter than expected
    start_idx = max(0, start_idx)
    end_idx = min(total_samples, end_idx)
    
    trimmed_audio = y[start_idx:end_idx]
    
    # Save the trimmed audio
    sf.write(output_path, trimmed_audio, sr)

def process_machine_data(machine_type):
    """
    Process audio files for a specific machine type.
    
    Args:
        machine_type (str): Type of machine ('fan', 'pump', 'slider', or 'valve')
    """
    # Fixed number of samples per ID
    SAMPLES_PER_ID = 30
    
    # Set up input and output paths
    base_dir = r"C:\Users\julia\DS4440\machine-noise-generator\data"
    input_dir = os.path.join(base_dir, machine_type)
    output_dir = os.path.join(base_dir, f"trimmed_{machine_type}")
    
    print(f"Processing {machine_type} data...")
    
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create condition subdirectories
    for condition in ["normal", "abnormal"]:
        Path(os.path.join(output_dir, condition)).mkdir(parents=True, exist_ok=True)
    
    # Process each id folder
    id_folders = [folder for folder in os.listdir(input_dir) if folder.startswith("id_")]
    
    for id_folder in id_folders:
        id_path = os.path.join(input_dir, id_folder)
        
        # Process normal and abnormal folders
        for condition in ["normal", "abnormal"]:
            condition_path = os.path.join(id_path, condition)
            
            # Check if this path exists
            if os.path.exists(condition_path) and os.path.isdir(condition_path):
                # Create the output directory for this id and condition
                output_subdir = os.path.join(output_dir, condition, id_folder)
                os.makedirs(output_subdir, exist_ok=True)
                
                # List all wav files
                wav_files = [f for f in os.listdir(condition_path) if f.endswith('.wav')]
                
                # Select exactly 30 files (or all if fewer)
                num_to_select = min(SAMPLES_PER_ID, len(wav_files))
                selected_files = random.sample(wav_files, num_to_select)
                
                print(f"  Selecting {num_to_select} {condition} files from {machine_type}/{id_folder}")
                
                # Process each selected file
                for i, filename in enumerate(selected_files):
                    input_path = os.path.join(condition_path, filename)
                    output_path = os.path.join(output_subdir, f"trimmed_{filename}")
                    
                    # Trim and save
                    try:
                        trim_middle_5_seconds(input_path, output_path)
                        # Print progress every 10 files
                        if (i + 1) % 10 == 0:
                            print(f"    Processed {i + 1}/{num_to_select} files")
                    except Exception as e:
                        print(f"    Error processing {input_path}: {e}")
                
                print(f"  Completed {id_folder}/{condition}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract and trim machine sound samples.')
    parser.add_argument('--machine', type=str, choices=['fan', 'pump', 'slider', 'valve', 'all'],
                        default='fan', help='Machine type to process (default: fan)')
    
    args = parser.parse_args()
    
    # List of all machine types
    all_machines = ['fan', 'pump', 'slider', 'valve']
    
    # Process based on the selected machine type
    if args.machine == 'all':
        for machine in all_machines:
            process_machine_data(machine)
    else:
        process_machine_data(args.machine)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()