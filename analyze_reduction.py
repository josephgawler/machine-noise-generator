import os
import math

def get_dir_stats(directory):
    """Get file count, folder count, and total size of a directory."""
    total_size = 0
    file_count = 0
    folder_count = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        folder_count += len(dirs)
        file_count += len(files)
        
        # Sum file sizes
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    
    return file_count, folder_count, total_size

def format_size(size_bytes):
    """Format bytes into a readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

# Paths
original_dir = r"C:\Users\julia\DS4440\machine-noise-generator\data\fan"
trimmed_dir = r"C:\Users\julia\DS4440\machine-noise-generator\data\trimmed_fan"

# Get stats for both directories
orig_files, orig_folders, orig_size = get_dir_stats(original_dir)
trim_files, trim_folders, trim_size = get_dir_stats(trimmed_dir)

# Print comparison
print("Directory Comparison:")
print("-" * 50)
print(f"{'Parameter':<15} {'Original':<20} {'Trimmed':<20}")
print("-" * 50)
print(f"{'Files:':<15} {orig_files:<20,d} {trim_files:<20,d}")
print(f"{'Folders:':<15} {orig_folders:<20,d} {trim_folders:<20,d}")
print(f"{'Total Size:':<15} {format_size(orig_size):<20} {format_size(trim_size):<20}")
print(f"{'Size Ratio:':<15} {'100%':<20} {(trim_size/orig_size*100):.2f}%")
print("-" * 50)

# Get per-folder details for the trimmed directory
print("\nTrimmed Data Details:")
print("-" * 50)
conditions = ["normal", "abnormal"]

for condition in conditions:
    cond_path = os.path.join(trimmed_dir, condition)
    if os.path.exists(cond_path):
        print(f"\n{condition.upper()} samples:")
        
        # Count files per ID folder
        for folder in os.listdir(cond_path):
            folder_path = os.path.join(cond_path, folder)
            if os.path.isdir(folder_path):
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                print(f"  {folder}: {len(files)} files")