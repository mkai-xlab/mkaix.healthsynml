# Utility functions for validating file streams and checking extensions
import os
import hashlib
from tqdm import tqdm

def _calculate_md5(file_path):
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except IOError as e:
        print(f"Could not read file {file_path}: {e}")
        return None
    return hash_md5.hexdigest()

def find_and_remove_duplicates(data_dir):
    """
    Finds and removes duplicate images in a directory based on MD5 hash.
    
    Args:
        data_dir (str): The root directory of the dataset to scan.
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        return

    hashes = {}
    duplicates_found = 0
    
    print(f"\nScanning for duplicate images in: {data_dir}")
    
    # Build a list of all file paths to process
    all_files = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(data_dir)
        for filename in filenames
        if filename.lower().endswith(('png', 'jpg', 'jpeg')) # Process only image files
    ]

    if not all_files:
        print("No image files found to process.")
        return

    for file_path in tqdm(all_files, desc="Hashing files", unit="file"):
        file_hash = _calculate_md5(file_path)
        if file_hash is None:
            continue
            
        if file_hash in hashes:
            print(f"  - Duplicate found: '{os.path.basename(file_path)}' is a copy of '{os.path.basename(hashes[file_hash])}'. Deleting.")
            try:
                os.remove(file_path)
                duplicates_found += 1
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
        else:
            hashes[file_hash] = file_path

    if duplicates_found == 0:
        print(f"Scan complete. No duplicate images were found in '{data_dir}'.")
    else:
        print(f"\nScan complete. Removed {duplicates_found} duplicate image(s) from '{data_dir}'.")