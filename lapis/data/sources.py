import os
import glob


def find_data_files(data_dir, pattern="*.jsonl *.txt *.json *.csv"):
    """Find data files in the data directory."""
    files = []
    for p in pattern.split():
        files.extend(glob.glob(os.path.join(data_dir, p)))
    return sorted(files)


def list_datasets(data_dir="data"):
    """List available datasets."""
    datasets = {}
    if not os.path.exists(data_dir):
        return datasets
    
    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path):
            files = find_data_files(entry_path)
            if files:
                datasets[entry] = files
    
    return datasets