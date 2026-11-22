import yaml
import os
import numpy as np

def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_numpy(i) for i in obj]
        return obj
    converted_data = convert_numpy(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(converted_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
