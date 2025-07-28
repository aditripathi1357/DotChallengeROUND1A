#!/usr/bin/env python3
"""Script to pre-download and verify ML models"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

def get_directory_size(directory: str) -> float:
    """Calculate total size of directory in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception as e:
        print(f"Error calculating directory size: {e}")
        return 0

def check_model_constraints(cache_dir: str, size_limit_mb: int = 200) -> bool:
    """Check if model size is within constraints"""
    if not os.path.exists(cache_dir):
        return True
    
    size_mb = get_directory_size(cache_dir)
    print(f"Model cache size: {size_mb:.2f} MB")
    return size_mb <= size_limit_mb

def download_model(model_name: str, cache_dir: str = "./models"):
    """Download and cache model locally"""
    print(f"Downloading model: {model_name}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        print(f"✅ Successfully downloaded: {model_name}")
        
        # Check size constraints
        if check_model_constraints(cache_dir):
            print(f"✅ Model size within 200MB limit")
        else:
            print(f"❌ Model size exceeds 200MB limit")
            
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False

def main():
    """Download recommended models"""
    models_to_try = [
        "prajjwal1/bert-tiny",      # ~17MB - Recommended
        "distilbert-base-uncased",  # ~67MB - Alternative
        "microsoft/DialoGPT-small"  # ~117MB - Backup
    ]
    
    selected_model = None
    
    for model_name in models_to_try:
        print(f"\n--- Testing {model_name} ---")
        success = download_model(model_name)
        if success and check_model_constraints("./models"):
            print(f"Model {model_name} ready to use!")
            selected_model = model_name
            break
        elif success:
            print(f"Model {model_name} downloaded but exceeds size limit")
    
    if not selected_model:
        print("❌ No suitable model found within size constraints")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model info
    model_info = {
        "selected_model": selected_model,
        "cache_dir": "./models",
        "size_limit_mb": 200,
        "status": "ready"
    }
    
    try:
        with open("models/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        print(f"\n✅ Model info saved to models/model_info.json")
    except Exception as e:
        print(f"❌ Error saving model info: {e}")

if __name__ == "__main__":
    main()