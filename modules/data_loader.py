# modules/data_loader.py
import pandas as pd
import os

def load_dataset(file_path):
    """Safely loads the csv dataset from the path specified."""
    if not os.path.exists(file_path):
        print(f"❌ Error: The file layout is missing '{file_path}'. Please check your data folder.")
        return None
    
    df = pd.read_csv(file_path)
    print(f"✅ Data successfully loaded. Total Rows: {df.shape[0]}, Total Features: {df.shape[1]}")
    return df