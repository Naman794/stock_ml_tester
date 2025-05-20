# fetch/csv_writer.py

import pandas as pd
import os

def save_to_csv(data, filename, folder="data"):
    if not data:
        return
    df = pd.DataFrame(data)
    os.makedirs(folder, exist_ok=True)
    df.to_csv(f"{folder}/{filename}", index=False)
