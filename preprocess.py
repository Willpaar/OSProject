import pandas as pd
import re
import os

import numpy as np
from sklearn.model_selection import train_test_split
import os
import feature_extraction

def preprocess_training_data(filepath=None, 
                             val_split=0.2,
                             save_processed=True,
                             use_spliced=False,
                             spliced_dir="data/spliced",
                             max_chunks=None):
    """
    Loads CSV or spliced chunks, extracts features, splits into train/val, and returns dict.
    """

    # -----------------------------
    # 1. Load data
    # -----------------------------
    if use_spliced:
        print("Loading spliced chunk files...")
        frames = []
        count = 0

        for file in os.listdir(spliced_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(spliced_dir, file))
                frames.append(df)
                count += 1
                if max_chunks and count >= max_chunks:
                    break

        df = pd.concat(frames, ignore_index=True)
    else:
        print(f"Loading main CSV file: {filepath}")
        df = pd.read_csv(filepath)

    print("Loaded data:", df.shape)

    # -----------------------------
    # 2. Must include a label column
    # -----------------------------
    if "Classification" not in df.columns:
        raise ValueError("ERROR: CSV does not contain 'Classification' column")

    y = df["Classification"].values

    # -----------------------------
    # 3. Extract numerical features
    # -----------------------------
    print("Extracting features...")
    try:
        X, df_with_features = feature_extraction.extract_features(df)
    except Exception as e:
        print("Feature extraction failed:", e)
        return None

    # Convert to numpy
    X = np.array(X)
    y = np.array(y)

    # -----------------------------
    # 4. Train/Val split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, shuffle=True
    )

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # -----------------------------
    # 5. Save outputs
    # -----------------------------
    if save_processed:
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/X_val.npy", X_val)
        np.save("data/processed/y_val.npy", y_val)

        print("Processed data saved to data/processed/")

    # -----------------------------
    # 6. Return structure required by training script
    # -----------------------------
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val
    }

CNN_FEATURE_COLUMNS = [
    'No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length',
    'Source Port', 'Destination Port', 'TCP Flags', 'Sequence Number',
    'Acknowledgment Number', 'Window Size', 'Payload Length',
    'TSval', 'TSecr', 'NodeWeight', 'EdgeWeight'
]

def preprocess_validation_data(csv_path):
    df = pd.read_csv(csv_path)
    print("Available columns in CSV:", df.columns)

    # Fill empty/missing values with 0
    df = df.fillna(0)

    # Keep only the columns the CNN expects
    missing_cols = [col for col in CNN_FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"Warning: missing columns in CSV, filling with 0: {missing_cols}")
        for col in missing_cols:
            df[col] = 0

    df = df[CNN_FEATURE_COLUMNS + ['Classification']]  # Keep order + label

    # Save processed CSV
    output_dir = r"C:\Users\will\OneDrive\Programs\Python\OS\Senior-Capstone-Project\processed_data"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    output_csv = os.path.join(output_dir, f"{name}_processed.csv")
    df.to_csv(output_csv, index=False)
    print(f"Data successfully structured and saved as '{output_csv}'!")

    # Extract features and labels
    y = df["Classification"]
    X = df.drop(columns=["Classification"])

    # Ensure numeric type
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X, y, df