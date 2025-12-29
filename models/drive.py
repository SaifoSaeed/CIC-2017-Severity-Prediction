import os
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from lib import WeightedLR, XGBoost, OrdinalNN, map_labels_to_severity, COST_MATRIX

CWD = str(os.getcwd())
# Default paths (can be overridden by args)
OUTPUT_DIR = r"./results" 
INPUT_FILE  = r"proc_dfs/ben_mal.csv" # Adjusted relative path

# Parse command line arguments.
def ParseArgs():
    parser = argparse.ArgumentParser(description="Training Driver for CIC-IDS-2017 Models.")
    
    parser.add_argument("-l", "--logistic", action="store_true", help="train weighted logistic regression model")
    parser.add_argument("-x", "--xgb", action="store_true", help="train XG-boost model")
    parser.add_argument("-n", "--neural", action="store_true", help="train neural network model")
    parser.add_argument("-e", "--ensemble", action="store_true", help="train an ensemble model using xgb and the neural network models")
    parser.add_argument("-o", "--output_dir", type=str, help="choose output directory")
    
    return parser.parse_args()

# Helper to visualize and save results

def SaveResults(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Results ---")
    print(classification_report(y_true, y_pred))
    
    # Calculate Cost
    cm = confusion_matrix(y_true, y_pred)
    total_cost = 0
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i < 4 and j < 4: # Safety check for matrix bounds
                total_cost += cm[i, j] * COST_MATRIX[i, j]
                
    print(f"Total Weighted Cost: {total_cost}")

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name}\nTotal Penalty Score: {total_cost}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

# Wrapper to handle data loading and splitting once
def GetSplits():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File not found at {INPUT_FILE}")
        return None, None, None, None

    print("Mapping labels...")
    df['Severity'] = map_labels_to_severity(df['Label'])

    if df['Severity'].isna().any():
        # print the specific labels that failed to map
        unmapped_labels = df[df['Severity'].isna()]['Label'].unique()
        print(f"\n[WARNING] Found {len(unmapped_labels)} unmapped labels that became NaN:")
        print(f"Labels: {unmapped_labels}")
    
    X = df.drop(columns=['Label', 'Severity'], errors='ignore')
    y = df['Severity']
    
    # Stratify is critical for rare classes (Heartbleed)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def TrainLog(X_train, X_test, y_train, y_test):
    print("\n--- Model: Weighted Logistic Regression ---")
    model = WeightedLR()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    SaveResults(y_test, y_pred, "Weighted_LR")

def TrainXGB(X_train, X_test, y_train, y_test):
    print("\n--- Model: XGBoost ---")
    model = XGBoost()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    SaveResults(y_test, y_pred, "XGBoost")

def TrainNN(X_train, X_test, y_train, y_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Model: Ordinal Neural Network ---")
    print(f"Training on device: {device.upper()}") # Verification print

    print("\n--- Model: Ordinal Neural Network ---")
    input_dim = X_train.shape[1]
    model = OrdinalNN(input_dim=input_dim, output_dim=4, device=device)
    model.fit(X_train, y_train, epochs=1000)
    y_pred = model.predict(X_test)
    SaveResults(y_test, y_pred, "Ordinal_NN")

# Add this function to drive.py
def TrainEnsemble(X_train, X_test, y_train, y_test):
    print("\n--- Model: Ensemble (XGBoost + Ordinal NN) ---")
    
    # 1. Train XGBoost
    xgb_model = XGBoost()
    xgb_model.fit(X_train, y_train)
    # Get probabilities (N_samples, 4_classes)
    xgb_probs = xgb_model.predict_proba(X_test)

    # 2. Train Neural Network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_model = OrdinalNN(input_dim=X_train.shape[1], output_dim=4, device=device)
    nn_model.fit(X_train, y_train, epochs=1000)
    
    # Get probabilities
    nn_model.eval()
    X_clean = clean_dataset(X_test)
    X_scaled = nn_model.scaler.transform(X_clean)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = nn_model.layer_stack(X_t)
        nn_probs = torch.softmax(logits, dim=1).cpu().numpy()

    # 3. Combine (Soft Voting)
    # We give XGBoost slightly more authority because it's generally more precise
    final_probs = (0.6 * xgb_probs) + (0.4 * nn_probs)
    y_pred = np.argmax(final_probs, axis=1)
    
    SaveResults(y_test, y_pred, "Ensemble_XGB_NN")

# IMPORTANT: You need to patch the XGBoost predict method in lib.py 
# to allow getting raw data for predict_proba if you haven't already.
# Or simpler: Just access self.model directly in drive.py as shown above.

def clean_dataset(X):
    """
    1. Sanitizes Infinity/NaN.
    2. Applies Log-Transform to skewed columns (compression).
    """
    # 1. Sanitize
    if hasattr(X, 'replace'): # Pandas
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        # Convert to numpy for log math
        X_val = X.values
    else: # Numpy
        X_val = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Log-Transform Skewed Features
    # Network data (Bytes, Duration, Packets) is highly skewed.
    # We apply log1p (log(x+1)) to compress the range from [0, 10^9] to [0, 20].
    # This allows the NN to see the difference between "Small" and "Medium" flows.
    # Note: We apply it to absolute values to handle negatives if any exist (though rare in counts)
    X_log = np.sign(X_val) * np.log1p(np.abs(X_val))
    
    return X_log

def WriteWrap(func, *args):
    # This wrapper executes the training function passed to it
    func(*args)

def PrepareDirs():
    global OUTPUT_DIR
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # else:
    #     for f in os.listdir(OUTPUT_DIR):
    #         if f.endswith(".png"):
    #             os.remove(os.path.join(OUTPUT_DIR, f))
    #     print(f"Cleaned old results in {OUTPUT_DIR}")

def main():
    global OUTPUT_DIR

    args = ParseArgs()

    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    PrepareDirs()
    
    # Load data once
    X_train, X_test, y_train, y_test = GetSplits()
    
    if X_train is None:
        return

    if args.logistic:
        WriteWrap(TrainLog, X_train, X_test, y_train, y_test)

    if args.xgb:
        WriteWrap(TrainXGB, X_train, X_test, y_train, y_test)

    if args.neural:
        WriteWrap(TrainNN, X_train, X_test, y_train, y_test)
    
    if args.ensemble:
        WriteWrap(TrainEnsemble, X_train, X_test, y_train, y_test)

    print("\nDone! Check output directory for results.")

if __name__ == "__main__":
    main()