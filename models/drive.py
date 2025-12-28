import os
import torch
import argparse
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

    print("\nDone! Check output directory for results.")

if __name__ == "__main__":
    main()