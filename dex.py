from dython.nominal import associations
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import time
import os
import gc

#Global paths.
dir = Path("dfs")
dfs = os.listdir(dir)
output_dir = Path("proc_dfs")
agg_df_file = output_dir / Path("ben_mal.csv")
output_files = [output_dir / Path("ben.csv"), output_dir / Path("mal.csv")]

#Parse command line arguments.
def ParseArgs():
    parser = argparse.ArgumentParser(description="Welcome to this CIC-IDS-2017 Data EXploration script.\nYou can use this to apply this to preprocess the data.")
    
    parser.add_argument( "-a", "--aggregate", action="store_true", help="aggregates benign and malicious data into two separate Dataframe objects in the 'proc_dfs' directory.")
    parser.add_argument( "-cu", "--clear-unique", action="store_true", help="aggregates benign and malicious data into two separate Dataframe objects in the 'proc_dfs' directory.")
    parser.add_argument("-dp", "--drop-perfect", action="store_true", help="deletes 1.0 column correlations.")
    parser.add_argument("-dx", "--drop-extra", action="store_true", help="drops superfluous columns. (subjective)")
    parser.add_argument("-fe", "--feature-extract", action="store_true", help="extracts columns from other columns using domain knowledge")
    return parser.parse_args()

#Filter all DataFrame objects by label, then combine into one.
def FilterBenMal():
    #Clean NaN, inf and duplicate rows from all.
    def CleanInvDups():
        ben_rows_dropped = 0
        mal_rows_dropped = 0
        
        print("Reading intermediate files...")
        ben_df = pd.read_csv(output_files[0])
        mal_df = pd.read_csv(output_files[1])
        
        # --- 1. Fix Labels Early ---
        print("Fixing corrupted labels...")
        fix_map = {
            'Web Attack � Brute Force': 'WebAttackBruteForce',
            'Web Attack � XSS': 'WebAttackXSS',
            'Web Attack � Sql Injection': 'WebAttackSqlInjection'
            }
        ben_df['Label'] = ben_df['Label'].replace(fix_map)
        mal_df['Label'] = mal_df['Label'].replace(fix_map)

        ben_rows_org = ben_df.shape[0]
        mal_rows_org = mal_df.shape[0]
        print(f"Original row counts: mal_rows = {mal_rows_org} | ben_rows = {ben_rows_org}")

        # --- 2. Drop Duplicates ---
        ben_df.drop_duplicates(inplace=True)
        mal_df.drop_duplicates(inplace=True)
        print(f"After dropping duplicates: mal_rows = {mal_df.shape[0]} | ben_rows = {ben_df.shape[0] }")

        # --- 3. Drop NaNs ---
        ben_df.dropna(inplace=True)
        mal_df.dropna(inplace=True)
        print(f"After dropping NaN values: mal_rows = {mal_df.shape[0]} | ben_rows = {ben_df.shape[0] }")

        # --- 4. Drop Infinity (The Fix) ---
        print("Scrubbing Infinity values...")
        cols_to_check = ['Flow Bytes/s', 'Flow Packets/s']
        
        for col in cols_to_check:
            # Force "Infinity" strings to be real numeric NaNs or Infs
            ben_df[col] = pd.to_numeric(ben_df[col], errors='coerce')
            mal_df[col] = pd.to_numeric(mal_df[col], errors='coerce')        

        # FIX: Do not chain inplace operations. Do them sequentially.
        # Replace Inf with NaN in the actual dataframe
        ben_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        mal_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Now drop the NaNs we just created
        ben_df.dropna(inplace=True)
        mal_df.dropna(inplace=True)

        print(f"After dropping inf and -inf values: mal_rows = {mal_df.shape[0]} | ben_rows = {ben_df.shape[0] }")

        ben_df.to_csv(output_files[0], index=False)
        mal_df.to_csv(output_files[1], index=False)

        ben_rows_final = ben_df.shape[0]
        mal_rows_final = mal_df.shape[0]

        del ben_df
        del mal_df
        gc.collect()

        return (ben_rows_final, mal_rows_final)
    
    def RowCheck(ben_cleaned_rows, mal_cleaned_rows, ben_read_rows, mal_read_rows):
        ben_read_over_cleaned = ben_read_rows / ben_cleaned_rows
        mal_read_over_cleaned = mal_read_rows / mal_cleaned_rows
        
        if not (np.isclose(1, ben_read_over_cleaned) or np.isclose(1, mal_read_over_cleaned)):
            raise Exception("Ratios are wrong.")

    def RatioedAggregate(ben_rows_final, mal_rows_final):
        comb_ben_df = pd.read_csv(output_files[0])
        comb_mal_df = pd.read_csv(output_files[1])

        ben_rows = comb_ben_df.shape[0]
        mal_rows = comb_mal_df.shape[0]

        RowCheck(ben_rows_final, mal_rows_final, ben_rows, mal_rows)

        percent = mal_rows / ben_rows
        print(f"mal_rows / ben_rows = {percent}")
        sample_ben_df = comb_ben_df.sample(frac=percent, random_state=42)

        return pd.concat([sample_ben_df, comb_mal_df], ignore_index=True)

    header_write = False
    
    ben_rows = mal_rows = 0

    for df_idx in range(len(dfs)):
        df = dfs[df_idx]
        temp_df = pd.read_csv(dir / Path(df))
        print(f"Processing DataFrame {df}: {str(temp_df.shape)}")
                
        temp_df.columns = temp_df.columns.str.strip()
        is_ben = temp_df['Label'] == "BENIGN"
        
        ben_df = temp_df[is_ben]
        mal_df = temp_df[~is_ben]

        ben_rows += ben_df.shape[0]
        mal_rows += mal_df.shape[0]

        if not ben_df.empty:
            ben_df.to_csv(output_files[0], mode='a', index=False, header=not header_write)

        if not mal_df.empty:
            mal_df.to_csv(output_files[1], mode='a', index=False, header=not header_write)

        header_write = True
        del temp_df, ben_df, mal_df
        gc.collect()

    ben_rows_final, mal_rows_final = CleanInvDups()

    return RatioedAggregate(ben_rows_final, mal_rows_final)

#Clear columns with one unique value.
def ClearUnique():
    df = pd.read_csv(agg_df_file)
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count == 1:
            print(f"Dropping column: {col} with unique_count = {unique_count} : {df[col].unique()}")
            df.drop(col, axis=1, inplace=True)
    return df

#Drops redundant columns (subjective).
def DropExtra():
    df = pd.read_csv(agg_df_file)
    to_drop = [
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "CWE Flag Count", "URG Flag Count", "ECE Flag Count",
        "Fwd Header Length", "Fwd Header Length.1", "Bwd Header Length", "Bwd Header Length.1", 
        "Total Length of Bwd Packets",
        "Unnamed: 0", "Unnamed: 0.1",
        "Fwd PSH Flags", "Fwd URG Flags", "Bwd PSH Flags", "Bwd URG Flags", 
        "Active Std", "Active Max", "Active Min", "Idle Std", "Idle Max", "Idle Min",
        "min_seg_size_forward", "act_data_pkt_fwd",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Average Packet Size",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "Flow IAT Max", "Flow IAT Min"
        ]
    
    to_drop = pd.Index(to_drop).str.strip()
    for col in to_drop:
        try:
            df.drop(axis=1, columns=[col], inplace=True)
            print(f"Dropping extra {col} successfully.")
        except Exception as e:
            print(f"Failed to drop {col} due to {e}.")

    return df

#Checks top 20 correlations in the dataset. 
def DropPerfect():
    df = pd.read_csv(agg_df_file)
    df.columns = df.columns.str.strip()
    
    print(f"Initial shape: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    sample_df = df.sample(frac=0.1, random_state=42)
    sample_df[numeric_cols] = sample_df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
    sample_df = sample_df.dropna()

    print(f"Computing correlations on sample shape: {sample_df.shape}")

    result = associations(
        sample_df, 
        nom_nom_assoc='cramer', 
        compute_only=True, 
        cramers_v_bias_correction=True
    )
    
    corr_matrix = result['corr']

    corr_pairs = []
    columns = corr_matrix.columns
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_1 = columns[i]
            col_2 = columns[j]
            val = corr_matrix.iloc[i, j]
            corr_pairs.append((col_1, col_2, val))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\n--- TOP 20 CORRELATIONS ---")
    for c1, c2, val in corr_pairs[:20]:
        print(f"{c1} vs {c2}: {val:.4f}")

    drop_candidates = set()
    print("\n--- PERFECT CORRELATIONS ---")
    for c1, c2, val in corr_pairs:
        if np.isclose(abs(val), 1.0, atol=1e-5):
            print(f"Removing {c2} (redundant to {c1})")
            drop_candidates.add(c2)

    prev_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    curr_rows = df.shape[0]
    
    print(f"Feature-induced duplicates dropped: {prev_rows - curr_rows}")
    
    return df

#Uses stratified sampling for 50/50 benign/malignant instances.
def RebalanceDataset(df: pd.DataFrame):
    print("\n--- REBALANCING DATASET (50/50 Stratified) ---")
    
    is_ben = df['Label'] == 'BENIGN'
    ben_df = df[is_ben]
    mal_df = df[~is_ben]
    
    ben_count = len(ben_df)
    mal_count = len(mal_df)
    
    target_count = min(ben_count, mal_count)
    
    if ben_count > target_count:
        ben_df = ben_df.sample(n=target_count, random_state=42)
        
    if mal_count > target_count:
        mal_df = mal_df.sample(n=target_count, random_state=42)
        
    balanced_df = pd.concat([ben_df, mal_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

#Calculates the Coefficient of Variation for Flow Inter-Arrival Time.
def CalculateIATCV(df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting Feature: Flow IAT Coefficient of Variation...")
    df['Flow IAT CV'] = df['Flow IAT Std'] / df['Flow IAT Mean']
    df['Flow IAT CV'] = df['Flow IAT CV'].replace([np.inf, -np.inf], 0)
    df['Flow IAT CV'] = df['Flow IAT CV'].fillna(0)
    return df

#Calculates the ratio of Forward Bytes to Backward Bytes.
def CalculateTrafficAsymmetry(df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting Feature: Traffic Asymmetry Ratio...")
    bwd_bytes_estimated = df['Total Backward Packets'] * df['Bwd Packet Length Mean']
    df['Traffic Asymmetry'] = df['Total Length of Fwd Packets'] / bwd_bytes_estimated
    df['Traffic Asymmetry'] = df['Traffic Asymmetry'].replace([np.inf, -np.inf], -1)
    df['Traffic Asymmetry'] = df['Traffic Asymmetry'].fillna(0)
    return df

#Adds various features based on domain knowledge.
def FeatureExtract():
    df = pd.read_csv(agg_df_file)
    df = CalculateIATCV(df)
    df = CalculateTrafficAsymmetry(df)
    return df

def WriteWrap(func):
    start = time.perf_counter()
    df = func()
    end = time.perf_counter()

    # CRASH ON INFINITY CHECK
    numeric_df = df.select_dtypes(include=[np.number])
    is_inf = np.isinf(numeric_df)
    
    if is_inf.any().any():
        cols_with_inf = is_inf.any()
        bad_cols = cols_with_inf[cols_with_inf].index.tolist()
        bad_row_count = is_inf.any(axis=1).sum()
        
        error_msg = (
            f"\n[CRITICAL ERROR] Infinity values generated by {func.__name__}!\n"
            f"Columns affected: {bad_cols}\n"
            f"Row count: {bad_row_count}\n"
        )
        df.to_csv("ERRORED_DF.csv")
        raise ValueError(error_msg)

    print(f"Done with {func.__name__}. ET: {end-start:.3f}.")
    df.to_csv(agg_df_file, index=False)

#Clear existing directories.
def DeleteExisting():
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for out in output_files:
        if os.path.exists(out):
            os.remove(out)
    
    if os.path.exists(agg_df_file):
        os.remove(agg_df_file)

def main():
    args = ParseArgs()

    if args.aggregate:
        DeleteExisting()
        WriteWrap(FilterBenMal)

    if args.drop_extra:
        WriteWrap(DropExtra)

    if args.clear_unique:
        WriteWrap(ClearUnique)

    if args.drop_perfect:
        def DropAndBalance():
            df = DropPerfect()
            df = RebalanceDataset(df)
            return df
        WriteWrap(DropAndBalance)

    if args.feature_extract:
        WriteWrap(FeatureExtract)
    
    print("Done!")

if __name__ == "__main__":
    main()