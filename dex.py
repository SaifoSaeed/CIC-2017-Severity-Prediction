from dython.nominal import associations
from pathlib import Path
import pandas as pd
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
    return parser.parse_args()

#Better print function (for me).
def PrintAll(df: pd.DataFrame, n: int = 4) -> None:
    cols_list = df.columns.to_list()
    lim = 4
    
    for col_idx in range(0, len(cols_list), lim):
        cols = cols_list[col_idx : col_idx + lim]
        print(df[cols].head(n),"\n")

#Filter all DataFrame objects by label, then combine into one.
def FilterBenMal():
    def FilterCheck(ben_rows, mal_rows):
        final_ben_rows = ben_rows
        final_mal_rows = mal_rows

        ben_ratio = final_ben_rows / ben_rows
        mal_ratio = final_mal_rows / mal_rows

        if ben_ratio != 1.0 or mal_ratio != 1.0:
            raise RuntimeError(f"Failed validation check.\nben_ratio = {ben_ratio} | mal_ratio = {mal_ratio}")
    
    def RatioedAggregate():
        comb_ben_df = pd.read_csv(output_files[0])
        comb_mal_df = pd.read_csv(output_files[1])
        ben_rows = comb_ben_df.shape[0]
        mal_rows = comb_mal_df.shape[0]
        FilterCheck(ben_rows, mal_rows)

        percent = mal_rows / ben_rows

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

    return RatioedAggregate()

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
        "Fwd Packet Length Max", "Fwd Packet Length Min",# "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",# "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "CWE Flag Count", "URG Flag Count", "ECE Flag Count",
        "Fwd Header Length", "Fwd Header Length.1", "Bwd Header Length", "Bwd Header Length.1", 
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

import pandas as pd
import numpy as np
from dython.nominal import associations

def DropPerfect():
    df = pd.read_csv(agg_df_file)
    
    df.columns = df.columns.str.strip()
    
    sample_df = df.sample(frac=0.1, random_state=42)
    print(f"Computing correlations on shape: {sample_df.shape}")

    # 2. Run Associations
    # nom_nom_assoc='cramer' forces symmetric output for categoricals (easier to filter)
    # If you use Theil's U (default), it is asymmetric.
    result = associations(
        sample_df, 
        nom_nom_assoc='cramer', 
        compute_only=True, 
        cramers_v_bias_correction=True
    )
    corr_matrix = result['corr']

    print("\n--- PERFECT CORRELATIONS (Redundant Features) ---")
    
    seen_pairs = set()
    drop_candidates = set()

    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_1 = columns[i]
            col_2 = columns[j]
            
            val = corr_matrix.iloc[i, j]
            
            if np.isclose(abs(val), 1.0, atol=1e-5):
                print(f"({col_1} vs {col_2}) : {val:.4f}")
                
                # Suggest dropping the second one in the pair
                drop_candidates.add(col_2)

    print(f"\nRecommended columns to drop (Redundant): {list(drop_candidates)}")
    return df

#Decorator.
def WriteWrap(func):
    start = time.perf_counter()
    df = func()
    end = time.perf_counter()

    print(f"info = {df.info()}")
    print(f"shape = {df.shape}")
    for col in df.columns:
        if(df[col].nunique() < 10):
            unique = df[col].unique()

            print(f"{col}__unique__ = {unique}")

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

if __name__ == "__main__":
    args = ParseArgs()

    if args.aggregate:
        DeleteExisting()
        WriteWrap(FilterBenMal)

    if args.drop_extra:
        WriteWrap(DropExtra)

    if args.clear_unique:
        WriteWrap(ClearUnique)

    if args.drop_perfect:
        WriteWrap(DropPerfect)

    try:
        df = pd.read_csv(agg_df_file)
        df.info()
        PrintAll(df)
    except Exception as e:
        print(f"Error, exception raised: {e}")