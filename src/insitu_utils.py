# src/insitu_utils.py
import pandas as pd
from pathlib import Path

def load_insitu_data(csv_path):
    """
    Intelligently loads and standardizes in-situ gauge data.
    Handles unnamed precipitation columns and standardizes station IDs and names.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Standardize Station ID column
    if 'station_no' in df.columns:
        df = df.rename(columns={'station_no': 'station_id'})
    elif 'station_id' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'station_id'})

    # Extract station name if available
    cols_to_keep = ["station_id", "date"]
    
    if 'station_name' in df.columns:
        cols_to_keep.append('station_name')
        
    # 2. Standardize Precipitation column
    if 'precipitation_sum' in df.columns:
        df = df.rename(columns={'precipitation_sum': 'Gauge_Observed'})
    elif 'Gauge_Observed' in df.columns:
        pass # Already correct
    elif 'precip_insitu' in df.columns:
        df = df.rename(columns={'precip_insitu': 'Gauge_Observed'})
    else:
        last_col = df.columns[-1]
        df = df.rename(columns={last_col: 'Gauge_Observed'})
        print(f"  ℹ️ Renamed unnamed column '{last_col}' to 'Gauge_Observed'")
        
    cols_to_keep.append("Gauge_Observed")

    # 3. Clean data types
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"])
    df["Gauge_Observed"] = pd.to_numeric(df["Gauge_Observed"], errors="coerce")
    
    # 4. Ensure we only return the necessary columns
    return df[cols_to_keep]

def inject_insitu_into_master(master_parquet_path, insitu_csv_path, gauge_metadata_path, output_parquet_path):
    """
    Merges in-situ data and station names into the master parquet file.
    """
    print(f"🔄 Loading master data from {master_parquet_path}...")
    master_df = pd.read_parquet(master_parquet_path)
    master_df["station_id"] = master_df["station_id"].astype(str).str.strip()
    master_df["date"] = pd.to_datetime(master_df["date"])
    
    print(f"🔄 Loading in-situ data from {insitu_csv_path}...")
    insitu_df = load_insitu_data(insitu_csv_path)

    # Bring in the station names from the gauge metadata if they aren't in the insitu data
    if 'station_name' not in insitu_df.columns:
        print(f"🔄 Loading gauge metadata from {gauge_metadata_path}...")
        meta_df = pd.read_csv(gauge_metadata_path)
        meta_df["station_id"] = meta_df["station_id"].astype(str).str.strip()
        
        # Merge names into the master file
        master_df = master_df.merge(meta_df[['station_id', 'station_name']], on='station_id', how='left')
    
    print("🔄 Merging datasets on ['station_id', 'date']...")
    
    # LEFT JOIN: Preserves all master rows
    merged_df = master_df.merge(insitu_df, on=["station_id", "date"], how="left")
    
    # Reorder columns to put station_name right after station_id
    cols = merged_df.columns.tolist()
    if 'station_name' in cols:
        cols.insert(1, cols.pop(cols.index('station_name')))
        merged_df = merged_df[cols]
    
    merged_df = merged_df.sort_values(["station_id", "date"]).reset_index(drop=True)
    
    out_path = Path(output_parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(out_path, compression="snappy", index=False)
    
    print("-" * 60)
    print(f"✅ SUCCESS! Saved master dataset to: {out_path}")
    print(f"   Shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
    
    valid_insitu = merged_df["Gauge_Observed"].notna().sum()
    total_rows = len(merged_df)
    print(f"   Gauge_Observed: {valid_insitu:6d} valid / {total_rows} ({100 * valid_insitu / total_rows:5.1f}%)")
    
    return merged_df