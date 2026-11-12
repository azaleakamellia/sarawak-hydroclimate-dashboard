# inject_station_names.py
import pandas as pd
from pathlib import Path

print("🔄 Loading existing master Parquet file...")
project_root = Path(__file__).parent
parquet_in = project_root / "outputs" / "processed" / "sarawak_MASTER_with_insitu.parquet"
df = pd.read_parquet(parquet_in)

print("🔄 Loading station metadata from CSV...")
gauge_csv = project_root / "data" / "gauges" / "sarawak_all_gauges.csv"
meta = pd.read_csv(gauge_csv)

# ️ FIX: Force both station_id columns to be strings to prevent type mismatch errors
df['station_id'] = df['station_id'].astype(str).str.strip()
meta['station_id'] = meta['station_id'].astype(str).str.strip()

# Keep only the ID and Name, and remove duplicates just in case
meta = meta[['station_id', 'station_name']].drop_duplicates()

print("🔄 Merging station names back into the master dataset...")
# Left join to ensure we don't lose any satellite data
df = df.merge(meta, on='station_id', how='left')

# Reorder columns so station_name is right next to station_id
cols = ['station_id', 'station_name', 'lon', 'lat', 'date'] + [c for c in df.columns if c not in ['station_id', 'station_name', 'lon', 'lat', 'date']]
df = df[cols]

print(f"💾 Saving new master file...")
parquet_out = project_root / "outputs" / "processed" / "sarawak_hydro_data_FULL_amended.parquet"
df.to_parquet(parquet_out, compression='snappy', index=False)

print(f"✅ SUCCESS! Saved {parquet_out}")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Unique Stations: {df['station_id'].nunique()}")
print(f"   Sample Names: {df['station_name'].dropna().unique()[:5]}")