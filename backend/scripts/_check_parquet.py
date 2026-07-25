"""Quick parquet inspection script."""
import pandas as pd

df = pd.read_parquet("data/processed/features_combined.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"First 5 cols: {df.columns[:5].tolist()}")
print(f"Last 10 cols: {df.columns[-10:].tolist()}")
meta_cols = {"txId", "timeStep", "class"}
feature_cols = [c for c in df.columns if c not in meta_cols]
nan_total = df[feature_cols].isna().sum().sum()
print(f"Feature columns: {len(feature_cols)}")
print(f"NaN total: {nan_total}")
if "class" in df.columns:
    print(f"Class values: {df['class'].unique()}")
    print(f"Class counts: {df['class'].value_counts().to_dict()}")
if "timeStep" in df.columns:
    print(f"TimeStep range: [{df['timeStep'].min()}, {df['timeStep'].max()}]")
print(f"txId unique: {df['txId'].nunique()} / {len(df)}")
