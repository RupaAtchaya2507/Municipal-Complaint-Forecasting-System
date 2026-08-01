import pandas as pd
import sys

print("Loading synthetic_complaints.csv (this may take 1-2 mins)...")
sys.stdout.flush()

df = pd.read_csv("data/synthetic_complaints.csv", encoding="latin-1")
print(f"Loaded {len(df):,} rows")
sys.stdout.flush()

df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", dayfirst=False, errors="coerce")
df_filtered = df[df["created_at"].dt.year <= 2022].copy()
print(f"Filtered to 2019-2022: {len(df_filtered):,} rows")
sys.stdout.flush()

out_path = "data/synthetic_complaints_2019_2022.csv"
print(f"Saving to {out_path}...")
sys.stdout.flush()

df_filtered.to_csv(out_path, index=False)
print("Done! File saved successfully.")
print(f"Year distribution:\n{df_filtered['created_at'].dt.year.value_counts().sort_index().to_string()}")
