import os
import shutil
import kagglehub

print("Downloading dataset using kagglehub...")
path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
print("Downloaded dataset to:", path)

target_dir = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(target_dir, exist_ok=True)

print(f"Copying files to {target_dir}...")

csv_files = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]

found_files = 0
for root, dirs, files in os.walk(path):
    for file in files:
        if file in csv_files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            print(f"Copying {file}...")
            shutil.copy2(src_path, dst_path)
            found_files += 1

if found_files == len(csv_files):
    print("All required CSV files copied successfully.")
else:
    print(f"Warning: Only found {found_files}/{len(csv_files)} required CSV files.")
