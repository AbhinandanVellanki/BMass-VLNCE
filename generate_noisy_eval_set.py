import os
import shutil
import random
from glob import glob

BASE = "/usr1/datasets/rxr/RxR_VLNCE_v0"

# Directories
ID_SOURCE_DIR = f"{BASE}/noised_bert_features/rxr_val_seen/dropout_10"
CLEAN_DIR     = f"{BASE}/text_features/rxr_val_seen"
NOISY_BASE    = f"{BASE}/noised_bert_features/rxr_val_seen"
OUTPUT_DIR    = f"{BASE}/rxr_val_seen_combined_noisy_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Noise category → percentage of TOTAL ID count
NOISY_PCTS = {
    "dropout_10": 0.07,
    "dropout_25": 0.07,
    "dropout_50": 0.07,
    "dropout_75": 0.07,
    "little":    0.07,
    "moderate":  0.07,
    "heavy":     0.08,
}

# -------------------------------------------------------
# Step 1 — Load final ID list (from dropout10)
# -------------------------------------------------------
id_files = sorted(glob(os.path.join(ID_SOURCE_DIR, "*.npz")))
ids = [os.path.basename(f) for f in id_files]
num_total = len(ids)

print(f"Loaded {num_total} final IDs.")

# -------------------------------------------------------
# Step 2 — Split IDs: 50% clean, 50% noisy
# -------------------------------------------------------
random.shuffle(ids)
num_clean = int(num_total * 0.50)
num_noisy_total = num_total - num_clean

clean_ids  = ids[:num_clean]
noisy_ids_pool = ids[num_clean:]   # IDs reserved for noisy selection

print(f"Clean IDs: {len(clean_ids)}")
print(f"Noisy IDs needed: {len(noisy_ids_pool)}")

# -------------------------------------------------------
# Step 3 — Assign noisy IDs by noise percentages
# -------------------------------------------------------
noisy_assignments = {}  # id → category
remaining_ids = noisy_ids_pool.copy()

for category, pct in NOISY_PCTS.items():
    needed = int(num_total * pct)
    
    chosen = random.sample(remaining_ids, needed)
    for cid in chosen:
        noisy_assignments[cid] = category
    
    # remove assigned
    remaining_ids = [x for x in remaining_ids if x not in chosen]

# Safety check: remaining should be ~0 (or tiny rounding diff)
if len(remaining_ids) > 0:
    print(f"Distributing remaining {len(remaining_ids)} IDs to 'heavy'")
    for cid in remaining_ids:
        noisy_assignments[cid] = "heavy"

print(f"Total noisy files assigned: {len(noisy_assignments)}")

# -------------------------------------------------------
# Step 4 — Copy files to output
# -------------------------------------------------------
# Clean copies
for cid in clean_ids:
    src = os.path.join(CLEAN_DIR, cid)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing clean file: {src}")
    shutil.copy(src, os.path.join(OUTPUT_DIR, cid))

# Noisy copies
for cid, category in noisy_assignments.items():
    src = os.path.join(NOISY_BASE, category, cid)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing noisy file: {src}")
    shutil.copy(src, os.path.join(OUTPUT_DIR, cid))

print("Done!")
print("Total files in new dataset:", len(os.listdir(OUTPUT_DIR)))
