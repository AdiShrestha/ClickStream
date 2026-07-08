import numpy as np
from src.feature_extraction import load_cmu_features

X, subjects, sessions, feature_cols = load_cmu_features()
hold_idx = [i for i, c in enumerate(feature_cols) if c.startswith("H.")]
dd_idx = [i for i, c in enumerate(feature_cols) if c.startswith("DD.")]

subj_mask = (subjects == "s049")
enroll_mask = np.isin(sessions, (1, 2, 3, 4))
mask = subj_mask & enroll_mask

X_s049 = X[mask]

# 0.5s / 2.0s thresholds
hold_thresh = 0.5
dd_thresh = 2.0

outliers = []
col_trip_counts = {c: 0 for c in feature_cols}
rows_by_trip_count = {}

for i, row in enumerate(X_s049):
    h_vals = row[hold_idx]
    dd_vals = row[dd_idx]
    
    tripped_cols = []
    for j, val in enumerate(h_vals):
        if val > hold_thresh:
            col_name = feature_cols[hold_idx[j]]
            tripped_cols.append((col_name, val))
            col_trip_counts[col_name] += 1
            
    for j, val in enumerate(dd_vals):
        if val > dd_thresh:
            col_name = feature_cols[dd_idx[j]]
            tripped_cols.append((col_name, val))
            col_trip_counts[col_name] += 1
            
    if tripped_cols:
        outliers.append((i, tripped_cols))
        cnt = len(tripped_cols)
        rows_by_trip_count[cnt] = rows_by_trip_count.get(cnt, 0) + 1

print(f"Total flagged rows: {len(outliers)} out of {X_s049.shape[0]}")
print("\nClustering (How many columns tripped simultaneously per row?):")
for cnt in sorted(rows_by_trip_count.keys()):
    print(f"  {rows_by_trip_count[cnt]} rows tripped exactly {cnt} column(s)")

print("\nPer-Column Trip Frequencies:")
for col, count in sorted(col_trip_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {col}: {count} trips")

print("\nRaw Flagged Values for the 49 Rows:")
for i, trips in outliers:
    trip_str = ", ".join([f"{col}={val:.4f}s" for col, val in trips])
    print(f"  Row {i:03d}: {trip_str}")
