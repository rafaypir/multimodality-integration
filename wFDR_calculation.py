import pandas as pd
import os

ICA_FILE_PATH = "path/to/your/normalized_ica_data.tsv"
PATHWAY_DIR = "path/to/your/processed_pathways_directory/"
OUTPUT_FILE_PATH = "path/to/your/output_scores.tsv"

ica_df = pd.read_csv(ICA_FILE_PATH, sep="\t", index_col=0)
ica_df.index = [f"ic{i:03d}" for i in range(1, 101)]
patients = ica_df.columns

all_go_ids = set()
pathway_data = {f"ic{i:03d}": {} for i in range(1, 101)}

for ic in range(1, 101):
    ic_name = f"ic{ic:03d}"
    file_path = os.path.join(PATHWAY_DIR, f"{ic_name}_top10_pathways.txt")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep="\t")
        for _, row in df.iterrows():
            go_id = row["GO.ID"]
            log_fdr = row["log_FDR"]
            pathway_data[ic_name][go_id] = log_fdr
            all_go_ids.add(go_id)

all_go_ids = sorted(list(all_go_ids))

score_data = {go_id: [] for go_id in all_go_ids}
for patient in patients:
    loadings = ica_df[patient]
    for go_id in all_go_ids:
        score = 0.0
        for ic in [f"ic{i:03d}" for i in range(1, 101)]:
            log_fdr = pathway_data[ic].get(go_id, 0.0)
            ic_loading = loadings[ic]
            score += log_fdr * ic_loading
        score_data[go_id].append(score)

score_df = pd.DataFrame(score_data, index=patients)
score_df.to_csv(OUTPUT_FILE_PATH, sep="\t")

print(f"Pathway scores saved to {OUTPUT_FILE_PATH}")
print(f"Shape: {score_df.shape} (patients x pathways)")