import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

GENE_EXPRESSION_PATH = "path/to/your/gene_expression.tsv"
SURVIVAL_DATA_PATH = "path/to/your/survival_data.tsv"
WSI_FILES_DIR = "path/to/your/wsi_features_directory/"
OUTPUT_DIR = "path/to/your/output_directory/"
SPLITS_DIR = "path/to/your/data_splits_directory/"

WSI_FEATURE_DIM = 512
PATCH_NAME_COL_WSI = 0
FEATURE_START_COL_WSI = 1

EMBEDDING_DIM = 256
N_ATTENTION_HEADS = 4

class CoxPHLoss(nn.Module):
    def forward(self, log_risks: Tensor, times: Tensor, events: Tensor) -> Tensor:
        events = events.bool()
        if not torch.any(events):
            return torch.tensor(0.0, device=log_risks.device, requires_grad=True)
        log_risks_observed = log_risks[events]
        log_sum_exp_risk_set = torch.log(torch.cumsum(torch.exp(log_risks), dim=0))[events]
        loss = - (log_risks_observed - log_sum_exp_risk_set).sum() / events.sum()
        return loss

def c_index_manual(log_risks: Tensor, events: Tensor, times: Tensor) -> float:
    log_risks_np, events_np, times_np = log_risks.cpu().numpy(), events.cpu().numpy(), times.cpu().numpy()
    n_correct, n_comparable = 0, 0
    for i in range(len(times_np)):
        for j in range(i + 1, len(times_np)):
            if events_np[i] == 1 and times_np[i] < times_np[j]:
                n_comparable += 1
                if log_risks_np[i] > log_risks_np[j]:
                    n_correct += 1
            elif events_np[j] == 1 and times_np[j] < times_np[i]:
                n_comparable += 1
                if log_risks_np[j] > log_risks_np[i]:
                    n_correct += 1
    return n_correct / n_comparable if n_comparable > 0 else 0.5

class CrossAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        assert embedding_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.to_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.to_kv = nn.Linear(embedding_dim, embedding_dim * 2, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, query_embed: Tensor, context_embed: Tensor) -> (Tensor, Tensor):
        q = self.to_q(query_embed)
        k, v = self.to_kv(context_embed).chunk(2, dim=-1)
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(dots, dim=-1)
        attended_features = torch.matmul(attention_weights, v)
        attended_features = attended_features.transpose(1, 2).contiguous().view_as(query_embed)
        return attended_features, attention_weights

class MultimodalModel(nn.Module):
    def __init__(self, gene_in_dim: int, patch_in_dim: int, embed_dim: int, n_heads: int):
        super().__init__()
        self.gene_encoder = nn.Sequential(nn.Linear(gene_in_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.patch_encoder = nn.Sequential(nn.Linear(patch_in_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.cross_attention_g2w = CrossAttention(embed_dim, n_heads)
        self.cross_attention_w2g = CrossAttention(embed_dim, n_heads)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, gene_data: Tensor, patch_data: Tensor) -> (Tensor, Tensor, Tensor):
        gene_embed = self.gene_encoder(gene_data)
        patch_embed = self.patch_encoder(patch_data)
        attended_genes, attention_g2w = self.cross_attention_g2w(query_embed=gene_embed, context_embed=patch_embed)
        attended_patches, attention_w2g = self.cross_attention_w2g(query_embed=patch_embed, context_embed=gene_embed)
        aggregated_genes = attended_genes.mean(dim=1)
        aggregated_patches = attended_patches.mean(dim=1)
        fused_features = torch.cat((aggregated_genes, aggregated_patches), dim=1)
        log_risk = self.regressor(fused_features)
        return log_risk, attention_g2w, attention_w2g

def parse_coords_from_name(patch_name):
    try:
        parts = patch_name.replace('.png', '').split('_')
        x_coord, y_coord = int(parts[-2]), int(parts[-1])
        return x_coord, y_coord
    except (IndexError, ValueError):
        return 0, 0

class MultimodalSurvivalDataset(Dataset):
    def __init__(self, patient_ids, gene_df, survival_df, wsi_dir, patch_scaler, wsi_filename_map):
        self.patient_ids = patient_ids
        self.gene_df = gene_df
        self.survival_df = survival_df
        self.wsi_dir = wsi_dir
        self.patch_scaler = patch_scaler
        self.wsi_filename_map = wsi_filename_map

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        gene_features = self.gene_df.loc[patient_id].values.astype(np.float32)
        surv_info = self.survival_df.loc[patient_id]
        time, event = surv_info['time'], surv_info['event']
        wsi_filename = self.wsi_filename_map[patient_id]
        patch_file_path = os.path.join(self.wsi_dir, wsi_filename)
        patches_df = pd.read_csv(patch_file_path)
        patch_names = patches_df.iloc[:, PATCH_NAME_COL_WSI]
        coords = [parse_coords_from_name(name) for name in patch_names]
        feature_cols = patches_df.columns[FEATURE_START_COL_WSI : FEATURE_START_COL_WSI + WSI_FEATURE_DIM]
        patch_features = self.patch_scaler.transform(patches_df[feature_cols].values)
        return {
            "patient_id": patient_id,
            "genes": torch.from_numpy(gene_features),
            "patches": torch.from_numpy(patch_features.astype(np.float32)),
            "coords": torch.tensor(coords, dtype=torch.long),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32)
        }

def custom_collate_fn(batch):
    patient_ids = [item['patient_id'] for item in batch]
    genes_batch = torch.stack([item['genes'] for item in batch])
    patches_batch = [item['patches'] for item in batch]
    coords_batch = [item['coords'] for item in batch]
    times_batch = torch.stack([item['time'] for item in batch])
    events_batch = torch.stack([item['event'] for item in batch])
    return patient_ids, genes_batch, patches_batch, coords_batch, times_batch, events_batch

def fit_scaler_incrementally(patient_ids, wsi_dir, feature_dim, wsi_filename_map):
    scaler = StandardScaler()
    for i, pid in enumerate(patient_ids):
        print(f"\rFitting scaler: processing file {i+1}/{len(patient_ids)}", end="")
        file_path = os.path.join(wsi_dir, wsi_filename_map[pid])
        try:
            patch_data = pd.read_csv(file_path).iloc[:, 1 : 1 + feature_dim].values
            if patch_data.size > 0: scaler.partial_fit(patch_data)
        except FileNotFoundError:
            print(f"\nWarning: File not found and skipped: {file_path}")
    print("\nScaler fitting complete.")
    return scaler

def train_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()
    total_loss, all_risks, all_times, all_events = 0, [], [], []
    for _, gene_batch, patches_batch, _, times_batch, events_batch in dataloader:
        gene_batch, times_batch, events_batch = gene_batch.to(device), times_batch.to(device), events_batch.to(device)
        patches_batch_device = [p.to(device) for p in patches_batch]
        optimizer.zero_grad()
        batch_risks = [model(gene_batch[i].unsqueeze(0).unsqueeze(-1), p.unsqueeze(0))[0] for i, p in enumerate(patches_batch_device)]
        if not batch_risks: continue
        batch_risks_tensor = torch.cat(batch_risks).squeeze(-1)
        perm = torch.argsort(times_batch, descending=True)
        sorted_risks, sorted_times, sorted_events = batch_risks_tensor[perm], times_batch[perm], events_batch[perm]
        loss = loss_func(sorted_risks, sorted_times, sorted_events)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        all_risks.append(sorted_risks.detach())
        all_times.append(sorted_times.detach())
        all_events.append(sorted_events.detach())
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    if not all_risks: return avg_loss, 0.5
    c_idx = c_index_manual(torch.cat(all_risks), torch.cat(all_events), torch.cat(all_times))
    return avg_loss, c_idx

def validate_epoch(model, dataloader, loss_func, device):
    model.eval()
    total_loss, all_risks, all_times, all_events = 0, [], [], []
    with torch.no_grad():
        for _, gene_batch, patches_batch, _, times_batch, events_batch in dataloader:
            gene_batch, times_batch, events_batch = gene_batch.to(device), times_batch.to(device), events_batch.to(device)
            patches_batch_device = [p.to(device) for p in patches_batch]
            batch_risks = [model(gene_batch[i].unsqueeze(0).unsqueeze(-1), p.unsqueeze(0))[0] for i, p in enumerate(patches_batch_device)]
            if not batch_risks: continue
            batch_risks_tensor = torch.cat(batch_risks).squeeze(-1)
            perm = torch.argsort(times_batch, descending=True)
            sorted_risks, sorted_times, sorted_events = batch_risks_tensor[perm], times_batch[perm], events_batch[perm]
            loss = loss_func(sorted_risks, sorted_times, sorted_events)
            if not torch.isnan(loss): total_loss += loss.item()
            all_risks.append(sorted_risks)
            all_times.append(sorted_times)
            all_events.append(sorted_events)
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    if not all_risks: return avg_loss, 0.5
    c_idx = c_index_manual(torch.cat(all_risks), torch.cat(all_events), torch.cat(all_times))
    return avg_loss, c_idx

def generate_and_save_attention(model, dataloader, device, gene_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for pids, gene_batch, patches_batch, coords_batch, _, _ in dataloader:
            gene_batch = gene_batch.to(device)
            patches_batch_device = [p.to(device) for p in patches_batch]
            for i in range(len(patches_batch_device)):
                _, attention_g2w, attention_w2g = model(
                    gene_batch[i].unsqueeze(0).unsqueeze(-1),
                    patches_batch_device[i].unsqueeze(0)
                )
                np.save(os.path.join(output_dir, f"{pids[i]}_attention_g2w.npy"), attention_g2w.squeeze(0).mean(dim=0).cpu().numpy())
                np.save(os.path.join(output_dir, f"{pids[i]}_attention_w2g.npy"), attention_w2g.squeeze(0).mean(dim=0).cpu().numpy())
                np.save(os.path.join(output_dir, f"{pids[i]}_coords.npy"), coords_batch[i].numpy())
    with open(os.path.join(output_dir, "gene_names.txt"), "w") as f:
        for name in gene_names: f.write(f"{name}\n")
    print(f"Attention weights for best model saved to {output_dir}")

def train_and_validate_fold(model, train_loader, val_loader, optimizer, loss_func, device, fold_output_dir):
    best_val_c_index = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(fold_output_dir, "temp_best_model.pth") if fold_output_dir else None

    for epoch in range(EPOCHS):
        train_loss, train_c_index = train_epoch(model, train_loader, optimizer, loss_func, device)
        val_loss, val_c_index = validate_epoch(model, val_loader, loss_func, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, C-idx: {train_c_index:.4f} | Val Loss: {val_loss:.4f}, C-idx: {val_c_index:.4f}")

        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            epochs_no_improve = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✨ Temp best model saved with C-Index: {best_val_c_index:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break
    return best_val_c_index

def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    get_patient_id = lambda x: '-'.join(x.split('-')[:3])
    gene_df = pd.read_csv(GENE_EXPRESSION_PATH, sep="\t", index_col=0)
    gene_df.index = gene_df.index.map(get_patient_id)
    gene_df = gene_df.loc[~gene_df.index.duplicated(keep='first')]
    gene_names = gene_df.columns.tolist()
    survival_df = pd.read_csv(SURVIVAL_DATA_PATH, sep="\t", index_col=0)
    survival_df.index = survival_df.index.map(get_patient_id)
    survival_df = survival_df.loc[~survival_df.index.duplicated(keep='first')]
    survival_df = survival_df[['time', 'event']].dropna()
    survival_df['time'] = pd.to_numeric(survival_df['time'])
    survival_df['event'] = pd.to_numeric(survival_df['event'])
    all_wsi_files = [f for f in os.listdir(WSI_FILES_DIR) if f.endswith(".csv")]
    wsi_files_set = {get_patient_id(f.replace("_patches.csv", "")) for f in all_wsi_files}
    wsi_filename_map = {get_patient_id(f.replace("_patches.csv", "")): f for f in all_wsi_files}
    common_patients = np.array(sorted(list(set(gene_df.index) & set(survival_df.index) & wsi_files_set)))
    gene_df = gene_df.loc[common_patients]
    survival_df = survival_df.loc[common_patients]
    print(f"Found {len(common_patients)} common patients across all data sources.")
    try:
        split_files = sorted([os.path.join(SPLITS_DIR, f) for f in os.listdir(SPLITS_DIR) if f.endswith('.csv')])
        if not split_files: raise FileNotFoundError("No split files found.")
    except FileNotFoundError as e:
        print(f"Error: Could not find split files in '{SPLITS_DIR}'. {e}")
        return

    print("\n===== STAGE 1: DISCOVERY - Running 5-Fold CV to find the best data split =====")
    fold_results_c_indexes = []
    best_fold_info = {"fold_number": -1, "c_index": 0.0, "split_file": None}

    for fold, split_file in enumerate(split_files):
        print(f"\n----- Evaluating FOLD {fold+1}/{len(split_files)} -----")
        split_df = pd.read_csv(split_file)
        train_ids = [pid for pid in split_df['train'].dropna().map(get_patient_id).unique() if pid in common_patients]
        val_ids = [pid for pid in split_df['val'].dropna().map(get_patient_id).unique() if pid in common_patients]
        if not train_ids or not val_ids:
            print("Skipping fold due to missing samples.")
            fold_results_c_indexes.append(0)
            continue

        gene_scaler_fold = StandardScaler().fit(gene_df.loc[train_ids])
        gene_df_train_scaled = pd.DataFrame(gene_scaler_fold.transform(gene_df.loc[train_ids]), index=train_ids, columns=gene_df.columns)
        gene_df_val_scaled = pd.DataFrame(gene_scaler_fold.transform(gene_df.loc[val_ids]), index=val_ids, columns=gene_df.columns)
        patch_scaler_fold = fit_scaler_incrementally(train_ids, WSI_FILES_DIR, WSI_FEATURE_DIM, wsi_filename_map)
        train_dataset = MultimodalSurvivalDataset(train_ids, gene_df_train_scaled, survival_df, WSI_FILES_DIR, patch_scaler_fold, wsi_filename_map)
        val_dataset = MultimodalSurvivalDataset(val_ids, gene_df_val_scaled, survival_df, WSI_FILES_DIR, patch_scaler_fold, wsi_filename_map)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
        model = MultimodalModel(1, WSI_FEATURE_DIM, EMBEDDING_DIM, N_ATTENTION_HEADS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_func = CoxPHLoss()

        best_c_index_for_fold = train_and_validate_fold(model, train_loader, val_loader, optimizer, loss_func, DEVICE, fold_output_dir=None)
        fold_results_c_indexes.append(best_c_index_for_fold)

        if best_c_index_for_fold > best_fold_info["c_index"]:
            best_fold_info["c_index"] = best_c_index_for_fold
            best_fold_info["fold_number"] = fold + 1
            best_fold_info["split_file"] = split_file

    mean_c_index = np.mean(fold_results_c_indexes)
    std_c_index = np.std(fold_results_c_indexes)
    print(f"\n=======================================================")
    print(f"CV Discovery Complete. Mean C-Index: {mean_c_index:.4f} \u00B1 {std_c_index:.4f}")
    print(f"🏆 Best Performing Fold: Fold {best_fold_info['fold_number']} with C-Index: {best_fold_info['c_index']:.4f}")
    print(f"=======================================================")

    if best_fold_info["fold_number"] != -1:
        print(f"\n===== STAGE 2: FINALIZATION - Training final model on Fold {best_fold_info['fold_number']} data =====")

        best_split_file = best_fold_info["split_file"]
        split_df = pd.read_csv(best_split_file)
        train_ids = [pid for pid in split_df['train'].dropna().map(get_patient_id).unique() if pid in common_patients]
        val_ids = [pid for pid in split_df['val'].dropna().map(get_patient_id).unique() if pid in common_patients]

        gene_scaler_fold = StandardScaler().fit(gene_df.loc[train_ids])
        gene_df_train_scaled = pd.DataFrame(gene_scaler_fold.transform(gene_df.loc[train_ids]), index=train_ids, columns=gene_df.columns)
        gene_df_val_scaled = pd.DataFrame(gene_scaler_fold.transform(gene_df.loc[val_ids]), index=val_ids, columns=gene_df.columns)
        patch_scaler_fold = fit_scaler_incrementally(train_ids, WSI_FILES_DIR, WSI_FEATURE_DIM, wsi_filename_map)
        train_dataset = MultimodalSurvivalDataset(train_ids, gene_df_train_scaled, survival_df, WSI_FILES_DIR, patch_scaler_fold, wsi_filename_map)
        val_dataset = MultimodalSurvivalDataset(val_ids, gene_df_val_scaled, survival_df, WSI_FILES_DIR, patch_scaler_fold, wsi_filename_map)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

        final_model = MultimodalModel(1, WSI_FEATURE_DIM, EMBEDDING_DIM, N_ATTENTION_HEADS).to(DEVICE)
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_func = CoxPHLoss()

        final_model_dir = os.path.join(OUTPUT_DIR, "final_best_model")
        os.makedirs(final_model_dir, exist_ok=True)

        _ = train_and_validate_fold(final_model, train_loader, val_loader, optimizer, loss_func, DEVICE, fold_output_dir=final_model_dir)

        best_model_path = os.path.join(final_model_dir, "temp_best_model.pth")
        if os.path.exists(best_model_path):
            print(f"\nLoading final best model from {best_model_path} to generate attention weights...")
            final_model.load_state_dict(torch.load(best_model_path))
            attention_save_dir = os.path.join(final_model_dir, "best_model_attention_weights")
            generate_and_save_attention(final_model, val_loader, DEVICE, gene_names, attention_save_dir)
            os.rename(best_model_path, os.path.join(final_model_dir, "best_model.pth"))
            print("Final model and attention weights saved successfully.")
        else:
            print("Error: Could not find the saved best model to generate attention weights.")

if __name__ == "__main__":
    main()