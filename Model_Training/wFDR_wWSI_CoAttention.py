import os
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

try:
    from misc.loss import CoxPHLoss
except ImportError:
    print("Warning: 'misc' library not found. Using a basic CoxPHLoss implementation.")
    class CoxPHLoss(nn.Module):
        def forward(self, log_risks, times, events):
            events = events.bool()
            if not torch.any(events):
                return torch.tensor(0.0, device=log_risks.device, requires_grad=True)
            log_risks_observed = log_risks[events]
            log_sum_exp_risk_set = torch.log(torch.cumsum(torch.exp(log_risks), dim=0))[events]
            loss = - (log_risks_observed - log_sum_exp_risk_set).sum() / events.sum()
            return loss


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 5

# Paths (replace with your own)
PATHWAY_SCORES_PATH = "<PATH_TO_PATHWAY_SCORES_TSV>"
SURVIVAL_DATA_PATH = "<PATH_TO_SURVIVAL_DATA_TSV>"
WSI_FILES_DIR = "<PATH_TO_WSI_CSV_FILES>"
OUTPUT_DIR = "<OUTPUT_DIR>"
SPLITS_DIR = "<PATH_TO_SPLITS_DIR>"

WSI_FEATURE_DIM = 512
PATCH_NAME_COL_WSI = 0
FEATURE_START_COL_WSI = 1

EMBEDDING_DIM = 256
N_ATTENTION_HEADS = 4


class CrossAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        assert embedding_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.to_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.to_kv = nn.Linear(embedding_dim, embedding_dim * 2, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, query_embed: Tensor, context_embed: Tensor):
        q = self.to_q(query_embed)
        k, v = self.to_kv(context_embed).chunk(2, dim=-1)
        q = q.view(*q.shape[:-1], self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(*k.shape[:-1], self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(*v.shape[:-1], self.n_heads, self.head_dim).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(dots, dim=-1)
        attended_features = torch.matmul(attention_weights, v)
        attended_features = attended_features.transpose(1, 2).contiguous().view(*query_embed.shape)
        return attended_features, attention_weights


class MultimodalModel(nn.Module):
    def __init__(self, pathway_in_dim: int, patch_in_dim: int, embed_dim: int, n_heads: int):
        super().__init__()
        self.pathway_encoder = nn.Sequential(nn.Linear(pathway_in_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.patch_encoder = nn.Sequential(nn.Linear(patch_in_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.cross_attention_p2w = CrossAttention(embed_dim, n_heads)
        self.cross_attention_w2p = CrossAttention(embed_dim, n_heads)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, pathway_data: Tensor, patch_data: Tensor):
        pathway_embed_per_pathway = self.pathway_encoder(pathway_data.unsqueeze(-1))
        patch_embed = self.patch_encoder(patch_data)
        pathway_embed_b = pathway_embed_per_pathway.unsqueeze(0)
        patch_embed_b = patch_embed.unsqueeze(0)
        attended_pathways, attention_p2w = self.cross_attention_p2w(pathway_embed_b, patch_embed_b)
        attended_patches, attention_w2p = self.cross_attention_w2p(patch_embed_b, pathway_embed_b)
        aggregated_pathways = attended_pathways.mean(dim=1)
        aggregated_patches = attended_patches.mean(dim=1)
        fused_features = torch.cat((aggregated_pathways, aggregated_patches), dim=1)
        log_risk = self.regressor(fused_features)
        return log_risk.squeeze(0), attention_p2w.squeeze(0), attention_w2p.squeeze(0)


def parse_coords_from_name(patch_name):
    try:
        name_without_ext = os.path.splitext(patch_name)[0]
        parts = name_without_ext.split('_')
        y_coord, x_coord = int(parts[-1]), int(parts[-2])
        return x_coord, y_coord
    except (IndexError, ValueError):
        return 0, 0


class MultimodalSurvivalDataset(Dataset):
    def __init__(self, patient_ids, pathway_df, survival_df, wsi_dir, patch_scaler, wsi_filename_map):
        self.patient_ids, self.pathway_df, self.survival_df = patient_ids, pathway_df, survival_df
        self.wsi_dir, self.patch_scaler, self.wsi_filename_map = wsi_dir, patch_scaler, wsi_filename_map

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        pathway_features = self.pathway_df.loc[patient_id].values.astype(np.float32)
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
            "pathways": torch.from_numpy(pathway_features),
            "patches": torch.from_numpy(patch_features.astype(np.float32)),
            "coords": torch.tensor(coords, dtype=torch.long),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32)
        }


def custom_collate_fn_multimodal(batch):
    patient_ids = [item['patient_id'] for item in batch]
    pathways_batch = torch.stack([item['pathways'] for item in batch])
    patches_batch = [item['patches'] for item in batch]
    coords_batch = [item['coords'] for item in batch]
    times_batch = torch.stack([item['time'] for item in batch])
    events_batch = torch.stack([item['event'] for item in batch])
    return (patient_ids, pathways_batch, patches_batch, coords_batch, times_batch, events_batch)


def c_index_manual(log_risks, events, times):
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


def fit_scaler_incrementally(patient_ids, wsi_dir, feature_dim, wsi_filename_map):
    scaler = StandardScaler()
    for pid in patient_ids:
        file_path = os.path.join(wsi_dir, wsi_filename_map[pid])
        try:
            patch_data = pd.read_csv(file_path).iloc[:, FEATURE_START_COL_WSI : FEATURE_START_COL_WSI + feature_dim].values
            if patch_data.size > 0:
                scaler.partial_fit(patch_data)
        except FileNotFoundError:
            print(f"Warning: missing file skipped {file_path}")
    return scaler


def train_epoch(model, dataloader, optimizer, loss_func, device):
    model.train()
    total_loss, all_risks, all_times, all_events = 0, [], [], []
    for _, pathway_batch, patches_batch, _, times_batch, events_batch in dataloader:
        pathway_batch, times_batch, events_batch = pathway_batch.to(device), times_batch.to(device), events_batch.to(device)
        optimizer.zero_grad()
        patient_risks_in_batch = [model(pathway_batch[i], p.to(device))[0] for i, p in enumerate(patches_batch)]
        if not patient_risks_in_batch:
            continue
        patient_risks_tensor = torch.cat(patient_risks_in_batch)
        perm = torch.argsort(times_batch, descending=True)
        sorted_risks, sorted_times, sorted_events = patient_risks_tensor[perm], times_batch[perm], events_batch[perm]
        loss = loss_func(sorted_risks, sorted_times, sorted_events)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        all_risks.append(patient_risks_tensor.detach())
        all_times.append(times_batch)
        all_events.append(events_batch)
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    if not all_risks:
        return avg_loss, 0.5
    c_idx = c_index_manual(torch.cat(all_risks), torch.cat(all_events), torch.cat(all_times))
    return avg_loss, c_idx


def validate_epoch(model, dataloader, loss_func, device):
    model.eval()
    total_loss, all_risks, all_times, all_events = 0, [], [], []
    with torch.no_grad():
        for _, pathway_batch, patches_batch, _, times_batch, events_batch in dataloader:
            pathway_batch, times_batch, events_batch = pathway_batch.to(device), times_batch.to(device), events_batch.to(device)
            patient_risks_in_batch = []
            for i in range(len(patches_batch)):
                pathway_data, patch_data = pathway_batch[i], patches_batch[i].to(device)
                log_risk, _, _ = model(pathway_data, patch_data)
                patient_risks_in_batch.append(log_risk)
            if not patient_risks_in_batch:
                continue
            patient_risks_tensor = torch.cat(patient_risks_in_batch)
            perm = torch.argsort(times_batch, descending=True)
            loss = loss_func(patient_risks_tensor[perm], times_batch[perm], events_batch[perm])
            if not torch.isnan(loss):
                total_loss += loss.item()
            all_risks.append(patient_risks_tensor)
            all_times.append(times_batch)
            all_events.append(events_batch)
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    if not all_risks:
        return avg_loss, 0.5
    c_idx = c_index_manual(torch.cat(all_risks), torch.cat(all_events), torch.cat(all_times))
    return avg_loss, c_idx


def train_and_validate_fold(model, train_loader, val_loader, optimizer, loss_func, device, fold_output_dir=None):
    best_val_c_index = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(fold_output_dir, "temp_best_model.pth") if fold_output_dir else None

    for epoch in range(EPOCHS):
        train_loss, train_c = train_epoch(model, train_loader, optimizer, loss_func, device)
        val_loss, val_c = validate_epoch(model, val_loader, loss_func, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, C-idx: {train_c:.4f} | Val Loss: {val_loss:.4f}, C-idx: {val_c:.4f}")
        if val_c > best_val_c_index:
            best_val_c_index = val_c
            epochs_no_improve = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break
    return best_val_c_index


def generate_and_save_attention(model, dataloader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for pids, pathway_batch, patches_batch, coords_batch, _, _ in dataloader:
            for i in range(len(patches_batch)):
                patient_id = pids[i]
                pathway_data = pathway_batch[i].to(device)
                patch_data = patches_batch[i].to(device)
                _, attention_p2w, attention_w2p = model(pathway_data, patch_data)
                np.save(os.path.join(output_dir, f"{patient_id}_attention_p2w.npy"), attention_p2w.mean(dim=0).cpu().numpy())
                np.save(os.path.join(output_dir, f"{patient_id}_attention_w2p.npy"), attention_w2p.mean(dim=0).cpu().numpy())
                np.save(os.path.join(output_dir, f"{patient_id}_coords.npy"), coords_batch[i].numpy())


def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    get_patient_id = lambda x: '-'.join(x.split('-')[:3])
    pathway_df = pd.read_csv(PATHWAY_SCORES_PATH, sep="\t", index_col=0)
    pathway_df.index = pathway_df.index.map(get_patient_id)
    pathway_df = pathway_df.loc[~pathway_df.index.duplicated(keep='first')]

    survival_df = pd.read_csv(SURVIVAL_DATA_PATH, sep="\t")
    survival_df = survival_df.rename(columns={survival_df.columns[0]: 'patient_id'})
    survival_df['patient_id'] = survival_df['patient_id'].map(get_patient_id)
    survival_df = survival_df.set_index('patient_id').loc[~survival_df.index.duplicated(keep='first')]
    survival_df = survival_df[["time", "event"]].dropna()

    all_wsi_files = [f for f in os.listdir(WSI_FILES_DIR) if f.endswith(".csv")]
    wsi_files_set = {get_patient_id(f.replace("_patches.csv", "")) for f in all_wsi_files}
    common_patients = np.array(sorted(list(set(pathway_df.index) & set(survival_df.index) & wsi_files_set)))
    wsi_filename_map = {get_patient_id(f.replace("_patches.csv", "")): f for f in all_wsi_files}
    pathway_df, survival_df = pathway_df.loc[common_patients], survival_df.loc[common_patients]
    split_files = sorted([os.path.join(SPLITS_DIR, f) for f in os.listdir(SPLITS_DIR) if f.endswith('.csv')])

    fold_results_c_indexes = []
    best_fold_info = {"fold_number": -1, "c_index": 0.0, "split_file": None}

    for fold, split_file in enumerate(split_files):
        print(f"\nFold {fold+1}/{len(split_files)}")
        split_df = pd.read_csv(split_file)
        train_ids = [pid for pid in split_df['train'].dropna().unique() if pid in common_patients]
        val_ids = [pid for pid in split_df['val'].dropna().unique() if pid in common_patients]

        pathway_scaler_fold = StandardScaler().fit(pathway_df.loc[train_ids])
        pathway_df_train_scaled = pd.DataFrame(pathway_scaler_fold.transform(pathway_df.loc[train_ids]), index=train_ids, columns=pathway_df.columns)
        pathway_df_val_scaled = pd.DataFrame(pathway_scaler_fold.transform(pathway_df.loc[val_ids]), index=val_ids, columns=pathway_df.columns)
        patch_scaler_fold = fit_scaler_incrementally(train_ids, WSI_FILES_DIR, WSI_FEATURE_DIM, wsi_filename_map)

        train_dataset = MultimodalSurvivalDataset(train_ids, pathway_df_train_scaled, survival_df, WSI_FILES_DIR, patch_scaler_fold, wsi_filename_map)
        val_dataset = MultimodalSurvivalDataset(val_ids, pathway_df_val_scaled, survival_df, WSI_FILES_DIR, patch_scaler_fold, wsi_filename_map)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_multimodal)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_multimodal)

        model = MultimodalModel(1, WSI_FEATURE_DIM, EMBEDDING_DIM, N_ATTENTION_HEADS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_func = CoxPHLoss()

        best_c_index_for_fold = train_and_validate_fold(model, train_loader, val_loader, optimizer, loss_func, DEVICE)
        fold_results_c_indexes.append(best_c_index_for_fold)
        print(f"Fold {fold+1} Best Val C-Index: {best_c_index_for_fold:.4f}")

        if best_c_index_for_fold > best_fold_info["c_index"]:
            best_fold_info.update({"fold_number": fold+1, "c_index": best_c_index_for_fold, "split_file": split_file})

    mean_c_index, std_c_index = np.mean(fold_results_c_indexes), np.std(fold_results_c_indexes)
    print(f"\nCross-validation Results: Mean C-Index = {mean_c_index:.4f} ± {std_c_index:.4f}")
    print(f"Best Fold: {best_fold_info['fold_number']} | C-Index = {best_fold_info['c_index']:.4f}")

    best_split_file = best_fold_info["split_file"]
    split_df = pd.read_csv(best_split_file)
    best_train_ids = [pid for pid in split_df['train'].dropna().unique() if pid in common_patients]
    best_val_ids = [pid for pid in split_df['val'].dropna().unique() if pid in common_patients]

    pathway_scaler = StandardScaler().fit(pathway_df.loc[best_train_ids])
    pathway_df_train_scaled = pd.DataFrame(pathway_scaler.transform(pathway_df.loc[best_train_ids]), index=best_train_ids, columns=pathway_df.columns)
    pathway_df_val_scaled = pd.DataFrame(pathway_scaler.transform(pathway_df.loc[best_val_ids]), index=best_val_ids, columns=pathway_df.columns)
    patch_scaler = fit_scaler_incrementally(best_train_ids, WSI_FILES_DIR, WSI_FEATURE_DIM, wsi_filename_map)

    best_train_dataset = MultimodalSurvivalDataset(best_train_ids, pathway_df_train_scaled, survival_df, WSI_FILES_DIR, patch_scaler, wsi_filename_map)
    best_val_dataset = MultimodalSurvivalDataset(best_val_ids, pathway_df_val_scaled, survival_df, WSI_FILES_DIR, patch_scaler, wsi_filename_map)
    best_train_loader = DataLoader(best_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_multimodal)
    best_val_loader = DataLoader(best_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_multimodal)

    final_model = MultimodalModel(1, WSI_FEATURE_DIM, EMBEDDING_DIM, N_ATTENTION_HEADS).to(DEVICE)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_loss_func = CoxPHLoss()

    final_c_index = train_and_validate_fold(final_model, best_train_loader, best_val_loader, final_optimizer, final_loss_func, DEVICE)
    print(f"Final training on best fold completed. Best validation C-Index: {final_c_index:.4f}")

    attention_output_dir = os.path.join(OUTPUT_DIR, "attention_results")
    generate_and_save_attention(final_model, best_val_loader, DEVICE, attention_output_dir)
    print("Attention weights saved.")


if __name__ == "__main__":
    main()
