import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from pycox.evaluation import EvalSurv
from misc.discrete_time import output2hazard, hazard2surv
from misc.loss import CoxPHLoss

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Define the single-head cross-attention model 
class FusionCrossAttention(nn.Module):
    def __init__(self, input_mod1, input_mod2, hidden_dim1, hidden_dim2, output_dim, dropout_prob):
        super(FusionCrossAttention, self).__init__()
        
        # Layers of the neural network       
        if input_mod1 == input_mod2:
            
            self.W_q = torch.nn.Linear(input_mod1, input_mod1)
            self.W_k = torch.nn.Linear(input_mod1, input_mod1)
            self.W_v = torch.nn.Linear(input_mod1, input_mod1)
            self.fc1 = nn.Linear(2*input_mod1, hidden_dim2) 
        else:
            self.hidden_dim1 = hidden_dim1
            self.fc_mod1 = nn.Linear(input_mod1, hidden_dim1)
            self.fc_mod2 = nn.Linear(input_mod2, hidden_dim1)
            
            self.W_q = torch.nn.Linear(hidden_dim1, hidden_dim1)
            self.W_k = torch.nn.Linear(hidden_dim1, hidden_dim1)
            self.W_v = torch.nn.Linear(hidden_dim1, hidden_dim1)
            self.fc1 = nn.Linear(2*hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, modality1, modality2):     
        # Apply fully connected layers to get modality embeddings of equal dimension
        if modality1.shape != modality2.shape:
            modality1 = self.fc_mod1(modality1)
            modality2 = self.fc_mod2(modality2)
        
        mod1_att_mod2 = []
        mod2_att_mod1 = []
        for idx, vector1 in enumerate(modality1):
            # Apply cross-attention between the two modalities (how modality 2 attends to modality 1)
            
            vector1 = vector1.float()
            vector2 = modality2[idx].float()       
            Q1 = self.W_q(vector1).unsqueeze(0)
            K2 = self.W_k(vector2).unsqueeze(0)
            V2 = self.W_v(vector2).unsqueeze(0)                 
            # Calculate attention scores
            scores = torch.matmul(Q1, K2.T) / torch.sqrt(torch.tensor(K2.shape[1], dtype=torch.float32))
            attention_weights = F.softmax(scores, dim=-1)
            
            weighted_sum = torch.matmul(attention_weights, V2)
            mod1_att_mod2.append(weighted_sum)
            
            # Apply cross-attention between the two modalities (how modality 1 attends to modality 2)
            
            Q1 = self.W_q(vector2).unsqueeze(0)
            K2 = self.W_k(vector1).unsqueeze(0)
            V2 = self.W_v(vector1).unsqueeze(0)
            # Calculate attention scores
            scores = torch.matmul(Q1, K2.T) / torch.sqrt(torch.tensor(K2.shape[1], dtype=torch.float32))
            attention_weights = F.softmax(scores, dim=-1)
            # Weighted sum of vector1 based on the attention scores
            weighted_sum = torch.matmul(attention_weights, V2)       
            mod2_att_mod1.append(weighted_sum)
        
        # Fuse two modalities
        mod1_att_mod2 = torch.cat(mod1_att_mod2, axis=0)
        mod2_att_mod1 = torch.cat(mod2_att_mod1, axis=0)
        fused_representation = torch.cat((mod1_att_mod2, mod2_att_mod1), axis=1)

        # Apply fully connected layers with ReLU and Dropout
        x = self.fc1(fused_representation)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
                
        return x

# Define the multi-head cross-attention model
class FusionMultiHeadCrossAttention(nn.Module):
    def __init__(self, input_mod1, input_mod2, hidden_dim1, hidden_dim2, output_dim, dropout_prob, num_heads):
        super(FusionMultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads

        # Layers of the neural network         
        if input_mod1 == input_mod2:
            assert input_mod1 % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.head_dim = input_mod1 // num_heads
            
            self.W_q = torch.nn.Linear(input_mod1, input_mod1)
            self.W_k = torch.nn.Linear(input_mod1, input_mod1)
            self.W_v = torch.nn.Linear(input_mod1, input_mod1)
            
            self.W_o = nn.Linear(input_mod1, input_mod1)
            self.fc1 = nn.Linear(2*input_mod1, hidden_dim2) 
        else:
            self.hidden_dim1 = hidden_dim1
            assert hidden_dim1 % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.fc_mod1 = nn.Linear(input_mod1, hidden_dim1)
            self.fc_mod2 = nn.Linear(input_mod2, hidden_dim1)
            self.head_dim = hidden_dim1 // num_heads
            
            self.W_q = torch.nn.Linear(hidden_dim1, hidden_dim1)
            self.W_k = torch.nn.Linear(hidden_dim1, hidden_dim1)
            self.W_v = torch.nn.Linear(hidden_dim1, hidden_dim1)
            
            self.W_o = nn.Linear(hidden_dim1, hidden_dim1)
            self.fc1 = nn.Linear(2*hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def split_heads(self, x):
        x = x.view(self.num_heads, self.head_dim)
        return x

    def forward(self, modality1, modality2, mask=None):   
        
        if modality1.shape != modality2.shape:
            modality1 = self.fc_mod1(modality1)
            modality2 = self.fc_mod2(modality2)
        mod1_att_mod2 = []
        mod2_att_mod1 = []
        for idx, vector1 in enumerate(modality1):
                        
            vector1 = vector1.float()
            vector2 = modality2[idx].float()       
            Q1 = self.W_q(vector1).unsqueeze(0)
            K2 = self.W_k(vector2).unsqueeze(0)
            V2 = self.W_v(vector2).unsqueeze(0)           
            # Split heads
            Q1 = self.split_heads(Q1)
            K2 = self.split_heads(K2)
            V2 = self.split_heads(V2)           
            
            scores = torch.matmul(Q1, K2.T) / (self.head_dim ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            
            # Apply softmax to obtain attention weights
            attention_weights = F.softmax(scores, dim=-1)
            
            weighted_sum = torch.matmul(attention_weights, V2)
            
            weighted_sum = weighted_sum.contiguous().view(-1, modality1.shape[1])
            
            mod1_att_mod2.append(self.W_o(weighted_sum))
            
            
            
            Q1 = self.W_q(vector2).unsqueeze(0)
            K2 = self.W_k(vector1).unsqueeze(0)
            V2 = self.W_v(vector1).unsqueeze(0) 
            # Split heads
            Q1 = self.split_heads(Q1)
            K2 = self.split_heads(K2)
            V2 = self.split_heads(V2)
            
            scores = torch.matmul(Q1, K2.T) / (self.head_dim ** 0.5) 
            # Apply softmax to obtain attention weights
            attention_weights = F.softmax(scores, dim=-1)
            
            weighted_sum = torch.matmul(attention_weights, V2)
            
            weighted_sum = weighted_sum.contiguous().view(-1, modality1.shape[1])
           
            mod2_att_mod1.append(self.W_o(weighted_sum))
        # Fuse two modalities
        mod1_att_mod2 = torch.cat(mod1_att_mod2, axis=0)
        mod2_att_mod1 = torch.cat(mod2_att_mod1, axis=0)
        fused_representation = torch.cat((mod1_att_mod2, mod2_att_mod1), axis=1)

        # Apply fully connected layers with ReLU and Dropout
        x = self.fc1(fused_representation)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def output2surv(output: Tensor, epsilon: float = 1e-7) -> Tensor:
    hazards = output2hazard(output)
    return hazards, hazard2surv(hazards, epsilon)

def make_mlp(in_features: int, out_features: int) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, out_features)
    )
    return net

def train(x_train, x_test, net, epochs, optimizer, scheduler, loss_func, device):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(x_train):
            x, duration, event = data
            x = x.to(device=device)
            duration = duration.to(device=device)
            event = event.to(device=device)
            optimizer.zero_grad()
            output = net(x)
            loss = loss_func(output, duration, event)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step() 

    net.eval()
    with torch.no_grad():
        x_test = x_test.to(device=device)
        output = net(x_test)
        hazards, surv = output2surv(output)
    surv = surv.cpu()
    
    return surv

def train_fusion(x1_train, x1_test, x2_train, x2_test, y_train, epochs, batch_size, learning_rate, device, fusion_type):
    input_mod1 = x1_train.shape[1] 
    input_mod2 = x2_train.shape[1]
    hidden_dim1 = 512
    hidden_dim2 = 512
    output_dim = 1
    dropout_prob = 0.4
    num_heads = 8
    
    y_train_duration = torch.from_numpy(y_train[0])
    y_train_event = torch.from_numpy(y_train[1])
    train_dataset = TensorDataset(x1_train, y_train_duration, y_train_event, x2_train)
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    
    if fusion_type.lower() == 'single-head':
        net = FusionCrossAttention(input_mod1, input_mod2, hidden_dim1, hidden_dim2, output_dim, dropout_prob)
    elif fusion_type.lower() == 'multi-head':
        net = FusionMultiHeadCrossAttention(input_mod1, input_mod2, hidden_dim1, hidden_dim2, output_dim, dropout_prob, num_heads)
    else:
        raise Exception("Fusion type can be 'single-head' or 'multi-head' attention.")
    
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    loss_func = CoxPHLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dl):
            x1, duration, event, x2 = data
            x1 = x1.to(device=device)
            x2 = x2.to(device=device)
            duration = duration.to(device=device)
            event = event.to(device=device)
            optimizer.zero_grad()
            output = net(x1, x2)
            loss = loss_func(output, duration, event)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step() 

    net.eval()
    with torch.no_grad():
        x1 = x1_test.to(device=device)
        x2 = x2_test.to(device=device)
        output = net(x1, x2)
        hazards, surv = output2surv(output)
    surv = surv.cpu()
    
    return surv

def standardize(X_train, X_test):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std

def strat_k_fold(X1, X2, y, n_splits, seed):   
    kf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    c_index_X1 = []
    c_index_X2 = []
    c_index_ef = []
    c_index_ca = []
    c_index_mhca = []
    cnt = 0
        
    for train_index, test_index in tqdm(kf.split(X1, y[:,1])):
        cnt += 1
        X1_train, X1_test = X1[train_index], X1[test_index]
        X2_train, X2_test = X2[train_index], X2[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = (y_train[:,0].astype(np.float32), y_train[:,1].astype(np.int32))
        y_test_duration, y_test_event = y_test[:,0], y_test[:,1]

        X1_train, X1_test = standardize(X1_train, X1_test)   
        X2_train, X2_test = standardize(X2_train, X2_test) 
        X_ef_train = np.hstack((X1_train, X2_train))
        X_ef_test = np.hstack((X1_test, X2_test))
        X1_train, X1_test = torch.from_numpy(X1_train), torch.from_numpy(X1_test)
        X2_train, X2_test = torch.from_numpy(X2_train), torch.from_numpy(X2_test)
        X_ef_train, X_ef_test = torch.from_numpy(X_ef_train), torch.from_numpy(X_ef_test)

        batch_size = X1_train.shape[0]
        epochs = 15
        learning_rate = 2.5e-3
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_func = CoxPHLoss()

        # mRNA modality
        in_features_X1 = X1_train.shape[1]
        out_features = 1
        net_X1 = make_mlp(in_features_X1, out_features).to(device)
        optimizer_X1 = torch.optim.AdamW(net_X1.parameters(), lr=learning_rate)
        scheduler_X1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_X1, T_max=epochs, eta_min=0)
        y_train_duration = torch.from_numpy(y_train[0])
        y_train_event = torch.from_numpy(y_train[1])
        train_X1_dataset = TensorDataset(X1_train, y_train_duration, y_train_event)
        train_X1_dataloader = DataLoader(train_X1_dataset, batch_size, shuffle=True)
        surv_X1 = train(train_X1_dataloader, X1_test, net_X1, epochs, optimizer_X1, scheduler_X1, loss_func, device)
        surv_X1_df = pd.DataFrame(surv_X1.numpy().transpose())
        ev_X1 = EvalSurv(surv_X1_df, y_test_duration, y_test_event) 

        # WSI modality
        in_features_X2 = X2_train.shape[1]
        net_X2 = make_mlp(in_features_X2, out_features).to(device)
        optimizer_X2 = torch.optim.AdamW(net_X2.parameters(), lr=learning_rate)
        scheduler_X2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_X2, T_max=epochs, eta_min=0)
        train_X2_dataset = TensorDataset(X2_train, y_train_duration, y_train_event)
        train_X2_dataloader = DataLoader(train_X2_dataset, batch_size, shuffle=True)
        surv_X2 = train(train_X2_dataloader, X2_test, net_X2, epochs, optimizer_X2, scheduler_X2, loss_func, device)
        surv_X2_df = pd.DataFrame(surv_X2.numpy().transpose())
        ev_X2 = EvalSurv(surv_X2_df, y_test_duration, y_test_event)

        # Early fusion
        in_features_ef = X_ef_train.shape[1]
        net_ef = make_mlp(in_features_ef, out_features).to(device)
        optimizer_ef = torch.optim.AdamW(net_ef.parameters(), lr=learning_rate)
        scheduler_ef = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ef, T_max=epochs, eta_min=0)
        train_ef_dataset = TensorDataset(X_ef_train, y_train_duration, y_train_event)
        train_ef_dataloader = DataLoader(train_ef_dataset, batch_size, shuffle=True)
        surv_ef = train(train_ef_dataloader, X_ef_test, net_ef, epochs, optimizer_ef, scheduler_ef, loss_func, device)
        surv_ef_df = pd.DataFrame(surv_ef.numpy().transpose())
        ev_ef = EvalSurv(surv_ef_df, y_test_duration, y_test_event)

        # Single-head cross-attention
        surv_ca = train_fusion(X1_train, X1_test, X2_train, X2_test, y_train, epochs, batch_size, learning_rate, device, fusion_type='single-head')
        surv_ca_df = pd.DataFrame(surv_ca.numpy().transpose())
        ev_ca = EvalSurv(surv_ca_df, y_test_duration, y_test_event)

        # Multi-head cross-attention
        surv_mhca = train_fusion(X1_train, X1_test, X2_train, X2_test, y_train, epochs, batch_size, learning_rate, device, fusion_type='multi-head')
        surv_mhca_df = pd.DataFrame(surv_mhca.numpy().transpose())
        ev_mhca = EvalSurv(surv_mhca_df, y_test_duration, y_test_event)

        print("\nC-index (mRNA): %.3f" % (ev_X1.concordance_td('antolini')))
        print("C-index (WSI): %.3f" % (ev_X2.concordance_td('antolini')))
        print("C-index (early fusion): %.3f" % (ev_ef.concordance_td('antolini')))
        print("C-index (single-head cross-attention): %.3f" % (ev_ca.concordance_td('antolini')))
        print("C-index (multi-head cross-attention): %.3f\n" % (ev_mhca.concordance_td('antolini')))
        
        c_index_X1.append(ev_X1.concordance_td('antolini'))
        c_index_X2.append(ev_X2.concordance_td('antolini'))
        c_index_ef.append(ev_ef.concordance_td('antolini'))
        c_index_ca.append(ev_ca.concordance_td('antolini'))
        c_index_mhca.append(ev_mhca.concordance_td('antolini'))

    return c_index_X1, c_index_X2, c_index_ef, c_index_ca, c_index_mhca

# =======================================================
# --- Configuration: UPDATE THESE VALUES BEFORE RUNNING ---
# =======================================================
# 1. Directory where your data files are located.
DATA_DIR = 'path/to/your/data/directory'

# 2. The dataset identifier (e.g., 'BLCA'). This is used to build the filenames.
dataset = 'BLCA'

# 3. The embedding type identifier (e.g., 'ICA'). This is also used for filenames.
embedding = 'ICA'

# 4. Random seed for reproducibility.
seed = 10
# =======================================================

# Set random seed for reproducibility
torch.set_num_threads(1)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False 

# Construct file paths
mrna_path = os.path.join(DATA_DIR, f'integr_{embedding}_mRNA_{dataset}.tsv')
labels_path = os.path.join(DATA_DIR, f'integr_survival_{dataset}.tsv')
wsi_path = os.path.join(DATA_DIR, f'integr_{embedding}_WSI_{dataset}.tsv')

# Load mRNA data
mRNA = pd.read_csv(mrna_path, delimiter='\t', index_col=0).T
labels = pd.read_csv(labels_path, delimiter='\t', index_col=None, usecols=['event','time'])
labels = labels[['time','event']]
removed_ind = labels.index[labels.isnull().any(axis=1)].tolist()
labels = labels.dropna()
labels = labels.values
mRNA = mRNA.drop(mRNA.index[removed_ind])
mRNA_features = mRNA.values.astype(np.float32)

# Load WSI data
WSI = pd.read_csv(wsi_path, delimiter='\t', index_col=0).T 
WSI = WSI.drop(WSI.index[removed_ind])
WSI_features = WSI.values.astype(np.float32)

# Ensure number of samples match
num_samples = mRNA_features.shape[0]
assert num_samples == WSI_features.shape[0], "Number of samples must be the same for mRNA and WSI matrices"

n_splits = 5
c_index_mRNA, c_index_WSI, c_index_ef, c_index_ca, c_index_mhca = strat_k_fold(mRNA_features, WSI_features, labels, n_splits, seed)

# Print performance averaged over folds
print('\n############')
print(f'{dataset}-{embedding}')
print('############')
print('Average c-index (mRNA): %.3f (%.2f)' % (np.mean(c_index_mRNA), np.std(c_index_mRNA)))
print('Average c-index (WSI): %.3f (%.2f)' % (np.mean(c_index_WSI), np.std(c_index_WSI)))
print('Average c-index (early fusion): %.3f (%.2f)' % (np.mean(c_index_ef), np.std(c_index_ef)))
print('Average c-index (single-head cross-attention): %.3f (%.2f)' % (np.mean(c_index_ca), np.std(c_index_ca)))

print('Average c-index (multi-head cross-attention): %.3f (%.2f)' % (np.mean(c_index_mhca), np.std(c_index_mhca)))
