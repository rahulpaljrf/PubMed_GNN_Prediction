# =========================================================
# 04_train_gnn_multilabel.py
# =========================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import trange

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "/root/PubMed_GNN_Prediction/scripts/data"
EMB_PATH = f"{DATA_DIR}/embeddings/patient_embeddings.pt"
EDGE_INDEX_PATH = f"{DATA_DIR}/graph/edge_index.pt"
EDGE_WEIGHT_PATH = f"{DATA_DIR}/graph/edge_weight.pt"
LABEL_PATH = f"{DATA_DIR}/labels/labels.pt"

MODEL_SAVE_PATH = "models/gnn_patient_classifier.pt"

HIDDEN_DIM = 256
EPOCHS = 100
LR = 1e-3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# LOAD DATA
# -------------------------
def load_embeddings(path):
    if os.path.isdir(path):
        parts = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')])
        if not parts:
            raise FileNotFoundError(f"No .pt files found in {path}")
        X_list = [torch.load(p) for p in parts]
        return torch.cat(X_list, dim=0)
    else:
        return torch.load(path)

# If EMB_PATH file doesn't exist, try the embeddings directory containing part files
if not os.path.exists(EMB_PATH) and os.path.isdir(f"{DATA_DIR}/embeddings"):
    EMB_PATH = f"{DATA_DIR}/embeddings"

X = load_embeddings(EMB_PATH)                 # (N, 768) or directory of parts
edge_index = torch.load(EDGE_INDEX_PATH)      # (2, E)
edge_weight = torch.load(EDGE_WEIGHT_PATH)
Y = torch.load(LABEL_PATH)                    # (N, C)

X = F.normalize(X, p=2, dim=1)

N, in_dim = X.shape
num_classes = Y.size(1)

print("Nodes:", N)
print("Input dim:", in_dim)
print("Classes:", num_classes)

# -------------------------
# BUILD PyG DATA OBJECT
# -------------------------
data = Data(
    x=X,
    edge_index=edge_index,
    edge_attr=edge_weight,
    y=Y
)

# -------------------------
# TRAIN / VAL / TEST SPLIT
# -------------------------
perm = torch.randperm(N)

train_end = int(TRAIN_RATIO * N)
val_end = int((TRAIN_RATIO + VAL_RATIO) * N)

train_idx = perm[:train_end]
val_idx = perm[train_end:val_end]
test_idx = perm[val_end:]

data.train_mask = torch.zeros(N, dtype=torch.bool)
data.val_mask = torch.zeros(N, dtype=torch.bool)
data.test_mask = torch.zeros(N, dtype=torch.bool)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True

data = data.to(device)

# -------------------------
# GNN MODEL
# -------------------------
class PatientGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x   # raw logits

model = PatientGNN(in_dim, HIDDEN_DIM, num_classes).to(device)

# -------------------------
# OPTIMIZATION
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# -------------------------
# TRAINING FUNCTIONS
# -------------------------
def train():
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits)

    preds = (probs[mask] > 0.5).int().cpu()
    true = data.y[mask].int().cpu()

    return f1_score(true, preds, average="micro")


# -------------------------
# TRAIN LOOP
# -------------------------
best_val_f1 = 0.0

for epoch in trange(1, EPOCHS + 1):
    loss = train()
    val_f1 = evaluate(data.val_mask)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Val Micro-F1 {val_f1:.4f}"
        )

# -------------------------
# FINAL TEST
# -------------------------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_f1 = evaluate(data.test_mask)

print("=" * 60)
print("Best Val Micro-F1:", best_val_f1)
print("Test Micro-F1:", test_f1)
print("Model saved to:", MODEL_SAVE_PATH)
print("=" * 60)
