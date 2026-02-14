import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CONFIG — match training settings
DATA_DIR = "/root/PubMed_GNN_Prediction/scripts/data"
EMB_PATH = f"{DATA_DIR}/embeddings/patient_embeddings.pt"
EDGE_INDEX_PATH = f"{DATA_DIR}/graph/edge_index.pt"
EDGE_WEIGHT_PATH = f"{DATA_DIR}/graph/edge_weight.pt"
LABEL_PATH = f"{DATA_DIR}/labels/labels.pt"
MODEL_PATH = "/root/PubMed_GNN_Prediction/models/gnn_patient_classifier.pt"
HIDDEN_DIM = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(path):
    if os.path.isdir(path):
        parts = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')])
        if not parts:
            raise FileNotFoundError(f"No .pt files found in {path}")
        X_list = [torch.load(p) for p in parts]
        return torch.cat(X_list, dim=0)
    else:
        return torch.load(path)

# If the single-file embedding path doesn't exist, fall back to the embeddings directory
if not os.path.exists(EMB_PATH) and os.path.isdir(f"{DATA_DIR}/embeddings"):
    EMB_PATH = f"{DATA_DIR}/embeddings"

X = load_embeddings(EMB_PATH)
edge_index = torch.load(EDGE_INDEX_PATH)
edge_weight = torch.load(EDGE_WEIGHT_PATH)
Y = torch.load(LABEL_PATH)

X = F.normalize(X, p=2, dim=1)

N, in_dim = X.shape

class PatientGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Build data object
data = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=Y)

# Deterministic split (seeded) — may differ from original training split
torch.manual_seed(42)
perm = torch.randperm(N)
train_end = int(0.70 * N)
val_end = int(0.85 * N)
train_idx = perm[:train_end]
val_idx = perm[train_end:val_end]
test_idx = perm[val_end:]

data.train_mask = torch.zeros(N, dtype=torch.bool)
data.val_mask = torch.zeros(N, dtype=torch.bool)
data.test_mask = torch.zeros(N, dtype=torch.bool)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True

num_classes = Y.size(1)

# Load model
model = PatientGNN(in_dim, HIDDEN_DIM, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@torch.no_grad()
def get_test_predictions():
    logits = model(data.x.to(device), data.edge_index.to(device))
    probs = torch.sigmoid(logits)

    preds = (probs[data.test_mask.to(device)] > 0.5).int().cpu()
    true  = data.y[data.test_mask].int().cpu()

    return true, preds


y_true, y_pred = get_test_predictions()

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    average=None   # per-label
)

label_names = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "L", "M", "N", "Z"
]

df_metrics = pd.DataFrame({
    "Label": label_names,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "Support": support
})

print(df_metrics)

os.makedirs("results", exist_ok=True)
df_metrics.to_csv("results/per_label_f1.csv", index=False)

plt.figure(figsize=(10, 5))
plt.bar(df_metrics["Label"], df_metrics["F1-score"])
plt.ylabel("F1-score")
plt.xlabel("MeSH Root Label")
plt.title("Per-label F1-score (Test Set)")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/per_label_f1.png", dpi=300)
plt.close()
