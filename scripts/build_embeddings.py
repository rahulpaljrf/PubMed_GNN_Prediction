import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "models/biobert-base-cased-v1.2"
ARTICLE_PATH = "/root/PubMed_GNN_Prediction/scripts/data/raw/Article_train.pkl"
SAVE_DIR = "/root/PubMed_GNN_Prediction/scripts/data/embeddings"
CHUNK_SIZE = 128
SAVE_EVERY = 500
NUM_HEADS = 8
EMBED_DIM = 768

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# LOAD DATA
# -------------------------
with open(ARTICLE_PATH, "rb") as f:
    Article_train = pickle.load(f)

print("Number of abstracts:", len(Article_train))

# -------------------------
# LOAD MODEL
# -------------------------
def load_tokenizer_and_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, local_files_only=True
        )
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        print("Loaded tokenizer and model from local cache.")
    except Exception as e:
        print("Local load failed:", e)
        # If MODEL_PATH is not a local directory, avoid blind online attempts
        if not os.path.isdir(model_path):
            print("Model path is not a local directory. Attempting online download...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=False)
                model = AutoModel.from_pretrained(model_path, local_files_only=False)
            except Exception as e2:
                print("Online download failed:", e2)
                raise RuntimeError(
                    f"Could not load model from local path '{model_path}' and online download failed.\n"
                    "Either place the pretrained model files under that path, set MODEL_PATH to a valid HuggingFace repo id, or enable internet access."
                ) from e2
        else:
            # model_path exists but local loading still failed
            raise
    return tokenizer, model

tokenizer, bert = load_tokenizer_and_model(MODEL_PATH)
bert = bert.to(device)
bert.eval()
for p in bert.parameters():
    p.requires_grad = False

# -------------------------
# UTILITIES
# -------------------------
def chunk_tokens(input_ids, attention_mask, chunk_size=128):
    chunks = []
    for start in range(0, input_ids.size(0), chunk_size):
        end = start + chunk_size
        chunks.append({
            "input_ids": input_ids[start:end],
            "attention_mask": attention_mask[start:end]
        })
    return chunks


@torch.no_grad()
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_chunks(chunks):
    embeddings = []
    for c in chunks:
        input_ids = c["input_ids"].unsqueeze(0).to(device)
        attention_mask = c["attention_mask"].unsqueeze(0).to(device)

        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings.append(pooled.squeeze(0))

    return torch.stack(embeddings)  # (n_chunks, 768)


# -------------------------
# CHUNK-LEVEL MHA
# -------------------------
class ChunkLevelMHA(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, B_i):
        B_i = B_i.unsqueeze(0)            # (1, n, 768)
        attn_out, _ = self.mha(B_i, B_i, B_i)
        attn_out = self.proj(attn_out)
        return attn_out.mean(dim=1).squeeze(0)  # (768,)


chunk_mha = ChunkLevelMHA(EMBED_DIM, NUM_HEADS).to(device)
chunk_mha.eval()

# -------------------------
# MAIN LOOP (SAFE)
# -------------------------
buffer = []
part_id = 0

for idx, text in enumerate(tqdm(Article_train, desc="Embedding patients")):
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        chunks = chunk_tokens(
            enc["input_ids"][0],
            enc["attention_mask"][0],
            CHUNK_SIZE
        )

        B_i = encode_chunks(chunks)
        f_i = chunk_mha(B_i.to(device))
        buffer.append(f_i.cpu())

    # SAVE INCREMENTALLY
    if (idx + 1) % SAVE_EVERY == 0:
        part_id += 1
        save_path = os.path.join(
            SAVE_DIR,
            f"patient_embeddings_part_{part_id:06d}.pt"
        )
        torch.save(torch.stack(buffer), save_path)
        buffer.clear()
        print(f"Saved {save_path}")

# SAVE REMAINING
if buffer:
    part_id += 1
    save_path = os.path.join(
        SAVE_DIR,
        f"patient_embeddings_part_{part_id:06d}.pt"
    )
    torch.save(torch.stack(buffer), save_path)
    print(f"Saved {save_path}")

print("✅ All patient embeddings saved successfully.")
