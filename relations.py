# =============================
# IMPORTS
# =============================
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================
# CONFIG
# =============================
REL_PATH = "/home/ubuntu_bomma/cv_project/cvdataset/relationships_v1_2.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
EMBED_DIM = 50

# =============================
# LOAD DATA
# =============================
with open(REL_PATH, "r") as f:
    data = json.load(f)

samples = []

for img_id, rels in data.items():
    for rel in rels:
        if len(rel) < 5:
            continue
        s, p, o, box_s, box_o = rel
        samples.append((s, p, o, box_s, box_o))

# =============================
# BUILD VOCAB
# =============================
word2idx = {}

def encode_word(w):
    w = w.lower()
    if w not in word2idx:
        word2idx[w] = len(word2idx) + 1
    return word2idx[w]

# =============================
# SPATIAL FEATURES
# =============================
def compute_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1+w1, x2+w2)
    yb = min(y1+h1, y2+h2)

    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter + 1e-6
    return inter / union


def spatial_features(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    dx = (x2 - x1) / (w1 + 1e-6)
    dy = (y2 - y1) / (h1 + 1e-6)

    return [
        dx, dy,
        w1*h1,
        w2*h2,
        compute_iou(b1, b2)
    ]

# =============================
# ENCODE DATA
# =============================
X_spatial = []
X_sub = []
X_obj = []
labels = []

for s, p, o, b1, b2 in samples:
    X_spatial.append(spatial_features(b1, b2))
    X_sub.append(encode_word(s))
    X_obj.append(encode_word(o))
    labels.append(p)

label_enc = LabelEncoder()
y = label_enc.fit_transform(labels)

# tensors
X_spatial = torch.tensor(X_spatial, dtype=torch.float32)
X_sub = torch.tensor(X_sub, dtype=torch.long)
X_obj = torch.tensor(X_obj, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# =============================
# SPLIT
# =============================
X_sp_tr, X_sp_val, X_s_tr, X_s_val, X_o_tr, X_o_val, y_tr, y_val = train_test_split(
    X_spatial, X_sub, X_obj, y, test_size=0.2
)

# =============================
# MODEL
# =============================
class RelationNetV2(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        self.embed = nn.Embedding(vocab_size + 1, EMBED_DIM)

        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM*2 + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, sub, obj, spatial):
        e1 = self.embed(sub)
        e2 = self.embed(obj)

        x = torch.cat([e1, e2, spatial], dim=1)
        return self.fc(x)

model = RelationNetV2(
    vocab_size=len(word2idx),
    num_classes=len(label_enc.classes_)
).to(DEVICE)

# =============================
# TRAIN
# =============================
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):

    model.train()

    preds = model(
        X_s_tr.to(DEVICE),
        X_o_tr.to(DEVICE),
        X_sp_tr.to(DEVICE)
    )

    loss = criterion(preds, y_tr.to(DEVICE))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_preds = model(
            X_s_val.to(DEVICE),
            X_o_val.to(DEVICE),
            X_sp_val.to(DEVICE)
        )

        pred_labels = torch.argmax(val_preds, dim=1)
        acc = (pred_labels.cpu() == y_val).float().mean()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")

# =============================
# SAVE
# =============================
torch.save({
    "model": model.state_dict(),
    "vocab": word2idx,
    "labels": label_enc.classes_
}, "relation_net_v2.pth")

print("Model saved.")