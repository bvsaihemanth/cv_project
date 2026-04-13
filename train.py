# =============================
# 🔇 SILENCE (OPTIONAL)
# =============================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
BATCH_SIZE = 64

# =============================
# LOAD DATA
# =============================
with open(REL_PATH, "r") as f:
    data = json.load(f)

samples = []

# Expected:
# [subject, predicate, object, box_s, box_o]
for img_id, rels in data.items():
    for rel in rels:
        if len(rel) < 5:
            continue
        s, p, o, box_s, box_o = rel

        samples.append({
            "s": s,
            "p": p,
            "o": o,
            "box_s": box_s,
            "box_o": box_o
        })

print(f"Loaded samples: {len(samples)}")

# =============================
# SPATIAL FEATURES
# =============================
def compute_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1*h1 + w2*h2 - inter + 1e-6

    return inter / union


def spatial_features(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    dx = (x2 - x1) / (w1 + 1e-6)
    dy = (y2 - y1) / (h1 + 1e-6)

    area1 = w1 * h1
    area2 = w2 * h2

    iou = compute_iou(b1, b2)

    return [dx, dy, area1, area2, iou]

# =============================
# ENCODE DATA
# =============================
predicates = [s["p"] for s in samples]
label_enc = LabelEncoder()
y = label_enc.fit_transform(predicates)

X = []

for s in samples:
    feat = spatial_features(s["box_s"], s["box_o"])
    X.append(feat)

X = np.array(X)

# convert to tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# =============================
# SPLIT
# =============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# DATA LOADER
# =============================
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE
)

# =============================
# MODEL
# =============================
class RelationNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = RelationNet(
    input_dim=5,
    num_classes=len(label_enc.classes_)
).to(DEVICE)

# =============================
# LOSS & OPTIMIZER
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =============================
# TRAIN LOOP
# =============================
for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            preds = model(xb)
            pred_labels = torch.argmax(preds, dim=1)

            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

# =============================
# SAVE MODEL
# =============================
torch.save({
    "model_state": model.state_dict(),
    "label_encoder": label_enc.classes_
}, "relation_net.pth")

print("Training complete. Model saved.")