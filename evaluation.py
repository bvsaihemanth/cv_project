# =========================================================
# BLIP QUALITATIVE ANALYSIS — Flickr30k
# Model  : Salesforce/blip-image-captioning-large
# Dataset: Flickr30k (results.csv + flickr30k_images/)
#
# Analyses:
#   1. Embedding Visualization (t-SNE + PCA)
#   2. Similarity Heatmap (Global + Per-Image)
#   3. KDE — Caption Length Distribution
#   4. ECDF — Length + Per-image BLEU
#   5. Scatter Plots — Metric Relationships
#   6. Multi-Caption Similarity Per Image
# =========================================================

# ── Silence noisy logs ───────────────────────────────────
import os
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

import warnings
warnings.filterwarnings("ignore")

# ── Core ─────────────────────────────────────────────────
import torch
import numpy as np
import pandas as pd
import random
import json
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ── Model ────────────────────────────────────────────────
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── Metrics ──────────────────────────────────────────────
from nltk.translate.bleu_score  import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score                 import rouge_scorer
from pycocoevalcap.cider.cider   import Cider

# ── ML / Stats ───────────────────────────────────────────
from sentence_transformers          import SentenceTransformer
from sklearn.manifold               import TSNE
from sklearn.decomposition          import PCA
from sklearn.metrics.pairwise       import cosine_similarity
from scipy.stats                    import gaussian_kde

# ── Plotting ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot        as plt
import matplotlib.gridspec      as gridspec
from matplotlib.colors          import LinearSegmentedColormap
import seaborn as sns

# =========================================================
# PATHS  — same as your original script
# =========================================================
CSV_PATH  = "/mnt/c/Users/bomma/Downloads/archive/flickr30k_images/results.csv"
IMAGE_DIR = "/mnt/c/Users/bomma/Downloads/archive/flickr30k_images/flickr30k_images"
OUT_DIR   = "./qualitative_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# STYLE  — clean white / light theme
# =========================================================
DARK_BG  = "#ffffff"   # figure background  → white
PANEL_BG = "#f7f9fc"   # axes background    → very light grey-blue
ACCENT1  = "#1f77b4"   # blue      → Predicted captions
ACCENT2  = "#d62728"   # red       → Ground Truth
ACCENT3  = "#2ca02c"   # green     → highlights / KDE
ACCENT4  = "#ff7f0e"   # orange
ACCENT5  = "#9467bd"   # purple
GRID_COL = "#d0d7de"   # light grey grid lines
TEXT_COL = "#1a1a2e"   # near-black text
MUTED    = "#555555"   # muted grey for ticks / secondary text

plt.rcParams.update({
    "figure.facecolor": DARK_BG,  "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COL, "axes.labelcolor":  TEXT_COL,
    "axes.titlecolor":  TEXT_COL, "axes.titlesize":   13,
    "axes.labelsize":   11,       "xtick.color":      MUTED,
    "ytick.color":      MUTED,    "xtick.labelsize":  9,
    "ytick.labelsize":  9,        "grid.color":       GRID_COL,
    "grid.linewidth":   0.6,      "text.color":       TEXT_COL,
    "legend.facecolor": "#ffffff", "legend.edgecolor": GRID_COL,
    "legend.labelcolor":TEXT_COL, "legend.fontsize":  9,
    "figure.dpi":       150,      "savefig.dpi":      200,
    "savefig.bbox":     "tight",  "savefig.facecolor":"#ffffff",
    "font.family":      "DejaVu Sans",
})

# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# =========================================================
# LOAD BLIP MODEL  (your exact model)
# =========================================================
print("\nLoading BLIP model …")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large").to(device)
model.eval()
print("BLIP ready ✓")

# =========================================================
# LOAD SENTENCE EMBEDDER  (for analyses 1, 2, 6)
# =========================================================
print("Loading sentence-transformer …")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedder ready ✓")

# =========================================================
# DATASET LOADER  (your exact logic)
# =========================================================
def load_dataset(csv_path, img_dir, limit=200):
    df = pd.read_csv(csv_path, sep="|")
    df.columns         = [c.strip() for c in df.columns]
    df["image_name"]   = df["image_name"].str.strip()
    df["comment"]      = df["comment"].str.strip().str.lower()

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["image_name"]].append(row["comment"])

    imgs, refs = [], []
    for img_name, caps in grouped.items():
        if len(caps) >= 5:
            path = os.path.join(img_dir, img_name)
            if os.path.exists(path):
                imgs.append(path)
                refs.append(caps[:5])           # exactly 5 GT captions
        if len(imgs) == limit:
            break

    print(f"Dataset loaded: {len(imgs)} images ✓")
    return imgs, refs

# =========================================================
# CAPTION GENERATORS
# =========================================================
def generate_caption(img_path):
    """Single greedy/beam caption — same as your original."""
    image  = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True).lower().strip()


def generate_multiple_captions(img_path, n=5):
    """Sampled captions for Analysis 6 diversity check."""
    image  = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    caps   = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=True,
                top_k=50, temperature=1.0, max_new_tokens=30)
        caps.append(processor.decode(out[0], skip_special_tokens=True).lower().strip())
    return caps

# =========================================================
# METRIC HELPERS
# =========================================================
smooth_fn = SmoothingFunction().method1
rouge_fn  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def per_image_metrics(preds_list, refs_list):
    bleus, meteors, rouges = [], [], []
    for pred, refs in zip(preds_list, refs_list):
        pt = pred.split()
        rt = [r.split() for r in refs]
        bleus.append(sentence_bleu(rt, pt, smoothing_function=smooth_fn))
        meteors.append(np.mean([meteor_score([r.split()], pt) for r in refs]))
        rouges.append(np.mean([rouge_fn.score(r, pred)['rougeL'].fmeasure for r in refs]))
    return np.array(bleus), np.array(meteors), np.array(rouges)

def per_image_cider(preds_list, refs_list):
    cider_obj       = Cider()
    gts = {i: refs_list[i]  for i in range(len(preds_list))}
    res = {i: [preds_list[i]] for i in range(len(preds_list))}
    _, scores = cider_obj.compute_score(gts, res)
    return np.array(scores)

def embed(texts, batch=256):
    return embedder.encode(texts, batch_size=batch,
                           show_progress_bar=False,
                           normalize_embeddings=True)

# =========================================================
# ── STEP 1: LOAD DATA + GENERATE CAPTIONS ────────────────
# =========================================================
print("\n" + "="*55)
print("  STEP 1 — Loading Flickr30k + Generating Captions")
print("="*55)

image_paths, ground_truths = load_dataset(CSV_PATH, IMAGE_DIR, limit=200)

# Cache preds so you don't re-run if you crash mid-way
CACHE = os.path.join(OUT_DIR, "_preds_cache.json")
if os.path.exists(CACHE):
    print("Found caption cache — loading …")
    with open(CACHE) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} cached captions ✓")
else:
    preds = []
    print("Generating captions …")
    for img in tqdm(image_paths, desc="BLIP"):
        try:
            preds.append(generate_caption(img))
        except Exception as e:
            print(f"  Skip {img}: {e}")
            preds.append(None)
    with open(CACHE, "w") as f:
        json.dump(preds, f)
    print("Captions saved to cache ✓")

# Filter Nones
valid        = [(p, g) for p, g in zip(preds, ground_truths) if p is not None]
preds_clean  = [v[0] for v in valid]
gt_clean     = [v[1] for v in valid]
paths_clean  = [image_paths[i] for i, (p, _) in
                enumerate(zip(preds, ground_truths)) if p is not None]

print(f"\nValid captions: {len(preds_clean)} / {len(preds)}")

# ── Quick aggregate metrics print ────────────────────────
bleus_all, meteors_all, rouges_all = per_image_metrics(preds_clean, gt_clean)
ciders_all = per_image_cider(preds_clean, gt_clean)
print("\n==== AGGREGATE METRICS ====")
print(f"BLEU   : {bleus_all.mean():.4f}")
print(f"METEOR : {meteors_all.mean():.4f}")
print(f"ROUGE-L: {rouges_all.mean():.4f}")
print(f"CIDEr  : {ciders_all.mean():.4f}")

# =========================================================
# ── ANALYSIS 1: EMBEDDING VISUALIZATION (t-SNE + PCA) ────
# =========================================================
print("\n[1/6] Embedding Visualization (t-SNE + PCA) …")

N_EMBED = min(300, len(preds_clean))
idx_e   = random.sample(range(len(preds_clean)), N_EMBED)

sample_preds = [preds_clean[i]    for i in idx_e]
sample_refs  = [gt_clean[i][0]    for i in idx_e]   # 1 GT per image

all_texts  = sample_preds + sample_refs
n_pred     = len(sample_preds)

labels_src = (["Predicted"]    * n_pred +
              ["Ground Truth"] * len(sample_refs))
colors_src = ([ACCENT1]        * n_pred +
              [ACCENT2]        * len(sample_refs))

# Length bucket (predicted only)
lengths_e   = [len(t.split()) for t in sample_preds]
len_col_map = {
    "Short  (<6w)":   ACCENT3,
    "Medium (6-10w)": ACCENT4,
    "Long   (>10w)":  ACCENT5,
}
len_labels  = ["Short  (<6w)" if l < 6
               else "Medium (6-10w)" if l <= 10
               else "Long   (>10w)"
               for l in lengths_e]

vecs_e = embed(all_texts)

tsne_xy = TSNE(n_components=2, perplexity=40,
               random_state=42, learning_rate="auto",
               init="pca").fit_transform(vecs_e)
pca_xy  = PCA(n_components=2, random_state=42).fit_transform(vecs_e)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Analysis 1 — Caption Embedding Space\n"
    "BLIP (Salesforce/blip-image-captioning-large) · Flickr30k",
    fontsize=14, fontweight="bold", color=TEXT_COL, y=1.02)

for ax, xy, title in zip(axes,
                          [tsne_xy, pca_xy],
                          ["t-SNE Projection", "PCA Projection"]):
    mask_p = np.array([l == "Predicted"    for l in labels_src])
    mask_g = np.array([l == "Ground Truth" for l in labels_src])

    ax.scatter(xy[mask_g, 0], xy[mask_g, 1],
               c=ACCENT2, s=16, alpha=0.50, label="Ground Truth", zorder=2)
    ax.scatter(xy[mask_p, 0], xy[mask_p, 1],
               c=ACCENT1, s=22, alpha=0.80, label="BLIP Predicted", zorder=3)
    ax.set_title(title, fontsize=12)
    ax.legend(markerscale=1.4)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    # Inset: predicted colored by caption length
    ax_ins = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
    ax_ins.set_facecolor("#f7f9fc")
    p_xy = xy[mask_p]
    for lbl, col in len_col_map.items():
        m = np.array([l == lbl for l in len_labels])
        if m.any():
            ax_ins.scatter(p_xy[m, 0], p_xy[m, 1],
                           c=col, s=7, alpha=0.85, label=lbl)
    ax_ins.set_title("By Length", fontsize=7, color=TEXT_COL)
    ax_ins.tick_params(labelsize=5)
    ax_ins.legend(fontsize=5, loc="lower right",
                  framealpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_embedding_viz.png"))
plt.close()
print("   → 01_embedding_viz.png ✓")

# =========================================================
# ── ANALYSIS 2: SIMILARITY HEATMAP ───────────────────────
# (a) Global — 50 BLIP predictions  cosine matrix
# (b) Per-image — pred + 5 GT captions  (3 sample images)
# =========================================================
print("\n[2/6] Similarity Heatmap …")

fig = plt.figure(figsize=(20, 7))
fig.suptitle(
    "Analysis 2 — Cosine Similarity Heatmaps\n"
    "BLIP predictions (global) · Pred vs GT per image",
    fontsize=14, fontweight="bold", color=TEXT_COL)
gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.45)

# ── (a) Global: 50 predicted captions ────────────────────
N_GLOBAL = min(50, len(preds_clean))
idx_g    = random.sample(range(len(preds_clean)), N_GLOBAL)
samp_g   = [preds_clean[i] for i in idx_g]
vecs_g   = embed(samp_g)
sim_g    = cosine_similarity(vecs_g)

cmap_g = LinearSegmentedColormap.from_list(
    "gh", ["#ffffff", "#cce4f7", ACCENT1, "#08306b"])

ax_g = fig.add_subplot(gs[0, :3])
im_g = ax_g.imshow(sim_g, cmap=cmap_g, vmin=0, vmax=1, aspect="auto")
plt.colorbar(im_g, ax=ax_g, fraction=0.03, pad=0.02)
ax_g.set_title("Global — 50 BLIP Predicted Captions\n"
               "(Cosine Similarity Matrix)", fontsize=11)
ax_g.set_xlabel("Caption index")
ax_g.set_ylabel("Caption index")

off_diag = sim_g[~np.eye(N_GLOBAL, dtype=bool)]
ax_g.text(0.02, 0.97,
          f"Mean off-diag: {off_diag.mean():.3f}\n"
          f"Std : {off_diag.std():.3f}",
          transform=ax_g.transAxes, fontsize=8,
          color=ACCENT3, va="top")

# ── (b) Per-image: 3 sample images ───────────────────────
n_per   = 3
idx_per = random.sample(range(len(preds_clean)), n_per)
cmap_p  = LinearSegmentedColormap.from_list(
    "ph", ["#ffffff", "#e8d5f7", ACCENT5, "#3b006e"])

for k, ii in enumerate(idx_per):
    texts = [preds_clean[ii]] + gt_clean[ii][:5]      # 6 total
    vecs2 = embed(texts)
    sim2  = cosine_similarity(vecs2)

    ax_p = fig.add_subplot(gs[0, 3 + k])
    im_p = ax_p.imshow(sim2, cmap=cmap_p, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im_p, ax=ax_p, fraction=0.05, pad=0.03)

    row_labels = ["BLIP"] + [f"GT{j+1}" for j in range(5)]
    ax_p.set_xticks(range(6))
    ax_p.set_xticklabels(row_labels, rotation=45, fontsize=7)
    ax_p.set_yticks(range(6))
    ax_p.set_yticklabels(row_labels, fontsize=7)
    ax_p.set_title(f"Image {k+1}\nPred–GT Similarity", fontsize=9)

    # Cell values
    for i in range(6):
        for j in range(6):
            ax_p.text(j, i, f"{sim2[i,j]:.2f}",
                      ha="center", va="center", fontsize=6,
                      color="#1a1a2e" if sim2[i,j] < 0.55 else "white")

plt.savefig(os.path.join(OUT_DIR, "02_similarity_heatmap.png"))
plt.close()
print("   → 02_similarity_heatmap.png ✓")

# =========================================================
# ── ANALYSIS 3: KDE — Caption Length Distribution ────────
# =========================================================
print("\n[3/6] KDE …")

pred_lens = [len(p.split()) for p in preds_clean]
gt_lens   = [len(r.split()) for refs in gt_clean for r in refs]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Analysis 3 — KDE: Caption Length Distribution\n"
    "BLIP predictions vs Flickr30k Human Captions",
    fontsize=14, fontweight="bold", color=TEXT_COL)

# Left: KDE overlay
ax = axes[0]
sns.kdeplot(pred_lens, ax=ax, color=ACCENT1, fill=True,
            alpha=0.45, linewidth=2.2, label="BLIP Predicted")
sns.kdeplot(gt_lens,   ax=ax, color=ACCENT2, fill=True,
            alpha=0.35, linewidth=2.2, label="Human GT")
ax.axvline(np.mean(pred_lens), color=ACCENT1, linestyle="--",
           linewidth=1.5, label=f"Pred μ={np.mean(pred_lens):.1f}w")
ax.axvline(np.mean(gt_lens),   color=ACCENT2, linestyle="--",
           linewidth=1.5, label=f"GT μ={np.mean(gt_lens):.1f}w")
ax.set_xlabel("Caption Length (words)")
ax.set_ylabel("Density")
ax.set_title("KDE Overlay")
ax.legend(); ax.grid(True, alpha=0.25)

# Right: histogram overlay
ax2 = axes[1]
bins = range(0, max(max(pred_lens), max(gt_lens)) + 2)
ax2.hist(pred_lens, bins=bins, color=ACCENT1, alpha=0.70,
         density=True, label="BLIP Predicted",
         edgecolor=DARK_BG, linewidth=0.3)
ax2.hist(gt_lens,   bins=bins, color=ACCENT2, alpha=0.50,
         density=True, label="Human GT",
         edgecolor=DARK_BG, linewidth=0.3)
ax2.set_xlabel("Caption Length (words)")
ax2.set_ylabel("Density")
ax2.set_title("Histogram Overlay")
ax2.legend(); ax2.grid(True, alpha=0.25)

fig.text(0.5, -0.03,
         f"BLIP — mean:{np.mean(pred_lens):.1f}  "
         f"std:{np.std(pred_lens):.1f}  "
         f"median:{np.median(pred_lens):.0f}  "
         f"range:[{min(pred_lens)},{max(pred_lens)}]     "
         f"Human GT — mean:{np.mean(gt_lens):.1f}  "
         f"std:{np.std(gt_lens):.1f}  "
         f"median:{np.median(gt_lens):.0f}  "
         f"range:[{min(gt_lens)},{max(gt_lens)}]",
         ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_kde.png"))
plt.close()
print("   → 03_kde.png ✓")

# =========================================================
# ── ANALYSIS 4: ECDF ─────────────────────────────────────
# =========================================================
print("\n[4/6] ECDF …")

def ecdf(data):
    xs = np.sort(data)
    ys = np.arange(1, len(xs)+1) / len(xs)
    return xs, ys

px, py = ecdf(pred_lens)
gx, gy = ecdf(gt_lens)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Analysis 4 — ECDF: Cumulative Distribution\n"
    "Caption Length · Per-image BLEU Score",
    fontsize=14, fontweight="bold", color=TEXT_COL)

# Left: length ECDF
ax = axes[0]
ax.step(px, py, where="post", color=ACCENT1, linewidth=2.2, label="BLIP Predicted")
ax.step(gx, gy, where="post", color=ACCENT2, linewidth=2.2, label="Human GT")

for pct, col in [(0.25, ACCENT3), (0.50, ACCENT4), (0.75, ACCENT5)]:
    val_p = np.percentile(pred_lens, pct * 100)
    ax.axhline(pct, color=col, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(val_p, color=col, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(val_p + 0.15, pct + 0.01,
            f"P{int(pct*100)}={val_p:.0f}w",
            color=col, fontsize=7)

ax.set_xlabel("Caption Length (words)")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("BLIP Predicted vs Human GT — Length ECDF")
ax.legend(); ax.grid(True, alpha=0.25)

# Right: per-image BLEU ECDF (use precomputed bleus_all)
ax2 = axes[1]
bx, by = ecdf(bleus_all)
ax2.step(bx, by, where="post", color=ACCENT3, linewidth=2.2,
         label=f"Per-image BLEU  (μ={bleus_all.mean():.3f})")

for pct, col in [(0.25, ACCENT1), (0.50, ACCENT2), (0.75, ACCENT4)]:
    val = np.percentile(bleus_all, pct * 100)
    ax2.axhline(pct, color=col, linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.axvline(val, color=col, linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.text(val + 0.003, pct + 0.01,
             f"P{int(pct*100)}={val:.3f}", color=col, fontsize=7)

ax2.fill_between(bx, by, alpha=0.15, color=ACCENT3, step="post")
ax2.set_xlabel("BLEU Score")
ax2.set_ylabel("Cumulative Proportion")
ax2.set_title("Per-image BLEU Score ECDF")
ax2.legend(); ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_ecdf.png"))
plt.close()
print("   → 04_ecdf.png ✓")

# =========================================================
# ── ANALYSIS 5: SCATTER PLOTS ─────────────────────────────
# =========================================================
print("\n[5/6] Scatter Plots …")

# Use already-computed per-image metrics
lengths_arr = np.array([len(p.split()) for p in preds_clean])

data_df = pd.DataFrame({
    "Length":  lengths_arr,
    "BLEU":    bleus_all,
    "METEOR":  meteors_all,
    "ROUGE-L": rouges_all,
    "CIDEr":   ciders_all,
})

pairs = [
    ("Length",  "CIDEr",   "Caption Length vs CIDEr"),
    ("BLEU",    "METEOR",  "BLEU vs METEOR"),
    ("METEOR",  "ROUGE-L", "METEOR vs ROUGE-L"),
    ("BLEU",    "CIDEr",   "BLEU vs CIDEr"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "Analysis 5 — Metric Relationship Scatter Plots\n"
    "BLIP · Flickr30k · Per-image scores",
    fontsize=14, fontweight="bold", color=TEXT_COL)
axes = axes.flatten()

for ax, (xk, yk, title) in zip(axes, pairs):
    x = data_df[xk].values
    y = data_df[yk].values
    r = np.corrcoef(x, y)[0, 1]

    # Density coloring
    try:
        xy_pts = np.vstack([x, y])
        dens   = gaussian_kde(xy_pts)(xy_pts)
        dens   = (dens - dens.min()) / (dens.ptp() + 1e-9)
    except Exception:
        dens = np.ones(len(x))

    sc = ax.scatter(x, y, c=dens, cmap="plasma",
                    s=14, alpha=0.70, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Density")

    # Trend line
    m, b   = np.polyfit(x, y, 1)
    x_fit  = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, m * x_fit + b,
            color=ACCENT2, linewidth=1.8, linestyle="--",
            label=f"Trend (R={r:.2f})")

    ax.set_xlabel(xk); ax.set_ylabel(yk)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.20)
    ax.text(0.02, 0.97, f"Pearson R = {r:.3f}",
            transform=ax.transAxes, fontsize=9,
            color=ACCENT3, va="top", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_scatter_plots.png"))
plt.close()
print("   → 05_scatter_plots.png ✓")

# =========================================================
# ── ANALYSIS 6: MULTI-CAPTION SIMILARITY PER IMAGE ───────
# For each sampled image:
#   • Generate 5 BLIP captions (sampling)  → diversity of model
#   • Row/Col 0 = greedy pred, 1-5 = sampled BLIP caps
#   • Annotate GT–GT mean vs Pred–GT mean similarity
# =========================================================
print("\n[6/6] Multi-Caption Similarity per Image …")

N_IMGS  = 8
idx_mc  = random.sample(range(len(preds_clean)), min(N_IMGS, len(preds_clean)))

ncols   = 4
nrows   = (N_IMGS + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 4.4, nrows * 4.2))
fig.suptitle(
    "Analysis 6 — Per-Image Multi-Caption Similarity\n"
    "Row/Col 0 = BLIP greedy · 1-5 = BLIP sampled · "
    "GT = Flickr30k human captions",
    fontsize=13, fontweight="bold", color=TEXT_COL)
axes_flat = axes.flatten()

cmap_mc = LinearSegmentedColormap.from_list(
    "mc", ["#ffffff", "#cce4f7", ACCENT1, "#08306b"])

model_mean_sims, gt_mean_sims = [], []

for k, ii in enumerate(idx_mc):
    ax = axes_flat[k]

    # Greedy prediction (already have it)
    greedy = preds_clean[ii]

    # 5 sampled captions from BLIP
    print(f"   Sampling captions for image {k+1}/{N_IMGS} …")
    sampled = generate_multiple_captions(paths_clean[ii], n=5)

    # 6×6: greedy + 5 sampled
    texts = [greedy] + sampled
    vecs  = embed(texts)
    sim   = cosine_similarity(vecs)

    # Model diversity: off-diagonal of full 6×6
    off_mask = ~np.eye(6, dtype=bool)
    model_mean_sims.append(sim[off_mask].mean())

    # Pred vs GT similarity (greedy vs each of 5 GT)
    gt_vecs    = embed([greedy] + gt_clean[ii][:5])
    gt_sim     = cosine_similarity(gt_vecs)
    pred_gt    = gt_sim[0, 1:].mean()
    gt_gt_mask = ~np.eye(5, dtype=bool)
    gt_gt_mean = gt_sim[1:, 1:][gt_gt_mask].mean()
    gt_mean_sims.append(gt_gt_mean)

    im = ax.imshow(sim, cmap=cmap_mc, vmin=0, vmax=1, aspect="auto")
    labels = ["Greedy"] + [f"Samp{j+1}" for j in range(5)]
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(
        f"Img {k+1}  Model-div:{sim[off_mask].mean():.2f}  "
        f"P→GT:{pred_gt:.2f}",
        fontsize=8)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{sim[i,j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="#1a1a2e" if sim[i,j] < 0.55 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes_flat[N_IMGS:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_multicap_similarity.png"))
plt.close()
print("   → 06_multicap_similarity.png ✓")

# ── Summary bar chart ─────────────────────────────────────
fig2, ax = plt.subplots(figsize=(11, 4))
fig2.suptitle(
    "Analysis 6b — Model Diversity vs Human GT Diversity\n"
    "Mean cosine similarity (lower = more diverse)",
    fontsize=13, fontweight="bold", color=TEXT_COL)

x = np.arange(len(model_mean_sims))
w = 0.35
ax.bar(x - w/2, model_mean_sims, width=w,
       color=ACCENT1, alpha=0.85, label="BLIP caption diversity")
ax.bar(x + w/2, gt_mean_sims,   width=w,
       color=ACCENT2, alpha=0.85, label="Human GT diversity")
ax.set_xticks(x)
ax.set_xticklabels([f"Img {i+1}" for i in x])
ax.set_ylabel("Mean Cosine Similarity")
ax.set_ylim(0, 1)
ax.axhline(np.mean(model_mean_sims), color=ACCENT1,
           linestyle="--", linewidth=1.4, alpha=0.8,
           label=f"BLIP avg ({np.mean(model_mean_sims):.2f})")
ax.axhline(np.mean(gt_mean_sims), color=ACCENT2,
           linestyle="--", linewidth=1.4, alpha=0.8,
           label=f"GT avg ({np.mean(gt_mean_sims):.2f})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06b_multicap_summary.png"))
plt.close()
print("   → 06b_multicap_summary.png ✓")

# =========================================================
# DONE
# =========================================================
print("\n" + "="*55)
print("  ALL QUALITATIVE ANALYSES COMPLETE")
print(f"  Output folder: {OUT_DIR}/")
print("="*55)
print("""
FILES:
  01_embedding_viz.png        t-SNE + PCA (BLIP vs GT)
  02_similarity_heatmap.png   Global 50×50 + 3 per-image
  03_kde.png                  Length KDE (BLIP vs Human)
  04_ecdf.png                 Length ECDF + BLEU ECDF
  05_scatter_plots.png        4 metric scatter plots
  06_multicap_similarity.png  8 per-image 6×6 heatmaps
  06b_multicap_summary.png    Diversity bar chart
  _preds_cache.json           Cached BLIP predictions
""")