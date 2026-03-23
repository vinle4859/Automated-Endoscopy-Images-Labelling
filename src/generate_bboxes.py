"""
PatchCore-style Feature-Space Anomaly Detection for Bounding Box Generation.

Why this replaces autoencoder pixel-reconstruction:
  - Tumors share the same pixel-level texture as normal tissue (same organ,
    same camera family), so pixel-error maps yield near-zero signal.
  - Feature vectors from a deep ImageNet backbone encode SEMANTIC patterns
    (tissue organisation, surface topology) that differ between normal and
    pathological tissue even when raw pixels look alike.

Method:
  1. Extract multi-scale patch features from all Normal images using frozen
     ResNet50 layers 2, 3 & 4 (spatial resolution + semantic depth + morphology).
  2. Build a 'Normal Memory Bank' of all normal patch feature vectors.
  3. For each abnormal image, compute per-patch distance to k-NN in the bank.
     High distance == feature pattern unseen in normal tissue == anomaly.
  4. Threshold the spatial anomaly score map -> single largest bounding box.

Reference: PatchCore (Roth et al., CVPR 2022)
"""
import os
import sys
import cv2
import shutil
import hashlib
import json
import itertools
from contextlib import contextmanager
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Abnormal class images live here (Malignant / Benign / NP sub-folders)
DATA_ROOT         = os.path.join(PROJECT_ROOT, "data", "sample_data", "train")
TEST_ROOT         = os.path.join(PROJECT_ROOT, "data", "sample_data", "test")
OUTPUT_ROOT       = os.path.join(PROJECT_ROOT, "data", "yolo_dataset")

# Visualisations now go into a timestamped run directory (see run_manager.py).
# This constant is overwritten at the start of generate_and_save() — kept here
# only as a fallback for imports / standalone usage.
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "results", "visualizations")
VAL_SPLIT_RATIO   = 0.2   # fraction of train images held out for YOLO validation

# Cached memory bank — saved as float16 compressed numpy archive.
# Stores the bank array + score_ceiling so feature extraction (the slow
# 20-minute step) only runs once. Pass --rebuild-bank to regenerate.
BANK_CACHE_PATH   = os.path.join(PROJECT_ROOT, "models", "patchcore_bank.npz")

# All confirmed-normal image directories used to build the memory bank.
NORMAL_DIRS = [
    os.path.join(PROJECT_ROOT, "data", "sample_data", "train", "Normal"),
    os.path.join(PROJECT_ROOT, "data", "normal_endoscopic"),
]

IMG_SIZE             = 256
K_NEIGHBORS          = 5    # k-NN neighbours used to score each patch
MAX_PATCHES_PER_IMG  = 100  # patches kept per Normal image IMMEDIATELY after
                             # extraction — bounds peak RAM to:
                             # 6005 imgs × 100 × 3584 × 4 B ≈ 8.6 GB
                             # (~10% spatial coverage per image; good quality)
MAX_BANK_PATCHES     = 300_000  # v9: reduced from 500K to 300K to offset the
                                 # increased feature dim (1536→3584). Net RAM:
                                 # 300K × 3584 × 4B ≈ 4.1 GB vs old 500K ×
                                 # 1536 × 4B ≈ 2.9 GB — manageable on CPU.
VISUAL_SAMPLE        = 20   # save diagnostic plots for first N images per class
                             # (increased from 5 to catch annotation quality issues
                             # before committing to YOLO training)

# ── Bbox extraction tuning ────────────────────────────────────────────────────
# These parameters control how the anomaly heatmap is converted to bounding
# boxes.  Tuned through iterative diagnostic analysis (v2→v5):
#   v1: Otsu + 25×25 morph + single-bbox → 84% full-image garbage
#   v2: 0.30 thresh + margin=ceiling → 0.78% detection rate (overcorrected)
#   v3: 0.15 thresh + margin=ceiling*0.5 → few sparse bboxes on wrong locations
#   v4: Feature-map-level extraction → better detection rate but bboxes on
#       artifacts (edges, specular) not on actual tumors
#   v5: Density-smoothed scoring.  Gaussian blur raw excess at feature level
#       so large tumor regions (many nearby moderate-anomaly patches) get
#       amplified, while isolated artifact spikes are diluted.
#   v6: Bbox merging + adaptive threshold + larger bridging kernel.
#       Addresses: (a) fragmented bboxes on the same tumor mass,
#       (b) partial captures (ACC-1 bottom-edge only), (c) better sensitivity
#       for subtle anomalies via Otsu on nonzero pixels.
#   v7: Specular suppression before smoothing.  Diagnostic (2026-02-24) showed
#       Normal images peak@spec=41.9 vs ceiling=37.7 — specular reflections
#       on Normal tissue inflate the baseline, shrinking the gap to true
#       anomalies (weakest NP at 42.5 = only 0.6 above Normal FPs).
#       Fix: zero out specular patches in the excess map BEFORE Gaussian
#       smoothing.  This drops Normal FPs without touching tumor signal.
#   v8: Hair artifact suppression before smoothing.  Hair strands (thin dark
#       lines on bright mucosa) are rare in the Normal bank and score as
#       highly anomalous.  Morphological blackhat transform detects hair,
#       and hair-dominated patches are zeroed out alongside speculars.
#       Also: session-based run grouping (--step all groups outputs).
#   v9: Tier 2 — Layer 4 features.  ResNet50 layer4 (2048 dims, 8×8)
#       is upsampled to 32×32 and concatenated with layers 2+3, giving
#       3584-dim features per patch.  Layer 4 captures deeper structural/
#       morphological semantics (tissue organisation, gland patterns) that
#       help differentiate NP from Normal mucosa — the primary gap in
#       v6–v8.  Bank reduced from 500K→300K patches to offset RAM.  Bank
#       dimension auto-check forces rebuild when feature dims change.
ANOMALY_FLOOR_THRESH = 0.04  # absolute minimum threshold (prevents noise).
                              # The adaptive Otsu may go lower but never below this.
ANOMALY_MARGIN_FRAC  = 0.15  # scoring margin = ceiling × this factor.
                              # v4 used 0.5 which compressed moderate anomaly
                              # patches into the 0.05–0.15 range (below threshold).
                              # 0.15 means a patch at 1.15× ceiling saturates to 1.0.
                              # After smoothing, the effective range is wider.
MIN_BBOX_AREA_FRAC   = 0.005 # reject tiny noise contours (<0.5% of image)
                              # v4 used 0.001 which allowed single-patch noise
                              # bboxes on artifacts.  User says tumors are LARGE
                              # tissue blocks → raise minimum.
MAX_BBOX_AREA_FRAC   = 0.85  # reject near-full-image contours (>85%)
MAX_BBOXES_PER_IMAGE = 3     # keep at most 3 bboxes (score-sorted).  User says
                              # tumors are big blocks → unlikely >3 per image.
# ── Density smoothing (v5) ─────────────────────────────────────────────────────
# The core problem in v4: PatchCore detects ANY deviation from Normal, not
# specifically tumors.  Anomalous patches are scattered across edges, specular
# spots, and texture variations — not concentrated at the tumor.
# Solution: Gaussian smooth the raw excess scores at feature-map resolution
# (32×32).  This acts as a density estimator:
#   - Large tumor regions (many nearby moderate-anomaly patches) → strong
#     smoothed signal → large connected region → good bbox
#   - Isolated edge/artifact anomalies → diluted by smoothing → filtered out
SCORE_SMOOTH_SIGMA   = 2.0   # Gaussian sigma at 32×32 (≈6% of image width)
SCORE_SMOOTH_KERNEL  = 7     # must be odd

# ── Feature-map-resolution bbox extraction (v4+) ──────────────────────────────
FEATURE_MAP_RES      = 32    # = IMG_SIZE / 8 (layer2 spatial resolution)
BBOX_PAD_FRAC        = 0.05  # expand each bbox by 5% on every side (tumors are
                              # large — generous padding avoids cutting off edges)
BRIDGE_DILATE_K      = 5     # dilation kernel at feature resolution (v6: 3→5
                              # to bridge wider gaps between hotspots within
                              # the same tumor mass)

# ── Bbox merging (v6) ──────────────────────────────────────────────────────────
# After connected-component extraction, merge bboxes whose edges are within
# BBOX_MERGE_GAP_FRAC of each other (in normalised coords).  This solves the
# fragmentation problem: a tumor with 2-3 hotspot clusters gets ONE bbox.
BBOX_MERGE_GAP_FRAC  = 0.08  # merge bboxes if edge gap < 8% of image dimension

# ── Specular suppression (v7) ─────────────────────────────────────────────────
# Specular highlights (bright, desaturated reflections) score far above the
# Normal ceiling because ImageNet ResNet50 has never seen such extreme white
# patches on mucosal surfaces.  This artificially inflates Normal baseline
# scores and anchors bboxes on reflections rather than tissue pathology.
# v7 zeroes out specular patches from the excess map BEFORE Gaussian smoothing,
# so they cannot contribute to the density estimate at all.
SPECULAR_V_THRESH    = 240   # HSV Value channel threshold (0–255)
SPECULAR_S_THRESH    = 30    # HSV Saturation channel threshold (0–255)
SPECULAR_COVERAGE    = 0.25  # fraction of receptive field that must be specular
                              # for a feature patch to be masked

# ── Hair artifact suppression (v8) ───────────────────────────────────────────
# Hair strands are thin dark lines on bright mucosal surfaces.  They are
# rarely present in Normal endoscopic images, so the memory bank has never
# seen them → PatchCore scores them as highly anomalous.
# Detection: morphological blackhat transform isolates thin dark structures
# on a bright background.  The blackhat response is thresholded to produce
# a binary hair mask, then downsampled to feature-map resolution.
# Any feature patch whose receptive field has > HAIR_COVERAGE hair pixels
# is zeroed out before density smoothing, exactly like specular suppression.
HAIR_BLACKHAT_KSIZE  = 11    # kernel size for blackhat (must be odd). Smaller
                              # than v8 to avoid absorbing broad dark texture.
HAIR_INTENSITY_THRESH = 45   # stricter than v8 (30): only strong hair-like
                              # responses are masked.
HAIR_COVERAGE        = 0.25  # stricter than v8 (0.15): require more patch
                              # coverage before suppressing a feature patch.

# ── Edge suppression ──────────────────────────────────────────────────────────
# Endoscopic images have dark borders, instrument tips, and vignetting at
# edges.  These score as highly anomalous but are NOT pathology.
# Zero out the outer N pixels of the feature map.  At 32×32 resolution,
# 2 pixels ≈ 6.25% of image per side → removes ~24% of feature map area.
# This is aggressive but justified: tumors are central ("between the walls").
EDGE_SUPPRESS_PX     = 2     # border pixels to zero at feature resolution

# ── v11 ROI-aware glare rejection ───────────────────────────────────────────
# Corner glare and dark circular border can dominate anomaly maps on the DINO
# branch. v11 adds tissue-ROI masking and border-component rejection.
ENABLE_FOV_MASK       = True
FOV_INTENSITY_THRESH  = 12    # grayscale threshold to separate tissue from black rim
FOV_MIN_COVERAGE      = 0.08  # minimum area for valid FOV component
BORDER_REJECT_MARGIN  = 4     # reject components too close to feature-map border

# ── v13 extraction improvements ────────────────────────────────────────────
# v13 replaces hard border-component rejection with soft border score decay,
# and uses a dual-threshold seed-grow mask to preserve weak contiguous lesion
# signal around strong anomaly cores.
ENABLE_V13_SOFT_BORDER = True
SOFT_BORDER_MARGIN_PX  = 4      # ramp width (feature-map pixels)
SOFT_BORDER_MIN_WEIGHT = 0.35   # weight at border (center remains 1.0)
ENABLE_V13_DUAL_THRESH = True
DUAL_THRESH_LOW_RATIO  = 0.55   # low threshold = high_threshold * ratio
DUAL_THRESH_LOW_FLOOR  = 0.015  # absolute floor for grow mask

# ── v14 draft gate-v2 constraints ───────────────────────────────────────────
# Governance-focused draft constraints to avoid accepting high-coverage
# pseudo-label sets that collapse specificity on Normal controls.
V14_MAX_NORMAL_FP_PCT = 20.0
V14_MAX_BBOX_TO_SIGNAL_RATIO = 1.80

# ImageNet normalisation required for pretrained ResNet weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Classes to annotate (Normal is excluded; used only for the memory bank)
CLASSES = {"Malignant": 0, "Benign": 1, "NP": 2}


@contextmanager
def _temporary_param_overrides(overrides):
    """Temporarily override module-level tuning constants."""
    if not overrides:
        yield
        return
    old_vals = {}
    for k, v in overrides.items():
        if k in globals():
            old_vals[k] = globals()[k]
            globals()[k] = v
    try:
        yield
    finally:
        for k, v in old_vals.items():
            globals()[k] = v


def get_fov_mask(img_path, fH, fW):
    """
    Estimate endoscope field-of-view mask at feature-map resolution.

    Keeps the largest non-black component. This removes circular black rims
    and corner regions that commonly trigger glare-driven false boxes.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.ones((fH, fW), dtype=bool)

    mask = (img > FOV_INTENSITY_THRESH).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_lbl <= 1:
        return np.ones((fH, fW), dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    full = (lbl == best).astype(np.float32)
    if full.mean() < FOV_MIN_COVERAGE:
        return np.ones((fH, fW), dtype=bool)

    down = cv2.resize(full, (fW, fH), interpolation=cv2.INTER_AREA)
    return down > 0.5


def _compute_soft_border_weights(h, w, margin_px, min_weight):
    """Return a feature-map weight matrix that softly attenuates border scores."""
    if margin_px <= 0:
        return np.ones((h, w), dtype=np.float32)

    yy, xx = np.indices((h, w))
    dist = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.float32)
    ramp = np.clip(dist / float(margin_px), 0.0, 1.0)
    return min_weight + (1.0 - min_weight) * ramp


# ── Data quality pre-check ─────────────────────────────────────────────────────
def validate_source_data():
    """
    Quick pre-flight check before running the expensive pipeline.
    Validates that all source images can be opened, checks for filename
    collisions across classes (which would silently overwrite labels in
    the flat yolo_dataset layout), and reports per-class counts.

    Returns True if safe to proceed, False otherwise.
    """
    print("\n── Data Quality Pre-Check ────────────────────────────────────────")
    ok = True
    all_fnames = {}       # fname -> (class, split) to detect collisions
    corrupt_files = []    # files that cannot be opened

    for split_name, root in [('train', DATA_ROOT), ('test', TEST_ROOT)]:
        for class_name in CLASSES:
            class_dir = os.path.join(root, class_name)
            if not os.path.exists(class_dir):
                continue
            files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            n_bad = 0
            for fname in files:
                # Check readability
                try:
                    img = Image.open(os.path.join(class_dir, fname))
                    img.verify()  # catches truncated files
                except Exception:
                    corrupt_files.append(os.path.join(class_dir, fname))
                    n_bad += 1

                # Check filename collision across classes
                if fname in all_fnames:
                    prev_cls, prev_split = all_fnames[fname]
                    print(f"  WARNING: Filename collision: '{fname}' appears in "
                          f"{prev_cls}/{prev_split} AND {class_name}/{split_name}. "
                          f"One will overwrite the other in yolo_dataset/.")
                    ok = False
                all_fnames[fname] = (class_name, split_name)

            status = f" ({n_bad} corrupt)" if n_bad else ""
            print(f"  {split_name:5s}/{class_name:12s}: {len(files):4d} images{status}")

    if corrupt_files:
        print(f"\n  FAIL: {len(corrupt_files)} corrupt/unreadable files found:")
        for p in corrupt_files[:10]:
            print(f"    {p}")
        if len(corrupt_files) > 10:
            print(f"    … and {len(corrupt_files) - 10} more")
        ok = False
    else:
        print(f"  OK: All images verified readable.")

    # Check Normal bank dirs
    for nd in NORMAL_DIRS:
        if os.path.exists(nd):
            nc = len([f for f in os.listdir(nd)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  bank/{os.path.basename(nd):24s}: {nc:5d} Normal images")

    print("──────────────────────────────────────────────────────────────────")
    return ok

# ── Feature extractor ─────────────────────────────────────────────────────────
class PatchCoreExtractor(nn.Module):
    """
    Frozen ResNet50 that returns concatenated layer2 + layer3 + layer4 features,
    all upsampled to the layer2 spatial resolution (IMG_SIZE/8 × IMG_SIZE/8).
    Output channels: 512 (layer2) + 1024 (layer3) + 2048 (layer4) = 3584.

    v9 change: added layer4 (2048 dims) for deeper structural/morphological
    semantics.  Layer4 features capture tissue organisation and gland patterns
    that help differentiate NP (structural anomaly) from Normal mucosa —
    the primary weakness in v6–v8 where NP detection was lowest.

    v10 change: optional DINO backbone.  When `backbone_path` is provided,
    the ResNet50 is initialised with self-supervised weights fine-tuned on
    5,957+ Normal endoscopic images (DINO, Tier 3).  This tightens the
    Normal feature cluster and widens the gap to pathological tissue.
    """
    def __init__(self, backbone_path=None):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # v10: load DINO-finetuned weights if provided
        if backbone_path is not None:
            if not os.path.isfile(backbone_path):
                raise FileNotFoundError(
                    f"DINO backbone not found: {backbone_path}\n"
                    f"Run `python src/main.py --step finetune` first.")
            print(f"  Loading DINO backbone from {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location='cpu',
                                    weights_only=False)
            # DINO saves {'backbone_state_dict': ..., 'epoch': ..., ...}
            state_dict = checkpoint.get('backbone_state_dict',
                                        checkpoint)
            # The DINO backbone has fc=Identity(); our ResNet50 has fc=Linear.
            # Filter out mismatched keys (fc layer) to avoid shape errors.
            model_dict = bb.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
            n_loaded = len(filtered)
            n_total = len(model_dict)
            bb.load_state_dict(filtered, strict=False)
            print(f"  DINO weights loaded: {n_loaded}/{n_total} parameters "
                  f"(epoch {checkpoint.get('epoch', '?')}, "
                  f"loss {checkpoint.get('loss', '?')})")

        children = list(bb.children())
        self.layer2 = nn.Sequential(*children[:6])    # → (B, 512, H/8,  W/8)
        self.layer3 = nn.Sequential(*children[:7])    # → (B,1024, H/16, W/16)
        self.layer4 = nn.Sequential(*children[:8])    # → (B,2048, H/32, W/32)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f2 = self.layer2(x)                                           # (B,512, H/8, W/8)
        f3 = self.layer3(x)                                           # (B,1024,H/16,W/16)
        f4 = self.layer4(x)                                           # (B,2048,H/32,W/32)
        target_size = f2.shape[-2:]                                    # (H/8, W/8)
        f3u = nn.functional.interpolate(
            f3, size=target_size, mode='bilinear', align_corners=False
        )                                                             # (B,1024,H/8, W/8)
        f4u = nn.functional.interpolate(
            f4, size=target_size, mode='bilinear', align_corners=False
        )                                                             # (B,2048,H/8, W/8)
        return torch.cat([f2, f3u, f4u], dim=1)                      # (B,3584,H/8, W/8)


# ── Memory bank construction ───────────────────────────────────────────────────
def build_memory_bank(extractor, normal_dirs, transform, device):
    """
    Build a Normal patch feature bank from one or more directories.

    After collecting all patches the bank is randomly subsampled to
    MAX_BANK_PATCHES (coreset approximation) so that the k-NN index
    remains tractable in memory and query time.

    Args:
        normal_dirs: str or list[str] — directories containing Normal images.
    Returns:
        (N, C) numpy float32 array, N ≤ MAX_BANK_PATCHES.
    """
    if isinstance(normal_dirs, str):
        normal_dirs = [normal_dirs]

    rng = np.random.default_rng(seed=42)
    all_feats = []
    for image_dir in normal_dirs:
        files = sorted(f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        tag = os.path.basename(image_dir)
        print(f"  [{tag}] {len(files)} Normal images")
        for fname in tqdm(files, desc=f"  {tag}"):
            img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
            t   = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = extractor(t)                                      # (1,C,fH,fW)
            # Flatten to (fH*fW, C)  —  e.g. (1024, 1536) for 256-px input
            f = f.squeeze(0).permute(1, 2, 0).reshape(-1, f.shape[1]).cpu().numpy()
            # ── Per-image subsample ───────────────────────────────────────────
            # Critical: accumulate only MAX_PATCHES_PER_IMG rows per image.
            # Without this, 6000 imgs × 1024 patches × 6 MB → >20 GB RAM.
            if f.shape[0] > MAX_PATCHES_PER_IMG:
                idx = rng.choice(f.shape[0], MAX_PATCHES_PER_IMG, replace=False)
                f   = f[idx]
            all_feats.append(f)

    bank = np.concatenate(all_feats, axis=0)
    print(f"  Raw bank : {bank.shape[0]:,} patches × {bank.shape[1]} features"
          f"  (~{bank.nbytes / 1e9:.2f} GB)")

    # Final coreset safety cap — trims raw patches down to MAX_BANK_PATCHES
    # (v9: 300K, reduced from 500K to offset 3584-dim features)
    if bank.shape[0] > MAX_BANK_PATCHES:
        idx  = rng.choice(bank.shape[0], MAX_BANK_PATCHES, replace=False)
        bank = bank[idx]
        print(f"  Coreset  : subsampled to {MAX_BANK_PATCHES:,} patches (seed=42)")

    return bank


# ── Specular mask at feature-map resolution ───────────────────────────────────
def get_specular_mask(img_path, fH, fW):
    """
    Detect specular highlights at FEATURE MAP resolution.

    Uses HSV thresholding: V > SPECULAR_V_THRESH and S < SPECULAR_S_THRESH
    (bright, desaturated = white/near-white reflections).

    Returns a boolean mask (fH, fW) where True = specular patch.
    A feature patch is labelled specular if more than SPECULAR_COVERAGE
    of its receptive field contains specular pixels.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return np.zeros((fH, fW), dtype=bool)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    specular_full = (hsv[:, :, 2] > SPECULAR_V_THRESH) & \
                    (hsv[:, :, 1] < SPECULAR_S_THRESH)
    # Downsample to feature resolution via area averaging
    spec_float = specular_full.astype(np.float32)
    spec_down = cv2.resize(spec_float, (fW, fH), interpolation=cv2.INTER_AREA)
    return spec_down > SPECULAR_COVERAGE


# ── Hair mask at feature-map resolution (v8) ─────────────────────────────────
def get_hair_mask(img_path, fH, fW):
    """
    Detect hair-like thin dark structures at FEATURE MAP resolution.

    Uses a morphological blackhat transform on the grayscale image:
      blackhat = closing(img) - img
    This isolates thin dark features (hair, fibres, instrument edges)
    that are darker than their local neighbourhood.

    The blackhat response is thresholded, then downsampled to (fH, fW).
    A feature patch is labelled as hair if more than HAIR_COVERAGE of
    its receptive field contains hair pixels.

    Returns a boolean mask (fH, fW) where True = hair-dominated patch.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return np.zeros((fH, fW), dtype=bool)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blackhat: large kernel captures structures thinner than the kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (HAIR_BLACKHAT_KSIZE, HAIR_BLACKHAT_KSIZE)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the blackhat response
    hair_full = (blackhat > HAIR_INTENSITY_THRESH).astype(np.float32)

    # Downsample to feature resolution via area averaging
    hair_down = cv2.resize(hair_full, (fW, fH), interpolation=cv2.INTER_AREA)
    return hair_down > HAIR_COVERAGE


# ── Normal score calibration ──────────────────────────────────────────────────
def calibrate_normal_score(bank, k, extractor, transform, device,
                           percentile=99):
    """
    Establish an ABSOLUTE reference for what a 'normal' patch score looks like
    by scoring FULL Normal images (all 1024 patches) against the bank.

    Previous approach (v1):
      Scored random bank patches against the bank itself.  This under-estimated
      the ceiling because the bank only kept 100 of 1024 patches per image.
      Full-image scoring produces MUCH higher distances for the 90% of spatial
      positions that were discarded → ceiling was set far too low → every image
      (including Normal) produced a saturated heatmap.

    Current approach (v2):
      Score ALL patches of the 48 sample_data/train/Normal images through the
      bank and take the `percentile`-th value.  This reflects the real score
      distribution that an incoming image will experience.

    Returns:
        score_ceiling (float): 99th-percentile normal full-image score.
    """
    # Use sample_data/train/Normal (same equipment as the abnormal images)
    primary_normal = NORMAL_DIRS[0]   # e.g. data/sample_data/train/Normal
    files = sorted(f for f in os.listdir(primary_normal)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not files:
        raise RuntimeError(f"No Normal images found in {primary_normal} for calibration")

    cal_nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute',
                                metric='euclidean', n_jobs=-1)
    cal_nbrs.fit(bank)

    all_scores = []
    for fname in tqdm(files, desc="  Calibrating on Normal images"):
        img   = Image.open(os.path.join(primary_normal, fname)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(tensor)                                  # (1,C,fH,fW)
        patches = feat.squeeze(0).permute(1, 2, 0).reshape(-1, feat.shape[1]).cpu().numpy()
        dists, _ = cal_nbrs.kneighbors(patches)
        scores   = dists.mean(axis=1)
        all_scores.extend(scores.tolist())

    all_scores = np.array(all_scores)
    ceiling    = float(np.percentile(all_scores, percentile))
    p50        = float(np.percentile(all_scores, 50))
    p95        = float(np.percentile(all_scores, 95))
    print(f"  Normal full-image scores  p50={p50:.4f}  p95={p95:.4f}  "
          f"p{percentile}={ceiling:.4f}")
    print(f"  (calibrated from {len(files)} images, {len(all_scores):,} patches)")
    return ceiling


# ── Anomaly map computation ────────────────────────────────────────────────────
def compute_raw_patch_scores(extractor, img_path, nbrs, transform, device):
    """Compute raw k-NN patch scores at feature-map resolution."""
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(tensor)
    _, C, fH, fW = feat.shape
    patches = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    dists, _ = nbrs.kneighbors(patches)
    scores = dists.mean(axis=1).reshape(fH, fW).astype(np.float32)
    return scores


def raw_scores_to_anomaly_map(raw_scores, img_path, score_ceiling, orig_w,
                              orig_h):
    """Convert raw k-NN scores to uint8 anomaly map using current settings."""
    fH, fW = raw_scores.shape

    # Step 1: subtractive baseline
    excess = np.maximum(raw_scores - score_ceiling, 0.0)

    # Step 2: edge suppression
    if EDGE_SUPPRESS_PX > 0:
        n = EDGE_SUPPRESS_PX
        excess[:n, :] = 0.0
        excess[-n:, :] = 0.0
        excess[:, :n] = 0.0
        excess[:, -n:] = 0.0

    # Step 2b: specular suppression
    spec_mask = get_specular_mask(img_path, fH, fW)
    excess[spec_mask] = 0.0

    # Step 2c: hair suppression
    hair_mask = get_hair_mask(img_path, fH, fW)
    excess[hair_mask] = 0.0

    # Step 2d: v11 FOV suppression
    if ENABLE_FOV_MASK:
        fov_mask = get_fov_mask(img_path, fH, fW)
        excess[~fov_mask] = 0.0

    # Step 2e: v13 soft border attenuation (replaces hard component rejection).
    if ENABLE_V13_SOFT_BORDER:
        wmap = _compute_soft_border_weights(
            fH, fW, SOFT_BORDER_MARGIN_PX, SOFT_BORDER_MIN_WEIGHT
        )
        excess *= wmap

    # Step 3: density smoothing
    excess_smooth = cv2.GaussianBlur(
        excess.astype(np.float32),
        (SCORE_SMOOTH_KERNEL, SCORE_SMOOTH_KERNEL),
        SCORE_SMOOTH_SIGMA
    )

    # Step 4: normalise
    margin = score_ceiling * ANOMALY_MARGIN_FRAC
    normed = np.clip(excess_smooth / (margin + 1e-8), 0.0, 1.0)
    amap = (normed * 255).astype(np.uint8)
    amap = cv2.resize(amap, (orig_w, orig_h))
    return amap


def compute_anomaly_map(extractor, img_path, nbrs, score_ceiling,
                        transform, device, orig_w, orig_h):
    """
    Per-patch anomaly score → density-smoothed anomaly heatmap (v8).

    v8 pipeline:
      1. Score each patch (mean k-NN distance to Normal bank)
      2. Subtract ceiling → zero out everything within Normal range
      3. Edge suppression → zero border patches
      4. **Specular suppression → zero specular patches before smoothing**
         (v7 change: prevents specular reflections from inflating the
         density estimate and anchoring bboxes on reflections)
      4b. **Hair suppression → zero hair artifact patches before smoothing**
          (v8 change: morphological blackhat detects thin dark structures)
      5. Gaussian smooth at feature-map resolution (32×32, σ=2.0)
         → large tumor regions consolidate; isolated artifacts dilute
      6. Normalize with reduced margin (ceiling × 0.15)
      7. Upscale to image resolution

    Returns a uint8 (orig_h, orig_w) heatmap.
    """
    raw_scores = compute_raw_patch_scores(extractor, img_path, nbrs,
                                          transform, device)
    return raw_scores_to_anomaly_map(raw_scores, img_path, score_ceiling,
                                     orig_w, orig_h)

# ── Bounding box extraction (v6: adaptive threshold + merging) ─────────────────
def _merge_bboxes(bboxes, gap_frac):
    """
    Iteratively merge bounding boxes whose edges are within gap_frac.

    Each bbox is (xc, yc, w, h) in normalised [0,1] coords.
    If two boxes' edges are within gap_frac on BOTH axes, they are merged
    into their bounding rectangle.  Repeats until no more merges possible.
    """
    if len(bboxes) <= 1:
        return bboxes

    def _to_xyxy(b):
        xc, yc, w, h = b
        return (xc - w/2, yc - h/2, xc + w/2, yc + h/2)

    def _to_xywh(x1, y1, x2, y2):
        return ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)

    def _should_merge(a, b, gap):
        """True if boxes a and b overlap or their edges are within gap."""
        ax1, ay1, ax2, ay2 = _to_xyxy(a)
        bx1, by1, bx2, by2 = _to_xyxy(b)
        # Gap between edges (negative = overlap)
        x_gap = max(ax1, bx1) - min(ax2, bx2)  # positive = separated
        y_gap = max(ay1, by1) - min(ay2, by2)
        # Merge if gap is within threshold on BOTH axes
        return x_gap < gap and y_gap < gap

    changed = True
    while changed:
        changed = False
        merged = []
        used = set()
        for i in range(len(bboxes)):
            if i in used:
                continue
            current = bboxes[i]
            for j in range(i + 1, len(bboxes)):
                if j in used:
                    continue
                if _should_merge(current, bboxes[j], gap_frac):
                    # Merge: take the bounding rect of both
                    ax1, ay1, ax2, ay2 = _to_xyxy(current)
                    bx1, by1, bx2, by2 = _to_xyxy(bboxes[j])
                    nx1 = min(ax1, bx1)
                    ny1 = min(ay1, by1)
                    nx2 = max(ax2, bx2)
                    ny2 = max(ay2, by2)
                    current = _to_xywh(nx1, ny1, nx2, ny2)
                    used.add(j)
                    changed = True
            merged.append(current)
            used.add(i)
        bboxes = merged

    return bboxes


def extract_bboxes(amap, orig_w, orig_h, param_overrides=None):
    """
    Convert an anomaly heatmap to 0–N YOLO bounding boxes.

    v6 approach (adaptive threshold + bbox merging):
      1. Downsample anomaly map to feature-map resolution (32×32).
      2. Adaptive threshold: use Otsu on the nonzero pixels of the feature
         map to find the optimal split.  This captures broader anomaly
         regions (like ACC-1) that a fixed high threshold would miss.
         Falls back to ANOMALY_FLOOR_THRESH if Otsu gives a lower value.
      3. Dilate with BRIDGE_DILATE_K (5×5) + close (5×5) to bridge gaps.
      4. Connected-component extraction with area filtering.
      5. Bbox merging: fuse nearby bboxes into one if edge gap < 8%.
         This eliminates the fragmentation problem where a single tumor
         produces 2–3 separate smaller bboxes.

    Returns list of (xc, yc, w, h) tuples in YOLO normalised format.
    """
    with _temporary_param_overrides(param_overrides):
        # Downsample to feature-map resolution. INTER_AREA preserves average
        # intensity, preventing aliasing from discarding hot pixels.
        fmap = cv2.resize(amap, (FEATURE_MAP_RES, FEATURE_MAP_RES),
                          interpolation=cv2.INTER_AREA)

    # ── Adaptive threshold (v6) ───────────────────────────────────────────
    # Fixed threshold (v5: 0.08) was too rigid — it either missed broad
    # low-intensity anomaly regions (ACC-1) or required tuning per class.
    # v6: apply Otsu on the nonzero pixels.  This finds the natural split
    # between background (normal tissue → 0 from subtractive scoring) and
    # anomaly signal, adapting per-image to the actual signal strength.
        nonzero_mask = fmap > 0
        nonzero_vals = fmap[nonzero_mask]
        if len(nonzero_vals) > 0:
            # Otsu on the nonzero distribution
            otsu_thr, _ = cv2.threshold(nonzero_vals, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Floor: never go below ANOMALY_FLOOR_THRESH to prevent noise
            floor_val = int(255 * ANOMALY_FLOOR_THRESH)
            thr_val = max(int(otsu_thr), floor_val)
        else:
            return []  # no anomaly signal at all

        if ENABLE_V13_DUAL_THRESH:
            # v13: seed-grow extraction to keep weak lesion shoulders connected
            # to confident cores while suppressing global low-level noise.
            high_thr = thr_val
            low_floor = int(255 * DUAL_THRESH_LOW_FLOOR)
            low_thr = max(int(high_thr * DUAL_THRESH_LOW_RATIO), low_floor)

            seed_mask = (fmap >= high_thr).astype(np.uint8)
            grow_mask = (fmap >= low_thr).astype(np.uint8)
            if seed_mask.sum() == 0:
                return []

            n_grow, grow_lbl, _, _ = cv2.connectedComponentsWithStats(grow_mask)
            thresh = np.zeros_like(grow_mask, dtype=np.uint8)
            for lid in range(1, n_grow):
                comp = (grow_lbl == lid)
                if np.any(seed_mask[comp] > 0):
                    thresh[comp] = 255
        else:
            _, thresh = cv2.threshold(fmap, thr_val, 255, cv2.THRESH_BINARY)

    # Bridge adjacent patches (5×5 dilate at 32×32 bridges wider gaps
    # between hotspot clusters within the same tumor)
        k_bridge = np.ones((BRIDGE_DILATE_K, BRIDGE_DILATE_K), np.uint8)
        bridged = cv2.dilate(thresh, k_bridge, iterations=1)

    # Close to fill internal gaps within a cluster (v6: 5×5 kernel)
        k_close = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, k_close)

    # Connected components at feature resolution
        n_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(closed)
        if n_labels <= 1:
            return []

    # Score each component by its mean anomaly intensity (for sorting)
        comp_scores = []
        for label_id in range(1, n_labels):
            mask = (labels == label_id)
            mean_val = float(fmap[mask].mean()) / 255.0
            comp_scores.append((label_id, mean_val))
        # Sort by mean anomaly score (strongest first)
        comp_scores.sort(key=lambda x: x[1], reverse=True)

        bboxes = []
        for label_id, mean_score in comp_scores:
            x_feat = comp_stats[label_id, cv2.CC_STAT_LEFT]
            y_feat = comp_stats[label_id, cv2.CC_STAT_TOP]
            w_feat = comp_stats[label_id, cv2.CC_STAT_WIDTH]
            h_feat = comp_stats[label_id, cv2.CC_STAT_HEIGHT]

            # v11 fallback: reject border-touching components only when v13
            # soft-border attenuation is disabled.
            if (not ENABLE_V13_SOFT_BORDER) and BORDER_REJECT_MARGIN > 0:
                if (x_feat <= BORDER_REJECT_MARGIN or y_feat <= BORDER_REJECT_MARGIN or
                    x_feat + w_feat >= FEATURE_MAP_RES - BORDER_REJECT_MARGIN or
                    y_feat + h_feat >= FEATURE_MAP_RES - BORDER_REJECT_MARGIN):
                    continue

        # Convert from feature-grid coordinates to normalised [0,1]
            x_min = x_feat / FEATURE_MAP_RES
            y_min = y_feat / FEATURE_MAP_RES
            w_norm = w_feat / FEATURE_MAP_RES
            h_norm = h_feat / FEATURE_MAP_RES

        # Add padding to cover lesion margins beyond the detected patches
            x_min = max(0.0, x_min - BBOX_PAD_FRAC)
            y_min = max(0.0, y_min - BBOX_PAD_FRAC)
            w_norm = min(1.0 - x_min, w_norm + 2 * BBOX_PAD_FRAC)
            h_norm = min(1.0 - y_min, h_norm + 2 * BBOX_PAD_FRAC)

        # Area check in normalised coordinates
            area_frac = w_norm * h_norm
            if area_frac < MIN_BBOX_AREA_FRAC:
                continue
            if area_frac > MAX_BBOX_AREA_FRAC:
                continue

        # YOLO format: center_x, center_y, width, height (all 0–1)
            xc = x_min + w_norm / 2.0
            yc = y_min + h_norm / 2.0
            bboxes.append((xc, yc, w_norm, h_norm))

    # ── Bbox merging (v6) ─────────────────────────────────────────────────
    # Merge nearby bboxes that belong to the same tumor mass.  A single
    # lesion can produce 2-3 disconnected hotspot clusters — merging them
    # gives YOLO one cohesive annotation per lesion.
        bboxes = _merge_bboxes(bboxes, BBOX_MERGE_GAP_FRAC)

    # Post-merge area check: merged boxes may now exceed MAX_BBOX_AREA_FRAC
        bboxes = [b for b in bboxes
              if MIN_BBOX_AREA_FRAC <= b[2] * b[3] <= MAX_BBOX_AREA_FRAC]

    # Sort by area (largest first) and limit
        bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        bboxes = bboxes[:MAX_BBOXES_PER_IMAGE]

        return bboxes


def compute_label_quality_metrics(output_root=OUTPUT_ROOT,
                                  normal_false_pos=0,
                                  normal_total=0,
                                  confidence_logs=None):
    """Compute gate metrics from generated YOLO label files."""
    metrics = {}
    total_boxes = 0
    total_edge = 0
    for split in ('train', 'val', 'test'):
        lbl_dir = os.path.join(output_root, 'labels', split)
        files = []
        if os.path.isdir(lbl_dir):
            files = sorted(f for f in os.listdir(lbl_dir) if f.endswith('.txt'))
        non_empty = 0
        boxes = 0
        edge = 0
        for fn in files:
            txt = open(os.path.join(lbl_dir, fn)).read().strip()
            if txt:
                non_empty += 1
            for line in txt.splitlines():
                s = line.split()
                if len(s) >= 5:
                    boxes += 1
                    xc = float(s[1]); yc = float(s[2])
                    if xc < 0.2 or xc > 0.8 or yc < 0.2 or yc > 0.8:
                        edge += 1
        total_boxes += boxes
        total_edge += edge
        metrics[split] = {
            'labels': len(files),
            'non_empty': non_empty,
            'non_empty_pct': (100.0 * non_empty / max(len(files), 1)),
            'boxes': boxes,
            'edge_box_pct': (100.0 * edge / max(boxes, 1)),
        }
    bbox_to_signal_vals = []
    if confidence_logs:
        for logs in confidence_logs:
            for row in logs:
                # row format: (filename, peak, n_bboxes, signal_frac,
                #              bbox_frac, bbox_to_signal_ratio)
                if len(row) >= 6:
                    bbox_to_signal_vals.append(float(row[5]))

    metrics['overall'] = {
        'boxes': total_boxes,
        'edge_box_pct': (100.0 * total_edge / max(total_boxes, 1)),
        'mean_bbox_to_signal_ratio': (
            float(np.mean(bbox_to_signal_vals)) if bbox_to_signal_vals else 0.0
        ),
        'p95_bbox_to_signal_ratio': (
            float(np.percentile(bbox_to_signal_vals, 95)) if bbox_to_signal_vals else 0.0
        ),
    }
    metrics['normal_negative_control'] = {
        'false_positives': int(normal_false_pos),
        'total': int(normal_total),
        'fp_pct': (100.0 * float(normal_false_pos) / max(int(normal_total), 1)),
    }
    return metrics


def evaluate_quality_gate(metrics, min_non_empty_pct=30.0,
                          max_edge_box_pct=20.0,
                          max_normal_fp_pct=None,
                          max_bbox_to_signal_ratio=None):
    """Return pass/fail booleans for resource-saving YOLO blocking."""
    train_pct = metrics.get('train', {}).get('non_empty_pct', 0.0)
    edge_pct = metrics.get('overall', {}).get('edge_box_pct', 100.0)
    normal_fp_pct = metrics.get('normal_negative_control', {}).get('fp_pct', 100.0)
    mean_bbox_to_signal = metrics.get('overall', {}).get('mean_bbox_to_signal_ratio', 999.0)

    normal_fp_pass = True
    if max_normal_fp_pct is not None:
        normal_fp_pass = normal_fp_pct <= max_normal_fp_pct

    bbox_signal_pass = True
    if max_bbox_to_signal_ratio is not None:
        bbox_signal_pass = mean_bbox_to_signal <= max_bbox_to_signal_ratio

    overall_pass = (train_pct >= min_non_empty_pct and
                    edge_pct <= max_edge_box_pct and
                    normal_fp_pass and
                    bbox_signal_pass)

    return {
        'train_non_empty_pass': train_pct >= min_non_empty_pct,
        'edge_box_pass': edge_pct <= max_edge_box_pct,
        'normal_fp_pass': normal_fp_pass,
        'bbox_to_signal_pass': bbox_signal_pass,
        'overall_pass': overall_pass,
        'mode': 'gate_v2' if (max_normal_fp_pct is not None or
                              max_bbox_to_signal_ratio is not None)
                else 'gate_v1',
        'thresholds': {
            'min_non_empty_pct': min_non_empty_pct,
            'max_edge_box_pct': max_edge_box_pct,
            'max_normal_fp_pct': max_normal_fp_pct,
            'max_mean_bbox_to_signal_ratio': max_bbox_to_signal_ratio,
        }
    }


def _collect_normal_patch_scores(extractor, nbrs, transform, device):
    """Collect raw normal patch scores once so percentile ceilings are cheap."""
    primary_normal = NORMAL_DIRS[0]
    files = sorted(f for f in os.listdir(primary_normal)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    all_scores = []
    for fname in tqdm(files, desc="  Collect normal scores"):
        img = Image.open(os.path.join(primary_normal, fname)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = extractor(tensor)
        patches = feat.squeeze(0).permute(1, 2, 0).reshape(-1, feat.shape[1]).cpu().numpy()
        dists, _ = nbrs.kneighbors(patches)
        all_scores.extend(dists.mean(axis=1).tolist())
    return np.array(all_scores, dtype=np.float32)


def _build_extractor_and_knn(backbone_path=None, rebuild_bank=False,
                             calibration_percentile=99, collect_normal_scores=False):
    """Shared setup for generation and sweep logic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = PatchCoreExtractor(backbone_path=backbone_path).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    bank_cache = BANK_CACHE_PATH.replace('.npz', '_dino.npz') if backbone_path else BANK_CACHE_PATH

    _probe = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        expected_dim = extractor(_probe).shape[1]
    del _probe

    bank = None
    if not rebuild_bank and os.path.exists(bank_cache):
        cached = np.load(bank_cache)
        bank = cached['bank'].astype(np.float32)
        if bank.shape[1] != expected_dim:
            bank = None
            rebuild_bank = True

    if bank is None:
        bank = build_memory_bank(extractor, NORMAL_DIRS, transform, device)
        os.makedirs(os.path.dirname(bank_cache), exist_ok=True)
        np.savez_compressed(bank_cache, bank=bank.astype(np.float16))

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='brute',
                            metric='euclidean', n_jobs=-1)
    nbrs.fit(bank)
    normal_scores = None
    if collect_normal_scores:
        normal_scores = _collect_normal_patch_scores(extractor, nbrs, transform,
                                                     device)
        ceiling = float(np.percentile(normal_scores, calibration_percentile))
    else:
        ceiling = calibrate_normal_score(bank, K_NEIGHBORS, extractor, transform,
                                         device, percentile=calibration_percentile)
    return extractor, nbrs, transform, device, ceiling, normal_scores


def run_v11_gate_sweep(backbone_path=None, rebuild_bank=False, session=None,
                       sample_per_class=24):
    """
    Run sensitivity sweep for v11 gate metrics and return best parameter set.

    Sweep objective: maximize abnormal bbox coverage while penalizing
    edge-centered detections and Normal false positives.
    """
    print("\n=== v11 Gate Sensitivity Sweep ===")
    extractor, nbrs, transform, device, _, normal_scores = _build_extractor_and_knn(
        backbone_path=backbone_path,
        rebuild_bank=rebuild_bank,
        calibration_percentile=99,
        collect_normal_scores=True,
    )

    # Build evaluation image set: sampled abnormal + all Normal test images
    eval_images = []
    for cls in CLASSES.keys():
        cdir = os.path.join(DATA_ROOT, cls)
        files = sorted([f for f in os.listdir(cdir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for fn in files[:sample_per_class]:
            eval_images.append((cls, os.path.join(cdir, fn), True))

    ntest = os.path.join(TEST_ROOT, 'Normal')
    if os.path.isdir(ntest):
        for fn in sorted(os.listdir(ntest)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                eval_images.append(('Normal', os.path.join(ntest, fn), False))

    # Cache raw score maps once
    raw_cache = {}
    for cls, path, is_abn in tqdm(eval_images, desc="  Precompute raw scores"):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        raw = compute_raw_patch_scores(extractor, path, nbrs, transform, device)
        raw_cache[path] = (cls, is_abn, raw, w, h)

    calib_grid = [95, 97, 99]
    margin_grid = [0.06, 0.08, 0.10, 0.12]
    floor_grid = [0.01, 0.02, 0.03, 0.04]

    rows = []
    ceiling_map = {p: float(np.percentile(normal_scores, p)) for p in calib_grid}

    for calib, margin, floor in itertools.product(calib_grid, margin_grid, floor_grid):
        ceiling = ceiling_map[calib]

        n_abn = 0
        abn_detect = 0
        n_norm = 0
        norm_fp = 0
        boxes = 0
        edge_boxes = 0

        overrides = {
            'ANOMALY_MARGIN_FRAC': margin,
            'ANOMALY_FLOOR_THRESH': floor,
            'ENABLE_FOV_MASK': True,
        }
        with _temporary_param_overrides(overrides):
            for path, (cls, is_abn, raw, w, h) in raw_cache.items():
                amap = raw_scores_to_anomaly_map(raw, path, ceiling, w, h)
                bbs = extract_bboxes(amap, w, h)
                if is_abn:
                    n_abn += 1
                    if bbs:
                        abn_detect += 1
                else:
                    n_norm += 1
                    if bbs:
                        norm_fp += 1

                for (xc, yc, bw, bh) in bbs:
                    boxes += 1
                    if xc < 0.2 or xc > 0.8 or yc < 0.2 or yc > 0.8:
                        edge_boxes += 1

        abn_cov = 100.0 * abn_detect / max(n_abn, 1)
        norm_fp_pct = 100.0 * norm_fp / max(n_norm, 1)
        edge_pct = 100.0 * edge_boxes / max(boxes, 1)
        objective = abn_cov - 0.6 * edge_pct - 0.5 * norm_fp_pct
        rows.append({
            'calibration_percentile': calib,
            'margin': margin,
            'floor': floor,
            'abnormal_coverage_pct': round(abn_cov, 3),
            'normal_fp_pct': round(norm_fp_pct, 3),
            'edge_box_pct': round(edge_pct, 3),
            'objective': round(objective, 3),
        })

    rows.sort(key=lambda r: r['objective'], reverse=True)
    best = rows[0]

    out_dir = os.path.join(PROJECT_ROOT, 'results', 'v11_sweeps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'latest_sweep.json')
    with open(out_path, 'w') as f:
        json.dump({'best': best, 'results': rows}, f, indent=2)

    print(f"  Sweep finished. Best config: {best}")
    print(f"  Sweep report: {out_path}")
    return best, out_path


def run_v12_advanced(rebuild_bank=False, session=None, dino_backbone_path=None):
    """
    v12 advanced mode: sweep both ImageNet and DINO backbones, pick best,
    then generate labels with that configuration.
    """
    print("\n=== v12 Advanced Backbone A/B ===")
    candidates = [('ImageNet', None)]
    if dino_backbone_path and os.path.isfile(dino_backbone_path):
        candidates.append(('DINO', dino_backbone_path))

    best_global = None
    best_backbone = None
    for name, bb_path in candidates:
        print(f"\n  Sweeping backbone: {name}")
        best, _ = run_v11_gate_sweep(backbone_path=bb_path,
                                     rebuild_bank=rebuild_bank,
                                     session=session)
        if best_global is None or best['objective'] > best_global['objective']:
            best_global = best
            best_backbone = bb_path

    print(f"\n  Selected backbone: {'DINO' if best_backbone else 'ImageNet'}")
    print(f"  Selected params: {best_global}")

    generate_dataset(
        rebuild_bank=rebuild_bank,
        session=session,
        backbone_path=best_backbone,
        anom_floor=best_global['floor'],
        anom_margin=best_global['margin'],
        calibration_percentile=best_global['calibration_percentile'],
        v11_mode=True,
    )

def generate_dataset(rebuild_bank=False, session=None, backbone_path=None,
                     anom_floor=None, anom_margin=None,
                     calibration_percentile=99, v11_mode=True,
                     v13_mode=True,
                     v14_mode=False,
                     min_non_empty_pct=30.0, max_edge_box_pct=20.0,
                     max_normal_fp_pct=None,
                     max_bbox_to_signal_ratio=None):
    """
    PatchCore-based pipeline:
    1. Load cached Normal memory bank, or build + cache it if absent / forced.
    2. Fit k-NN index on the bank (fast, ~30 s — always done at runtime).
    3. For training data: split 80/20 stratified into train/val, generate
       anomaly bboxes.  Images where PatchCore cannot localise an anomaly
       get an EMPTY label — accurate sparse labels beat garbage full-image boxes.
    4. For test data: generate bboxes for held-out test images.
    5. Normal-test negative control: score Normal test images, report false
       positive rate (ideally 0%).

    v2 fixes (from diagnostic analysis of first pipeline run):
      - Calibration: score full Normal images instead of bank self-scoring
      - Anomaly map: subtractive scoring (excess above ceiling, not ratio)
      - Bbox extraction: fixed threshold, smaller morphology, multi-bbox,
        area limits (reject <0.5% and >85% of image area)
      - Removed forced fallback that injected 322/384 full-image garbage bboxes

    Outputs per split:
        data/yolo_dataset/images/{train,val,test}/<img>.jpg
        data/yolo_dataset/labels/{train,val,test}/<img>.txt
        data/yolo_dataset/confidence_<split>_<class>.csv
        results/visualizations/<class>/sample_N_<img>.png  (train only, first N)

    Args:
        rebuild_bank: if True, ignore the cache and re-extract from scratch.
        session: optional Session object for grouped run output.
        backbone_path: optional path to DINO-finetuned backbone weights.
                       If None, uses default ImageNet ResNet50 weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Data quality pre-check ────────────────────────────────────────────────
    if not validate_source_data():
        print("\n  ✗ Data quality pre-check failed. Fix issues above before proceeding.")
        return

    # ── Clean stale YOLO output ────────────────────────────────────────────────
    # Previous runs may have left orphan files in yolo_dataset/.
    # Wipe it to guarantee a clean, consistent dataset.
    if os.path.exists(OUTPUT_ROOT):
        print(f"\n  Cleaning stale output at {OUTPUT_ROOT} …")
        shutil.rmtree(OUTPUT_ROOT)

    # ── Determine version and tier based on backbone selection ───────────────
    if backbone_path is not None:
        _version = 'v14' if v14_mode else ('v13' if v13_mode else ('v11' if v11_mode else 'v10'))
        _tier = 'Tier3_DINO'
        _backbone_label = f'DINO ({os.path.basename(backbone_path)})'
    else:
        _version = 'v14' if v14_mode else ('v13' if v13_mode else ('v11' if v11_mode else 'v9'))
        _tier = 'Tier2_Layer4'
        _backbone_label = 'ImageNet (default)'

    param_overrides = {
        'ANOMALY_FLOOR_THRESH': anom_floor if anom_floor is not None else ANOMALY_FLOOR_THRESH,
        'ANOMALY_MARGIN_FRAC': anom_margin if anom_margin is not None else ANOMALY_MARGIN_FRAC,
        'ENABLE_FOV_MASK': bool(v11_mode),
        'ENABLE_V13_SOFT_BORDER': bool(v13_mode),
        'ENABLE_V13_DUAL_THRESH': bool(v13_mode),
    }

    # ── Timestamped run directory for visualisations ──────────────────────────
    # Imports run_manager here (not at module level) to avoid circular deps
    # and to keep the module importable without run_manager for tests.
    try:
        from run_manager import create_run_dir
        run_dir = create_run_dir('generate', version=_version, params={
            'ANOMALY_FLOOR_THRESH': param_overrides['ANOMALY_FLOOR_THRESH'],
            'ANOMALY_MARGIN_FRAC': param_overrides['ANOMALY_MARGIN_FRAC'],
            'BBOX_MERGE_GAP_FRAC': BBOX_MERGE_GAP_FRAC,
            'MIN_BBOX_AREA_FRAC': MIN_BBOX_AREA_FRAC,
            'MAX_BBOX_AREA_FRAC': MAX_BBOX_AREA_FRAC,
            'MAX_BBOXES_PER_IMAGE': MAX_BBOXES_PER_IMAGE,
            'SCORE_SMOOTH_SIGMA': SCORE_SMOOTH_SIGMA,
            'BRIDGE_DILATE_K': BRIDGE_DILATE_K,
            'FEATURE_MAP_RES': FEATURE_MAP_RES,
            'HAIR_BLACKHAT_KSIZE': HAIR_BLACKHAT_KSIZE,
            'HAIR_INTENSITY_THRESH': HAIR_INTENSITY_THRESH,
            'HAIR_COVERAGE': HAIR_COVERAGE,
            'FEATURE_DIMS': 3584,
            'MAX_BANK_PATCHES': MAX_BANK_PATCHES,
            'TIER': _tier,
            'BACKBONE': _backbone_label,
            'CALIBRATION_PERCENTILE': calibration_percentile,
            'ENABLE_FOV_MASK': bool(v11_mode),
            'ENABLE_V13_SOFT_BORDER': bool(v13_mode),
            'ENABLE_V13_DUAL_THRESH': bool(v13_mode),
            'ENABLE_V14_DRAFT': bool(v14_mode),
            'V14_MAX_NORMAL_FP_PCT': max_normal_fp_pct,
            'V14_MAX_MEAN_BBOX_TO_SIGNAL_RATIO': max_bbox_to_signal_ratio,
        }, session=session)
        vis_dir_root = os.path.join(run_dir, 'visualizations')
    except ImportError:
        vis_dir_root = VISUALIZATION_DIR   # fallback
        run_dir = None

    # Override module-level VISUALIZATION_DIR so _process_split uses the
    # timestamped path (nonlocal isn't needed — we pass it through closure).
    nonlocal_vis_dir = vis_dir_root
    # ── Feature extractor (frozen ResNet50, optionally DINO-finetuned) ────────
    extractor = PatchCoreExtractor(backbone_path=backbone_path).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Load or build memory bank ─────────────────────────────────────────────
    # v10: Use separate bank cache for DINO backbone (different features)
    if backbone_path is not None:
        bank_cache = BANK_CACHE_PATH.replace('.npz', '_dino.npz')
        # DINO backbone produces different features → always rebuild if
        # switching from ImageNet, so also force rebuild_bank
        if not os.path.exists(bank_cache) and os.path.exists(BANK_CACHE_PATH):
            rebuild_bank = True  # don't reuse ImageNet bank
    else:
        bank_cache = BANK_CACHE_PATH

    # ── v9: Probe extractor output dimension for bank compatibility check ────
    _probe = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        _probe_feat = extractor(_probe)
    expected_dim = _probe_feat.shape[1]   # 3584 for v9 (layer2+3+4)
    del _probe, _probe_feat
    print(f"  Feature extractor output dimension: {expected_dim}")
    print(f"  Bank cache: {bank_cache}")

    bank = None
    score_ceiling = None

    if not rebuild_bank and os.path.exists(bank_cache):
        print(f"\n[1/3] Loading cached memory bank from {bank_cache} …")
        cached = np.load(bank_cache)
        bank   = cached['bank'].astype(np.float32)   # stored as float16
        print(f"  Loaded : {bank.shape[0]:,} patches × {bank.shape[1]} features")

        # ── v9: Dimension compatibility check ─────────────────────────────
        # If the cached bank has different feature dims (e.g. 1536 from v8)
        # than the current extractor (3584 for v9), force a rebuild.
        if bank.shape[1] != expected_dim:
            print(f"  ⚠ Bank dimension mismatch: cached={bank.shape[1]}, "
                  f"extractor={expected_dim}. Forcing rebuild…")
            rebuild_bank = True
            bank = None
            del cached
        else:
            # Always recalibrate — the calibration method (v2: full-image scoring)
            # may differ from what was cached (v1: bank self-scoring).
            print("  Recalibrating ceiling on full Normal images …")
            score_ceiling = calibrate_normal_score(
                bank, K_NEIGHBORS, extractor, transform, device,
                percentile=calibration_percentile
            )

    if bank is None:
        # Need to build from scratch (cache missing, forced rebuild, or dim mismatch)
        missing = [d for d in NORMAL_DIRS if not os.path.exists(d)]
        if missing:
            print("Error: the following Normal directories were not found:")
            for d in missing:
                print(f"  {d}")
            return

        if rebuild_bank and os.path.exists(bank_cache):
            print("\n[1/3] --rebuild-bank requested — rebuilding memory bank…")
        else:
            print("\n[1/3] No compatible cache — building PatchCore Normal memory bank…")

        bank = build_memory_bank(extractor, NORMAL_DIRS, transform, device)

        # Calibrate before saving — uses extractor to score full Normal images
        score_ceiling = calibrate_normal_score(
            bank, K_NEIGHBORS, extractor, transform, device,
            percentile=calibration_percentile
        )

        os.makedirs(os.path.dirname(bank_cache), exist_ok=True)
        np.savez_compressed(bank_cache,
                            bank=bank.astype(np.float16),
                            ceiling=np.array([score_ceiling]))
        print(f"  Bank cached to {bank_cache}")
        print(f"  ({bank.shape[1]}-dim float16 compressed)")

    # NOTE: algorithm='brute' is intentional.
    # ball_tree / kd_tree degrade in high-dimensional spaces (3584-dim here).
    # sklearn docs recommend brute force for d >> 15.
    print(f"\n[2/3] Fitting k-NN index (brute-force, {bank.shape[1]}-dim)…")
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='brute',
                            metric='euclidean', n_jobs=-1)
    nbrs.fit(bank)
    print(f"  k-NN fitted on {bank.shape[0]:,} patches.\n")

    # ── Output directories ────────────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', split), exist_ok=True)
    os.makedirs(nonlocal_vis_dir, exist_ok=True)

    # ── Helper: process a directory of images and write to a given split ──────
    def _process_split(source_dir, class_name, class_id, split, file_list,
                       do_visualise=False):
        """
        Score every image in *file_list* (from *source_dir*), extract bboxes,
        and write the image + label into OUTPUT_ROOT/<split>.

        Images where PatchCore finds no localised anomaly get an EMPTY label
        file — this is correct both for Normal images and for abnormal images
        whose features fall within the normal distribution.  Training with
        fewer but ACCURATE labels is far better than 300+ full-image garbage
        bboxes.

        Also records per-image peak anomaly score as a confidence metadata
        file alongside the label, enabling downstream noise-aware training.
        """
        img_out = os.path.join(OUTPUT_ROOT, 'images', split)
        lbl_out = os.path.join(OUTPUT_ROOT, 'labels', split)
        vis_dir = os.path.join(nonlocal_vis_dir, class_name)
        if do_visualise:
            os.makedirs(vis_dir, exist_ok=True)

        n_with_box = 0
        n_no_box   = 0
        confidence_log = []    # (fname, peak_score, n_bboxes, signal_frac, bbox_frac, ratio)
        for i, fname in enumerate(tqdm(file_list, desc=f"  {class_name}/{split}")):
            img_path = os.path.join(source_dir, fname)

            orig_bgr = cv2.imread(img_path)
            if orig_bgr is None:
                continue
            orig_h, orig_w = orig_bgr.shape[:2]
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

            amap = compute_anomaly_map(
                extractor, img_path, nbrs, score_ceiling,
                transform, device, orig_w, orig_h
            )

            # Track peak anomaly score (0–255 uint8 → 0.0–1.0 float)
            peak_score = float(amap.max()) / 255.0

            bboxes = extract_bboxes(amap, orig_w, orig_h)

            # Diagnose bbox-vs-heatmap size mismatch for failure-mode analysis.
            signal_mask = amap > 0
            signal_frac = float(signal_mask.mean())
            bbox_frac = float(sum((bw * bh) for (_, _, bw, bh) in bboxes))
            bbox_to_signal = bbox_frac / max(signal_frac, 1e-8)

            if bboxes:
                n_with_box += 1
            else:
                n_no_box += 1
            confidence_log.append((fname, peak_score, len(bboxes),
                                   signal_frac, bbox_frac, bbox_to_signal))

            # Diagnostic visualisation (train split only, first N)
            if do_visualise and i < VISUAL_SAMPLE:
                vis_img = orig_rgb.copy()
                for (xc, yc, bw, bh) in bboxes:
                    x1 = int((xc - bw / 2) * orig_w)
                    y1 = int((yc - bh / 2) * orig_h)
                    x2 = int((xc + bw / 2) * orig_w)
                    y2 = int((yc + bh / 2) * orig_h)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(orig_rgb);  axes[0].set_title('Original');         axes[0].axis('off')
                axes[1].imshow(amap, cmap='hot', vmin=0, vmax=255)
                axes[1].set_title(f'PatchCore Anomaly Map (peak={peak_score:.3f})')
                axes[1].axis('off')
                axes[2].imshow(vis_img);   axes[2].set_title('Predicted BBox');   axes[2].axis('off')
                plt.suptitle(f"{class_name} — {fname}", fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"sample_{i+1}_{fname}.png"), dpi=100)
                plt.close()

            shutil.copy(img_path, os.path.join(img_out, fname))
            txt_name = os.path.splitext(fname)[0] + '.txt'
            with open(os.path.join(lbl_out, txt_name), 'w') as f:
                for (xc, yc, bw, bh) in bboxes:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Persist per-image confidence metadata
        conf_csv = os.path.join(OUTPUT_ROOT, f'confidence_{split}_{class_name}.csv')
        with open(conf_csv, 'w') as cf:
            cf.write('filename,peak_anomaly_score,n_bboxes,signal_area_frac,bbox_area_frac,bbox_to_signal_ratio\n')
            for fn, ps, nb, sf, bf, br in confidence_log:
                cf.write(f'{fn},{ps:.6f},{nb},{sf:.6f},{bf:.6f},{br:.6f}\n')

        return len(file_list), n_with_box, n_no_box, confidence_log

    # ── Process training data with train/val split ────────────────────────────
    print("[3/5] Generating bounding boxes for training split…")
    stats = {}
    all_confidence_logs = []
    with _temporary_param_overrides(param_overrides):
        for class_name, class_id in CLASSES.items():
            class_dir = os.path.join(DATA_ROOT, class_name)
            if not os.path.exists(class_dir):
                print(f"  Warning: {class_dir} not found. Skipping.")
                continue

            files = sorted(f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))

            # Stratified train/val split (deterministic seed)
            train_files, val_files = train_test_split(
                files, test_size=VAL_SPLIT_RATIO, random_state=42
            )
            print(f"\n  Class '{class_name}' ({class_id}): "
                  f"{len(files)} images → {len(train_files)} train / {len(val_files)} val")

            total_t, box_t, nobox_t, conf_t = _process_split(
                class_dir, class_name, class_id, 'train', train_files, do_visualise=True
            )
            total_v, box_v, nobox_v, conf_v = _process_split(
                class_dir, class_name, class_id, 'val', val_files, do_visualise=False
            )
            all_confidence_logs.extend([conf_t, conf_v])
            stats[class_name] = {
                'train': (total_t, box_t, nobox_t),
                'val':   (total_v, box_v, nobox_v),
            }

    # ── Process held-out test data ────────────────────────────────────────────
    print("\n[4/5] Generating bounding boxes for test split…")
    with _temporary_param_overrides(param_overrides):
        for class_name, class_id in CLASSES.items():
            test_dir = os.path.join(TEST_ROOT, class_name)
            if not os.path.exists(test_dir):
                print(f"  Warning: {test_dir} not found. Skipping.")
                continue

            test_files = sorted(f for f in os.listdir(test_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            print(f"\n  Class '{class_name}' (test): {len(test_files)} images")

            total_te, box_te, nobox_te, conf_te = _process_split(
                test_dir, class_name, class_id, 'test', test_files, do_visualise=False
            )
            all_confidence_logs.append(conf_te)
            if class_name not in stats:
                stats[class_name] = {}
            stats[class_name]['test'] = (total_te, box_te, nobox_te)

    # ── Normal-test negative control (G10) ────────────────────────────────────
    # Score the held-out Normal test images through PatchCore. Ideally NONE
    # should produce a localised bbox — any that do indicate the calibration
    # ceiling is too low or Normal variance is too high.
    normal_test_dir = os.path.join(TEST_ROOT, 'Normal')
    false_pos = 0
    normal_files = []
    if os.path.isdir(normal_test_dir):
        print("\n[5/5] Normal-test negative control …")
        normal_files = sorted(f for f in os.listdir(normal_test_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        false_pos = 0

        # Save diagnostic visualisations for Normal images too — allows quick
        # human verification that Normal tissue produces near-zero heatmaps.
        normal_vis_dir = os.path.join(nonlocal_vis_dir, 'Normal')
        os.makedirs(normal_vis_dir, exist_ok=True)

        with _temporary_param_overrides(param_overrides):
            for idx, fname in enumerate(tqdm(normal_files, desc="  Normal/test")):
                img_path = os.path.join(normal_test_dir, fname)
                orig_bgr = cv2.imread(img_path)
                if orig_bgr is None:
                    continue
                orig_h, orig_w = orig_bgr.shape[:2]
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

                amap = compute_anomaly_map(
                    extractor, img_path, nbrs, score_ceiling,
                    transform, device, orig_w, orig_h
                )
                bboxes = extract_bboxes(amap, orig_w, orig_h)
                if bboxes:
                    false_pos += 1

                # Visualise first VISUAL_SAMPLE Normal images (expect blank heatmaps)
                if idx < VISUAL_SAMPLE:
                    vis_img = orig_rgb.copy()
                    for (xc, yc, bw, bh) in bboxes:
                        x1 = int((xc - bw / 2) * orig_w)
                        y1 = int((yc - bh / 2) * orig_h)
                        x2 = int((xc + bw / 2) * orig_w)
                        y2 = int((yc + bh / 2) * orig_h)
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    fp_label = " [FALSE POSITIVE]" if bboxes else ""
                    peak = float(amap.max()) / 255.0

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(orig_rgb)
                    axes[0].set_title('Original (Normal)')
                    axes[0].axis('off')
                    axes[1].imshow(amap, cmap='hot', vmin=0, vmax=255)
                    axes[1].set_title(f'Anomaly Map (peak={peak:.3f})')
                    axes[1].axis('off')
                    axes[2].imshow(vis_img)
                    axes[2].set_title(f'BBox Check{fp_label}')
                    axes[2].axis('off')
                    plt.suptitle(f"Normal (negative control) — {fname}", fontsize=10)
                    plt.tight_layout()
                    plt.savefig(os.path.join(normal_vis_dir,
                                f"normal_{idx+1}_{fname}.png"), dpi=100)
                    plt.close()

        fp_rate = 100 * false_pos / max(len(normal_files), 1)
        print(f"  Normal negative control: {false_pos}/{len(normal_files)} "
              f"false positives ({fp_rate:.1f}%)")
        if false_pos > 0:
            print(f"  ⚠ {false_pos} Normal images produced a bbox — "
                  f"consider raising calibration percentile or enlarging "
                  f"the Normal bank.")
    else:
        print("\n  ⚠ Normal test directory not found — skipping negative control.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Generation Summary ────────────────────────────────────────────")
    for cls, splits in stats.items():
        for split_name, (total, boxed, no_box) in splits.items():
            pct = 100 * boxed / total if total else 0
            nb_str = f"  ({no_box} no-box)" if no_box else ""
            print(f"  {cls:12s} {split_name:5s}: {boxed}/{total} localised "
                  f"({pct:.0f}%){nb_str}")
    quality_metrics = compute_label_quality_metrics(
        OUTPUT_ROOT,
        normal_false_pos=false_pos,
        normal_total=len(normal_files),
        confidence_logs=all_confidence_logs,
    )
    gate = evaluate_quality_gate(quality_metrics,
                                 min_non_empty_pct=min_non_empty_pct,
                                 max_edge_box_pct=max_edge_box_pct,
                                 max_normal_fp_pct=max_normal_fp_pct,
                                 max_bbox_to_signal_ratio=max_bbox_to_signal_ratio)
    quality_report = {
        'metrics': quality_metrics,
        'gate': gate,
    }
    gate_path = os.path.join(OUTPUT_ROOT, 'quality_gate.json')
    with open(gate_path, 'w') as gf:
        json.dump(quality_report, gf, indent=2)

    print(f"\n  YOLO dataset   → {OUTPUT_ROOT}")
    print(f"  Visualisations → {nonlocal_vis_dir}")
    print(f"  Quality gate   → {gate_path}")
    print(f"  Gate pass      → {gate['overall_pass']}")

    # ── Structured run summaries for reproducibility / failure-mode study ──
    summary = {
        'version': _version,
        'tier': _tier,
        'backbone': _backbone_label,
        'calibration_percentile': calibration_percentile,
        'v11_mode': bool(v11_mode),
        'v13_mode': bool(v13_mode),
        'v14_mode': bool(v14_mode),
        'stats': stats,
        'normal_negative_control': {
            'false_positives': false_pos,
            'total': len(normal_files),
            'fp_rate_pct': (100.0 * false_pos / max(len(normal_files), 1))
        },
        'quality_gate': gate,
        'paths': {
            'run_dir': run_dir,
            'yolo_dataset': OUTPUT_ROOT,
            'visualizations': nonlocal_vis_dir,
            'quality_gate_json': gate_path,
        }
    }

    if run_dir:
        summary_json = os.path.join(run_dir, 'summary.json')
        with open(summary_json, 'w', encoding='utf-8') as sf:
            json.dump(summary, sf, indent=2)

        summary_txt = os.path.join(run_dir, 'summary.txt')
        with open(summary_txt, 'w', encoding='utf-8') as tf:
            tf.write(f"version: {_version}\n")
            tf.write(f"backbone: {_backbone_label}\n")
            tf.write(f"calibration_percentile: {calibration_percentile}\n")
            tf.write(f"normal_fp: {false_pos}/{len(normal_files)}\n")
            tf.write(f"gate_pass: {gate['overall_pass']}\n")
            tf.write(f"gate_mode: {gate.get('mode', 'gate_v1')}\n")
            tf.write(f"train_non_empty_pass: {gate['train_non_empty_pass']}\n")
            tf.write(f"edge_box_pass: {gate['edge_box_pass']}\n")
            tf.write(f"normal_fp_pass: {gate.get('normal_fp_pass', True)}\n")
            tf.write(f"bbox_to_signal_pass: {gate.get('bbox_to_signal_pass', True)}\n")
            tf.write("localisation_summary:\n")
            for cls, splits in stats.items():
                for split_name, (total, boxed, no_box) in splits.items():
                    tf.write(f"  {cls}/{split_name}: {boxed}/{total} boxed, {no_box} no-box\n")

        artifacts = {
            'confidence_csvs': sorted([
                os.path.join(OUTPUT_ROOT, x)
                for x in os.listdir(OUTPUT_ROOT)
                if x.startswith('confidence_') and x.endswith('.csv')
            ]),
            'quality_gate_json': gate_path,
            'visualizations_dir': nonlocal_vis_dir,
            'summary_json': summary_json,
            'summary_txt': summary_txt,
        }
        with open(os.path.join(run_dir, 'artifacts_manifest.json'), 'w', encoding='utf-8') as af:
            json.dump(artifacts, af, indent=2)

if __name__ == "__main__":
    generate_dataset()
