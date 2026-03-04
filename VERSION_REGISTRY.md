# Version Registry — Endoscopic AI Pipeline

**Project:** Trustworthy Endoscopic AI Pipeline with PatchCore + YOLOv11  
**Last Updated:** 2026-02-28  
**Purpose:** Complete record of all pipeline versions, changes, results, and analysis for the failure mode study.

---

## Quick Reference

| Version | Tier | Key Change | Features | Bank | Backbone | Normal FP | Mal Det | Ben Det | NP Det | YOLO mAP50 |
|---------|------|------------|----------|------|----------|-----------|---------|---------|--------|-----------|
| v1 | — | Initial PatchCore | L2+L3 (1536) | 500K | ImageNet | — | — | — | — | — |
| v2 | — | Subtractive calibration | L2+L3 (1536) | 500K | ImageNet | — | — | — | — | — |
| v3 | — | Threshold tuning | L2+L3 (1536) | 500K | ImageNet | — | — | — | — | — |
| v4 | — | Feature-map extraction | L2+L3 (1536) | 500K | ImageNet | — | — | — | — | — |
| v5 | — | Density smoothing | L2+L3 (1536) | 500K | ImageNet | — | — | — | — | — |
| v6 | — | Bbox merging + adaptive | L2+L3 (1536) | 500K | ImageNet | 17/32 (53%) | 63/76 (83%) | 61/76 (80%) | 62/76 (82%) | **0.181** |
| v7 | 1 | Specular suppression | L2+L3 (1536) | 500K | ImageNet | — | — | — | — | — |
| v8 | 1 | Hair suppression + sessions | L2+L3 (1536) | 500K | ImageNet | 15/32 (47%) | 60/76 (79%) | 59/76 (78%) | 55/76 (72%) | **0.075** |
| v9 | 2 | Layer 4 features | L2+L3+L4 (3584) | 300K | ImageNet | OOM | OOM | OOM | OOM | OOM |
| v10 | 3 | DINO backbone | L2+L3+L4 (3584) | 300K | DINO | TBD | TBD | TBD | TBD | TBD |

> **Note:** v1–v5 and v7 were development iterations without full YOLO training runs. Results shown are from production runs with complete generate + YOLO evaluation.

---

## Tier Classification

| Tier | Focus | Versions | Description |
|------|-------|----------|-------------|
| **Tier 0** | Baseline Development | v1–v6 | Core PatchCore implementation, calibration, scoring, bbox extraction |
| **Tier 1** | Artifact Suppression | v7–v8 | Pre-processing to remove non-pathological anomaly sources |
| **Tier 2** | Feature Extraction | v9 | Deeper features (Layer 4), multi-resolution potential |
| **Tier 3** | Backbone Adaptation | v10 | Domain-adapted backbone via DINO self-supervised learning |

---

## Detailed Version Histories

### v1 — Initial PatchCore Implementation

**Status:** Superseded  
**Key Changes:**
- First working PatchCore pipeline with frozen ResNet50 (ImageNet weights)
- Layer 2 + Layer 3 concatenated features (1536-dim per patch)
- Bank built from all Normal images (100 patches per image, 500K cap)
- k-NN scoring (k=5, brute-force)
- Self-scoring calibration: scored bank patches against themselves for ceiling
- Single largest contour → one bbox per image

**Problems:**
- Calibration ceiling too low (bank self-scoring underestimates real Normal range)
- 84% of images got full-image garbage bounding boxes
- Forced fallback injected full-image boxes when no anomaly found

---

### v2 — Subtractive Calibration

**Status:** Superseded  
**Key Changes:**
- **Full-image calibration:** Score ALL 1024 patches of 48 Normal train images → 99th percentile as ceiling
- **Subtractive scoring:** `excess = max(raw_score - ceiling, 0)` instead of ratio-based
- **Removed forced fallback:** Empty labels for images with no localised anomaly

**Impact:**
- Fixed the fundamental calibration error from v1
- Detection rate dropped drastically (overcorrected) — most images showed no signal

---

### v3 — Threshold Tuning

**Status:** Superseded  
**Key Changes:**
- Tuned threshold parameters (floor=0.15, margin=ceiling×0.5)
- Still pixel-level bbox extraction

**Problems:**
- Few sparse bboxes, often in wrong locations
- Feature-map resolution mismatch with pixel-level extraction

---

### v4 — Feature-Map-Level Extraction

**Status:** Superseded  
**Key Changes:**
- Extract bboxes at feature-map resolution (32×32) instead of pixel level
- Edge suppression (zero outer 2px at feature resolution)

**Impact:**
- Better spatial alignment of bboxes with actual anomalies
- But bboxes still attached to artifacts (edges, specular reflections) not tumors

---

### v5 — Density Smoothing (Key Breakthrough)

**Status:** Superseded  
**Key Changes:**
- **Gaussian smoothing at feature-map resolution** — the core insight
- Smoothing acts as spatial density estimator:
  - Large tumor regions (many nearby moderate-anomaly patches) → strong signal
  - Isolated artifacts (edges, specular) → diluted below threshold
- Parameters: σ=2.0, kernel=7 at 32×32 resolution

**Impact:**
- First version where bboxes reliably covered tumor regions
- Isolated artifact bboxes largely eliminated

---

### v6 — Bbox Merging + Adaptive Threshold (First Full Evaluation)

**Status:** Baseline for comparison  
**Run:** `results/runs/20260225_000836_generate/` + `results/runs/20260225_020803_yolo/`

**Key Changes:**
- **Bbox merging:** Iteratively merge boxes within 8% edge gap
- **Adaptive threshold:** Otsu on nonzero excess pixels (minimum floor=0.04)
- **Bridging dilation:** kernel=5 bridges gaps between nearby hotspots
- **Multi-bbox support:** Up to 3 bboxes per image (score-sorted)

**Parameters:**
```json
{
  "ANOMALY_FLOOR_THRESH": 0.04,
  "ANOMALY_MARGIN_FRAC": 0.15,
  "BBOX_MERGE_GAP_FRAC": 0.08,
  "MIN_BBOX_AREA_FRAC": 0.005,
  "MAX_BBOX_AREA_FRAC": 0.85,
  "MAX_BBOXES_PER_IMAGE": 3,
  "SCORE_SMOOTH_SIGMA": 2.0,
  "FEATURE_DIMS": 1536,
  "MAX_BANK_PATCHES": 500000
}
```

**Results:**
| Metric | Value |
|--------|-------|
| Malignant train detection | 63/76 (82.9%) |
| Benign train detection | 61/76 (80.3%) |
| NP train detection | 62/76 (81.6%) |
| Normal test FP | 17/32 (53.1%) |
| YOLO mAP50 | **0.181** |
| YOLO mAP50-95 | 0.0453 |
| Generate time | ~41 min (CPU) |
| Bank size | 500K × 1536 × 4B ≈ 2.9 GB |

**Analysis:**
- v6 established the baseline with reasonable anomaly detection rates (80-83%)
- Normal FP rate of 53% is problematic — over half of Normal images trigger false detections
- Root cause: specular reflections on Normal tissue score above the calibration ceiling
- YOLO mAP50=0.181 is low but expected given noisy PatchCore annotations
- NP detection slightly lower than Malignant/Benign (expected — NP is structurally subtle)

---

### v7 — Specular Suppression (Tier 1)

**Status:** Superseded (folded into v8)  
**Key Changes:**
- **Specular mask at feature-map resolution:** HSV thresholding (V>240, S<30)
- Zero specular patches in excess map BEFORE Gaussian smoothing
- Coverage threshold: 25% of receptive field must be specular to mask

**Impact:**
- Addresses the root cause of Normal FPs: specular reflections scoring above ceiling
- Specular patches can no longer inflate the density estimate or anchor bboxes
- v7 was not run as a full pipeline — changes were carried forward into v8

---

### v8 — Hair Suppression + Session Grouping (Tier 1)

**Status:** Evaluated  
**Run:** `results/runs/20260227_155725_session/` (initial, aggressive) + `results/runs/20260227_182240_generate/` + `results/runs/20260227_224548_yolo/` (retuned)

**Key Changes:**
- **Hair artifact suppression:** Morphological blackhat transform detects thin dark structures
  - Blackhat kernel=11 (was 15 initially, then retuned down)
  - Intensity threshold=45 (was 30 initially, retuned up for strictness)
  - Coverage=0.25 (was 0.15 initially, retuned up for strictness)
- **Session-based run grouping:** `--step all` groups generate + YOLO under one directory
- **YOLO visualization:** Saves prediction plots with bboxes + confidence for inspection

**Retuning Note:**
The initial v8 had aggressive hair masking (ksize=15, thresh=30, coverage=0.15) that erroneously masked tumor tissue. After diagnostic analysis, parameters were tightened to be more conservative (ksize=11, thresh=45, coverage=0.25).

**Parameters (retuned):**
```json
{
  "ANOMALY_FLOOR_THRESH": 0.04,
  "ANOMALY_MARGIN_FRAC": 0.15,
  "BBOX_MERGE_GAP_FRAC": 0.08,
  "SCORE_SMOOTH_SIGMA": 2.0,
  "HAIR_BLACKHAT_KSIZE": 11,
  "HAIR_INTENSITY_THRESH": 45,
  "HAIR_COVERAGE": 0.25,
  "FEATURE_DIMS": 1536,
  "MAX_BANK_PATCHES": 500000
}
```

**Results (retuned):**
| Metric | Value | vs v6 |
|--------|-------|-------|
| Malignant train detection | 60/76 (78.9%) | ↓3.9pp |
| Benign train detection | 59/76 (77.6%) | ↓2.6pp |
| NP train detection | 55/76 (72.4%) | ↓9.2pp |
| Normal test FP | 15/32 (46.9%) | ↓6.2pp |
| YOLO mAP50 | **0.075** | ↓0.106 |
| Generate time | ~28 min (CPU) | — |

**Analysis:**
- Normal FP rate improved (53% → 47%) — specular + hair suppression helped
- But anomaly detection rates DROPPED across all classes, especially NP (82% → 72%)
- YOLO mAP50 **regressed significantly** (0.181 → 0.075) — the suppression masked too much signal
- Root cause: even with retuned parameters, the combination of specular + hair suppression is removing signal from legitimate anomalies, not just artifacts
- The fundamental problem is NOT artifacts — it's the domain gap between ImageNet features and endoscopic tissue

**Key Insight:**
> Artifact suppression (Tier 1) has reached diminishing returns. At the current
> feature quality, further artifact masking hurts true anomaly signal more than
> it helps Normal specificity. The path forward is NOT more suppression — it's
> **better features** (Tier 2/3).

---

### v9 — Layer 4 Features (Tier 2)

**Status:** Implemented, crashed on local (16GB RAM)  
**Run:** `results/runs/20260228_085424_generate/` (crashed during calibration)

**Key Changes:**
- **Layer 4 features added:** ResNet50 layer4 (2048 dims, 8×8) upsampled to 32×32
- **Total feature dim:** 512 (L2) + 1024 (L3) + 2048 (L4) = 3584 per patch
- **Bank reduced:** 500K → 300K patches to offset increased feature dimensions
- **Bank dimension auto-check:** Automatic rebuild if cached bank dims ≠ extractor dims
- **Per-image patch cap:** 100 patches/image (bounds peak RAM during extraction)

**Parameters:**
```json
{
  "ANOMALY_FLOOR_THRESH": 0.04,
  "ANOMALY_MARGIN_FRAC": 0.15,
  "SCORE_SMOOTH_SIGMA": 2.0,
  "HAIR_BLACKHAT_KSIZE": 11,
  "HAIR_INTENSITY_THRESH": 45,
  "HAIR_COVERAGE": 0.25,
  "FEATURE_DIMS": 3584,
  "MAX_BANK_PATCHES": 300000,
  "TIER": "Tier2_Layer4"
}
```

**Execution (partial — crashed):**
| Phase | Status | Notes |
|-------|--------|-------|
| Feature extraction | ✅ Completed | 5,957 images, ~28 min on CPU |
| Raw bank | ✅ Built | 600,500 patches × 3584 = ~8.61 GB |
| Coreset subsampling | ✅ Done | 300K patches |
| Calibration | ❌ CRASHED | OOM after ~1 image (48 total) |

**RAM Analysis:**
```
Raw bank (during extraction):   600K × 3584 × 4B = 8.61 GB
Coreset bank (after subsample): 300K × 3584 × 4B = 4.10 GB
k-NN brute-force (internal copy): ~4.10 GB
Calibration overhead (patches):  ~0.5 GB
OS + Python + PyTorch:           ~3 GB
TOTAL PEAK:                      > 16 GB  ← exceeded 16 GB system RAM
```

**Why it crashed:**
scikit-learn's `NearestNeighbors(algorithm='brute')` internally copies the bank when fitting + querying. With a 4.1 GB bank, the k-NN code alone needs ~8 GB (fitted data + query overhead). Combined with the bank itself, OS, and Python overhead, total exceeds 16 GB.

**Expected Impact (hypothesis):**
- Layer 4 features encode tissue organisation and gland architecture
- These morphological patterns are the key differentiator for NP (nasal polyps) vs Normal
- v6–v8's NP detection was the weakest class → Layer 4 should specifically help NP
- Needs ≥32 GB RAM or GPU server to evaluate

---

### v10 — DINO Self-Supervised Backbone (Tier 3)

**Status:** Implemented, awaiting cloud execution  
**Code:** `src/finetune_backbone.py` (DINO training) + `src/generate_bboxes.py` (backbone integration)

**Key Changes:**
- **DINO self-supervised fine-tuning** of ResNet50 on 5,957 Normal endoscopic images
- **Self-distillation:** Student and Teacher networks (EMA), no labels required
- **Domain adaptation:** Backbone learns endoscope-specific invariances (lighting, specular, tissue texture)
- **Drop-in replacement:** Fine-tuned backbone replaces ImageNet defaults in PatchCoreExtractor
- **Separate bank cache:** DINO bank saved to `models/patchcore_bank_dino.npz` to avoid conflicts

**DINO Architecture:**
```
Teacher network (EMA of Student):
  ResNet50 backbone → DINOHead (2048 → 2048 → 256 → 65536)
                         ↓
Student network (learned via gradient descent):
  ResNet50 backbone → DINOHead (2048 → 2048 → 256 → 65536)
                         ↓
Loss: Cross-entropy(teacher_sharpened, student_softened)
  + Centering (prevents mode collapse)
  + EMA momentum: 0.996 → 1.0 (cosine schedule)
```

**DINO Training Parameters:**
```json
{
  "method": "DINO",
  "backbone": "ResNet50",
  "epochs": 100,
  "batch_size": 32,
  "lr": 5e-4,
  "warmup_epochs": 10,
  "img_size": 224,
  "out_dim": 65536,
  "momentum_range": "0.996 → 1.0",
  "temp_student": 0.1,
  "temp_teacher": 0.04
}
```

**Pipeline Parameters (same as v9 + DINO backbone):**
```json
{
  "ANOMALY_FLOOR_THRESH": 0.04,
  "ANOMALY_MARGIN_FRAC": 0.15,
  "SCORE_SMOOTH_SIGMA": 2.0,
  "FEATURE_DIMS": 3584,
  "MAX_BANK_PATCHES": 300000,
  "TIER": "Tier3_DINO",
  "BACKBONE": "DINO (dino_resnet50.pth)"
}
```

**Expected Impact (hypothesis):**
1. **Tighter Normal cluster:** DINO forces the backbone to produce similar features for different views of the same Normal tissue → Normal patches cluster more tightly in feature space
2. **Wider anomaly gap:** Pathological tissue that is pixel-similar but structurally different from Normal will be pushed further from the Normal cluster
3. **Better specular invariance:** Augmentations during DINO training (color jitter, blur, crop) teach the backbone to ignore specular reflections → reduces Normal FPs without explicit masking
4. **NP improvement:** The tighter Normal cluster should make the subtle structural differences of NP more detectable

**Requires:** GPU with ≥8 GB VRAM for DINO training, ≥32 GB RAM for PatchCore bank

---

## Comparative Analysis Framework

When cloud results are available, update this section:

### Detection Rate Comparison

| Class | v6 (Baseline) | v8 (Tier 1) | v9 (Tier 2) | v10 (Tier 3) |
|-------|---------------|-------------|-------------|-------------- |
| Malignant | 63/76 (83%) | 60/76 (79%) | TBD | TBD |
| Benign | 61/76 (80%) | 59/76 (78%) | TBD | TBD |
| NP | 62/76 (82%) | 55/76 (72%) | TBD | TBD |
| Normal FP | 17/32 (53%) | 15/32 (47%) | TBD | TBD |

### YOLO Performance Comparison

| Metric | v6 | v8 | v9 | v10 |
|--------|-----|-----|-----|------|
| mAP50 | 0.181 | 0.075 | TBD | TBD |
| mAP50-95 | 0.045 | TBD | TBD | TBD |
| Malignant AP | — | — | TBD | TBD |
| Benign AP | — | — | TBD | TBD |
| NP AP | — | — | TBD | TBD |

### Key Questions to Answer

1. **Does Layer 4 help NP detection?** v9 vs v6/v8 NP detection rate
2. **Does DINO close the domain gap?** v10 Normal FP rate vs v6/v8
3. **Does DINO improve YOLO end-to-end?** v10 mAP50 vs all others
4. **Is the v8 regression recoverable?** v9/v10 should surpass v6 if the hypothesis is correct
5. **What is the calibration ceiling shift?** DINO backbone should produce lower Normal ceilings if the cluster is tighter

---

## File Inventory

| File | Purpose | Version |
|------|---------|---------|
| `src/generate_bboxes.py` | Core PatchCore + bbox pipeline | v10 (supports `backbone_path`) |
| `src/finetune_backbone.py` | DINO self-supervised fine-tuning | v10 (new) |
| `src/main.py` | CLI entry point | v10 (supports `--step finetune`, `--backbone`) |
| `src/train_yolo.py` | YOLOv11 training | v8 (session support) |
| `src/run_manager.py` | Timestamped run directories | v8 (session grouping) |
| `run_cloud.py` | Cloud orchestration script | v10 (new) |
| `models/patchcore_bank.npz` | ImageNet backbone bank cache | v9 |
| `models/patchcore_bank_dino.npz` | DINO backbone bank cache | v10 (created on cloud) |
| `models/dino_resnet50.pth` | DINO-finetuned backbone weights | v10 (created on cloud) |

---

## Cloud Execution Instructions

### Setup

```bash
# Clone/transfer project to cloud server
# Requires: Python 3.10+, NVIDIA GPU (≥8GB VRAM), ≥32GB RAM

cd Med_Img_Project
python -m venv .venv
source .venv/bin/activate   # Linux

# Install dependencies (GPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install psutil  # for RAM monitoring in run_cloud.py
```

### Run Full Comparative Study

```bash
# Complete study: DINO fine-tuning → v9 → v10 → comparison
python run_cloud.py --all

# Just v9 (if RAM is the only issue, no DINO needed):
python run_cloud.py --v9-only

# Resume after DINO is trained:
python run_cloud.py --all --skip-finetune
```

### Individual Steps

```bash
# DINO fine-tuning only:
python src/main.py --step finetune --finetune-epochs 100

# v9 generate + YOLO:
cd src && python main.py --step all

# v10 generate + YOLO (requires DINO backbone):
cd src && python main.py --step all --backbone dino
```

### Expected Timeline (GPU Server)

| Step | Estimated Time |
|------|---------------|
| DINO fine-tuning (100 epochs) | 2–4 hours (GPU) |
| v9 generate (3584-dim bank) | 15–20 min (GPU) |
| v9 YOLO training (75 epochs) | 10–15 min (GPU) |
| v10 generate (DINO bank) | 15–20 min (GPU) |
| v10 YOLO training (75 epochs) | 10–15 min (GPU) |
| **Total** | **~3–5 hours** |

---

## Improvement Roadmap (Future)

| Priority | Item | Tier | Status |
|----------|------|------|--------|
| 1 | v9 evaluation (Layer 4 features) | 2 | Code ready, needs cloud |
| 2 | v10 evaluation (DINO backbone) | 3 | Code ready, needs cloud |
| 3 | Multi-resolution ensembling | 2 | Not started |
| 4 | Two-stage detect & classify (EfficientNet) | 3 | Documented, not started |
| 5 | FAISS GPU-accelerated k-NN | 2 | Not started |
| 6 | Noise-aware YOLO training | 3 | Not started |
