# Version Registry — Endoscopic AI Pipeline

**Project:** Trustworthy Endoscopic AI Pipeline with PatchCore + YOLOv11  
**Last Updated:** 2026-03-14  
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
| v10 | 3 | DINO backbone | L2+L3+L4 (3584) | 300K | DINO | 0/32* | 1/76 (1%) | 1/76 (1%) | 4/76 (5%) | Train failed (dataset path) |
| v11 | 3 | ROI-aware glare rejection | L2+L3+L4 (3584) | 300K | ImageNet (selected) | 13/32 (40.6%) | 17/76 (22%) | 23/76 (30%) | 22/76 (29%) | Blocked by gate |
| v12 | 4 | Auto-calibration + A/B backbone sweep | L2+L3+L4 (3584) | 300K | ImageNet beats DINO | 13/32 (40.6%) | 17/76 (22%) | 23/76 (30%) | 22/76 (29%) | Blocked by gate |
| v13 | 4 | Soft-border + dual-threshold extraction | L2+L3+L4 (3584) | 300K | ImageNet (selected) | 22/32 (68.8%) | 66/76 (87%) | 70/76 (92%) | 66/76 (87%) | Gate passed (unsafe FP) |

> **Note:** `*` v10 Normal FP is based on the saved negative-control visualizations (all reported `peak=0.000`, no bbox overlays).

---

## Tier Classification

| Tier | Focus | Versions | Description |
|------|-------|----------|-------------|
| **Tier 0** | Baseline Development | v1–v6 | Core PatchCore implementation, calibration, scoring, bbox extraction |
| **Tier 1** | Artifact Suppression | v7–v8 | Pre-processing to remove non-pathological anomaly sources |
| **Tier 2** | Feature Extraction | v9 | Deeper features (Layer 4), multi-resolution potential |
| **Tier 3** | Backbone Adaptation | v10-v11 | DINO + ROI robustness; v11 recovered from corner-edge collapse but under-targeted lesions |
| **Tier 4** | Calibration Governance | v12-v13 | Sweep-driven sensitivity + extraction redesign (v13 recovered coverage but exposed gate weakness) |

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

**Status:** Executed locally, failed quality gate  
**Run:** `results/runs/20260311_184717_session/`  
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

**Observed outcomes (2026-03-11 run):**

| Metric | Value |
|---|---|
| Train labels with bbox | 6 / 228 (2.6%) |
| Val labels with bbox | 1 / 60 (1.7%) |
| Test labels with bbox | 3 / 96 (3.1%) |
| Edge-centered boxes (train) | 6 / 6 (100%) |
| Mean peak anomaly (train Benign/Malignant/NP) | 0.0015 / 0.0028 / 0.0047 |

**Failure analysis:**
- The anomaly map is near-zero for almost all lesion images and activates mostly on corner glare/border artifacts.
- Label sparsity is too severe for YOLO supervision (only 2.6% of train samples have any box).
- YOLO did not start because `configs/yolo_data.yaml` used `path: ../data/yolo_dataset`, which resolves outside the repo when run from project root.
- Fix applied on 2026-03-12: `configs/yolo_data.yaml` now uses `path: data/yolo_dataset`.

**Decision:**
- v10 is retained as a research checkpoint but **rejected for training** until detection coverage and spatial quality recover.

---

### v11 — ROI-Aware Glare Rejection (Tier 3)

**Status:** Evaluated (2026-03-13)  
**Run:** `results/runs/20260313_112259_generate/`

**Implemented changes:**
1. Endoscope field-of-view masking at feature-map level (`ENABLE_FOV_MASK=true`).
2. Border-connected component rejection (`BORDER_REJECT_MARGIN=4`).
3. Sensitivity from sweep-selected config (ImageNet):
   `calibration_percentile=99`, `ANOMALY_MARGIN_FRAC=0.08`, `ANOMALY_FLOOR_THRESH=0.03`.

**Observed outcomes:**
- Malignant train: 17/76 (22%)
- Benign train: 23/76 (30%)
- NP train: 22/76 (29%)
- Normal FP: 13/32 (40.6%)
- Train non-empty labels: 62/228 (27.2%)
- Edge-centered boxes: 0%
- Quality gate: **FAIL** (coverage below 30% threshold)

**Interpretation:**
- v11 successfully removed corner/edge artifacts (edge-box rate 0%).
- But lesion recall remained low and Normal FP remained high.
- This indicates the pipeline moved from "edge-glare dominated" to
  "under-targeted but still nonspecific" behavior.

---

### v12 — Auto-Calibration Quality Gate (Tier 4)

**Status:** Evaluated (2026-03-13)

**Implemented changes:**
1. Grid sweep over (`calibration_percentile`, `margin`, `floor`).
2. Objective function using abnormal coverage, normal FP, and edge-box penalty.
3. YOLO auto-block when generated labels fail quality thresholds.
4. Backbone A/B sweep (ImageNet vs DINO), selecting better objective.

**Observed outcomes:**
- ImageNet sweep best objective: 13.021 at (99, 0.08, 0.03)
- DINO sweep collapsed to all-zero objective under tested grid
  (all configs reported 0 abnormal coverage in sample sweep)
- Final selected backbone: ImageNet
- Final generation still failed quality gate (`overall_pass=false`)

**Important note:**
- `results/v11_sweeps/latest_sweep.json` is overwritten by the final backbone
  pass during v12 A/B. If DINO is swept second, the file may not reflect the
  actually selected global winner. The run log remains the source of truth.

---

### Root-Cause Analysis (v11-v12)

**Why false positives are still high (40.6% Normal FP):**
1. Historical pattern persists from older versions: specular/mucus/instrument
   patterns remain farther from the Normal bank than some true lesions.
2. Lowering margin to recover lesions increases diffuse internal false detections.
3. Current sweep objective still permits a high Normal FP if abnormal coverage improves.

**Why anomaly maps show signal but bboxes are not emitted:**
1. Signal is often present (`peak_anomaly_score >= 0.05`) but fragmented.
2. Border rejection at feature-map margin 4 (on 32x32) removes many near-wall
   connected components before bbox creation.
3. Area/morphology constraints filter weak/small components after thresholding,
   producing many `signal-but-no-box` outcomes.

**Evidence from confidence logs (latest run):**
- Train Malignant: 71 images with peak >= 0.05, but only 17 boxed.
- Train Benign: 67 images with peak >= 0.05, but only 23 boxed.
- Train NP: 72 images with peak >= 0.05, but only 22 boxed.

---

### v13 — Soft-Border + Dual-Threshold Extraction (Tier 4)

**Status:** Evaluated and replicated (2026-03-13 and 2026-03-14)  
**Run A:** `results/runs/20260313_161640_generate/`  
**Log A:** `results/logs/v13_advanced_20260313_154100.log`  
**Run B:** `results/runs/20260314_171000_generate/`  
**Log B:** `results/logs/v13_advanced_20260314_163000.log`

**Implemented changes:**
1. Soft border attenuation at feature-map score level (replacing hard border component rejection).
2. Dual-threshold seed-grow extraction (`high` seed + `low` grow mask).
3. Structured run outputs for failure-mode tracking:
   `run_info.txt`, `summary.json`, `summary.txt`, `artifacts_manifest.json`.
4. Confidence CSVs now include `signal_area_frac`, `bbox_area_frac`, `bbox_to_signal_ratio`.

**Observed outcomes:**
- Malignant train: 66/76 (87%)
- Benign train: 70/76 (92%)
- NP train: 66/76 (87%)
- Train non-empty labels: 202/228 (88.6%)
- Normal FP: 22/32 (68.8%)
- Edge-box rate: 9.42% overall
- Quality gate: **PASS** (`overall_pass=true`)

**Replication note:**
- The 2026-03-14 run is an independent repeat execution that reproduces the
  same key v13 outcome profile from 2026-03-13 (high abnormal coverage,
  high Normal FP, gate pass under current criteria).

**Interpretation:**
- v13 solved the low-coverage failure mode from v11/v12.
- However, specificity collapsed on held-out Normal controls.
- This is a governance failure mode: the current gate can pass while clinically
  risky false positives remain high.

**Box-vs-signal size diagnostics (v13):**
- Mean `bbox_to_signal_ratio` across confidence CSVs is ~1.31-1.52.
- Bboxes are moderately larger than heatmap support (expected from dilation,
  padding, and merging), but not catastrophically inflated in this run.

**v13 decision:**
- **Do not treat gate pass as approval for trusted YOLO training** until the
  gate includes Normal negative-control constraints.

---

## Comparative Analysis Framework

When cloud results are available, update this section:

### Detection Rate Comparison

| Class | v6 (Baseline) | v8 (Tier 1) | v10 (DINO) | v11/v12 (ImageNet+ROI+sweep) | v13 (soft-border + dual-threshold) |
|-------|---------------|-------------|------------|-------------------------------|-------------------------------------|
| Malignant | 63/76 (83%) | 60/76 (79%) | 1/76 (1%) | 17/76 (22%) | 66/76 (87%) |
| Benign | 61/76 (80%) | 59/76 (78%) | 1/76 (1%) | 23/76 (30%) | 70/76 (92%) |
| NP | 62/76 (82%) | 55/76 (72%) | 4/76 (5%) | 22/76 (29%) | 66/76 (87%) |
| Normal FP | 17/32 (53%) | 15/32 (47%) | 0/32* | 13/32 (40.6%) | 22/32 (68.8%) |

### YOLO Performance Comparison

| Metric | v6 | v8 | v10 | v11/v12 | v13 |
|--------|-----|-----|-----|------|
| mAP50 | 0.181 | 0.075 | Train failed (dataset path) | Blocked by label gate |
| mAP50-95 | 0.045 | TBD | TBD | Blocked by label gate | Not run (gate criteria under revision) |
| Malignant AP | — | — | TBD | Blocked by label gate | Not run |
| Benign AP | — | — | TBD | Blocked by label gate | Not run |
| NP AP | — | — | TBD | Blocked by label gate | Not run |

### Key Questions to Answer

1. **Can gate design include Normal-control constraints (e.g., Normal FP <= 20%)?**
2. **What objective weights avoid selecting high-recall/high-FP configurations?**
3. **Can per-class thresholds reduce NP/Benign collapse without increasing Normal FP?**
4. **Should v14 move to hybrid proposal+segmentation (Med-SAM) for refinement?**
5. **Can uncertainty-aware filtering remove low-trust pseudo-labels before YOLO?**

---

## File Inventory

| File | Purpose | Version |
|------|---------|---------|
| `src/generate_bboxes.py` | Core PatchCore + bbox pipeline | v13 (soft-border, dual-threshold, structured summaries) |
| `src/finetune_backbone.py` | DINO self-supervised fine-tuning | v10 (new) |
| `src/main.py` | CLI entry point | v13 (`--v12-advanced` running v13 extraction stack) |
| `src/train_yolo.py` | YOLOv11 training | v12 (automatic bad-label block) |
| `src/run_manager.py` | Timestamped run directories | v13 (run_info.txt + latest JSON + by-version pointers) |
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
| 1 | Gate v2: add Normal FP and confidence-calibrated precision constraints | 4 | Next |
| 2 | Per-class gate metrics and weighted objective redefinition | 4 | Next |
| 3 | v14 hybrid PatchCore proposal + Med-SAM refinement | 5 | Planned |
| 4 | Uncertainty-aware pseudo-label filtering before YOLO | 5 | Planned |
| 5 | Two-stage detect & classify (EfficientNet) | 5 | Documented, not started |
| 6 | FAISS GPU-accelerated k-NN | 2 | Not started |
