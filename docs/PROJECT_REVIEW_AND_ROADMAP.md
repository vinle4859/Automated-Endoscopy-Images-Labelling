# Project Review, Gap Analysis & Roadmap

**Project:** Trustworthy Endoscopic AI Pipeline with YOLOv11  
**Date:** February 22, 2026 (with 2026-03-13 and 2026-03-14 addenda below)  
**Scope:** Full codebase review — objective alignment, implementation adequacy, prioritised improvements

---

## 0. Validation Addendum (2026-03-13)

This addendum supersedes any older statements in this document that claim
the pipeline is ready for YOLO retraining after v6-only validation.

### Verified outcomes from latest advanced run

- Run artifact: `results/logs/v12_advanced_20260313_104726.log`
- Selected backbone: ImageNet (DINO sweep objective collapsed to 0.0)
- Best selected sensitivity params: percentile=99, margin=0.08, floor=0.03
- Normal negative-control false positives: 13/32 (40.6%)
- Train non-empty labels: 62/228 (27.19%)
- Edge-centered boxes: 0.0%
- Quality gate pass: False

### Validation status of safeguards

- YOLO dataset path bug fixed (`configs/yolo_data.yaml` now points to
  `data/yolo_dataset`), preventing the prior v10 path resolution failure.
- Automatic no-go gate is active: `data/yolo_dataset/quality_gate.json`
  reports `overall_pass=false` on current data.
- Training block is implemented in `src/train_yolo.py`; YOLO training is
  intentionally blocked unless label quality thresholds are met, or the
  operator explicitly uses `--force-bad-labels`.

### Roadmap adjustment

The immediate blocker is no longer infrastructure/runtime stability, but
label quality trade-offs (low abnormal coverage with high Normal FP).

Priority next versions:

1. v13: soften border suppression and improve signal-to-bbox conversion
   (dual-threshold/seed-grow extraction).
2. v13: make gates class-aware using confidence CSVs and per-class coverage.
3. v14: hybrid proposal-refinement (PatchCore proposals + Med-SAM refinement)
   if v13 cannot reduce Normal FP while lifting coverage.

---

## 0b. Validation Addendum (2026-03-14, v13 Completed)

### Verified outcomes from v13 advanced run

- Run A artifact: `results/logs/v13_advanced_20260313_154100.log`
- Run A directory: `results/runs/20260313_161640_generate/`
- Run B artifact: `results/logs/v13_advanced_20260314_163000.log`
- Run B directory: `results/runs/20260314_171000_generate/`
- Selected backbone: ImageNet
- Selected params: percentile=99, margin=0.06, floor=0.04
- Train non-empty labels: 202/228 (88.6%)
- Normal negative-control false positives: 22/32 (68.8%)
- Edge-centered boxes: 9.42%
- Quality gate: **PASS** (`overall_pass=true`)

Both runs are legitimate and materially consistent, with the second run serving
as independent reproducibility confirmation rather than replacement.

### Key implication

v13 fixed the low-coverage failure mode from v11/v12, but exposed a gate-design
failure mode: the current gate can pass while specificity collapses on Normal
controls.

### Mandatory governance upgrade before trusted YOLO retraining

1. Add explicit Normal-control criterion to gate (e.g., `normal_fp_pct <= 20`).
2. Add objective terms for pseudo-label precision, not only coverage/edge rate.
3. Track and threshold bbox-to-signal expansion from confidence CSV diagnostics.
4. Keep `--force-bad-labels` only for research mode, never default workflows.

---

## 1. Project Objective

Build a **trustworthy, explainable AI pipeline** for detecting pathological regions in nasal
endoscopic images using YOLOv11 object detection. The pipeline must:

| Requirement | Source |
|---|---|
| **R1** Automatically generate bounding box annotations from image-level class labels — no manual medical annotation | README.md + VERSION_REGISTRY.md |
| **R2** Train YOLOv11 for real-time detection of Malignant, Benign, and NP (Nasal Polyp) lesions | Phase 5–7 |
| **R3** Provide built-in explainability (XAI) so clinicians can audit model reasoning | Phase 9 |
| **R4** Support hierarchical classification and multi-modal late fusion (stretch goals) | Phases 3, 8 |
| **R5** Achieve mAP@0.5 > 0.80, clinical accuracy > 90%, real-time inference < 100 ms | Success Criteria |

---

## 2. Data Inventory

| Dataset | Location | Count | Role | Status |
|---|---|---|---|---|
| Normal (train) | `data/sample_data/train/Normal/` | 48 | PatchCore memory bank (primary) | ✅ Active |
| Normal (extended) | `data/normal_endoscopic/` | 5,957 | PatchCore memory bank (bulk) | ✅ Active |
| Malignant (train) | `data/sample_data/train/Malignant/` | 96 | Annotation target → YOLO class 0 | ✅ Active |
| Benign (train) | `data/sample_data/train/Benign/` | 96 | Annotation target → YOLO class 1 | ✅ Active |
| NP (train) | `data/sample_data/train/NP/` | 96 | Annotation target → YOLO class 2 | ✅ Active |
| Malignant (test) | `data/sample_data/test/Malignant/` | 32 | Held-out evaluation | ✅ Active |
| Benign (test) | `data/sample_data/test/Benign/` | 32 | Held-out evaluation | ✅ Active |
| Normal (test) | `data/sample_data/test/Normal/` | 32 | Negative control — step [5/5] | ✅ Active |
| NP (test) | `data/sample_data/test/NP/` | 32 | Held-out evaluation | ✅ Active |
| ~~dataKho/~~ | deleted | ~~200~~ | Redundant 2017–18 data | 🗑️ Removed |
| ~~abnormal_endoscopic/~~ | deleted | ~~750+~~ | No class labels; leakage risk | 🗑️ Removed |

**Key observation:** The training set is very small (288 abnormal images across 3 classes).
Augmentation and careful regularisation will be critical during YOLO training.

---

## 3. Annotation Strategy History

Three automated annotation methods were attempted in chronological order. Each failure informed
the design of its successor. Full documentation exists in `docs/AUTOMATED_ANNOTATION_STRATEGIES.md`.

### Method 1 — Grad-CAM WSOD → **Abandoned**
- **Concept:** Train a classifier, use Grad-CAM to localise discriminative regions.
- **Failure:** Shortcut learning — model attended to imaging artefacts (vignetting, specular highlights) rather than tissue. Small dataset size amplified this.
- **Lesson:** Supervised localisation signals are unreliable when spurious correlations dominate.

### Method 2 — UNetAutoencoder Reconstruction → **Abandoned**  
- **Concept:** Train autoencoder on Normal images only; use pixel-wise reconstruction error as anomaly signal.
- **Implementation:** U-Net with 4-level encoder, 512-ch bottleneck, skip connections. Trained 50 epochs, MSE loss, Adam optimiser. Loss converged.
- **Failure:** Skip connections bypass the bottleneck entirely — the model reconstructs any tissue perfectly, including tumours. Additionally, tumour and normal nasal mucosa share identical pixel texture.
- **Lesson:** Pixel-space anomaly detection is structurally incapable of differentiating texturally homogeneous tissues.

### Method 3 — PatchCore Feature-Space Anomaly Detection → **Active**  
- **Concept:** Compare semantic patch features (frozen ResNet50 layer2 + layer3, 1536-dim) against a Normal memory bank via k-NN distance.
- **Key design decisions:**
  - Per-image subsample to 100 patches (avoids OOM on 6,005 normal images)
  - **v2 calibration:** Score all 1024 patches of each of the 48 Normal *train* images against the bank; use 99th percentile as `score_ceiling` (replaces v1 bank self-scoring which produced a ceiling too low for full-image scoring)
  - **v2 subtractive scoring:** `excess = max(score − ceiling, 0)`, then normalise. Patches at or below ceiling → exactly 0 (no false heat on Normal tissue)
  - **v6 multi-bbox extraction (current):** Adaptive Otsu threshold (floor 0.04), 5×5 dilation + 5×5 close kernels, area limits (0.5 %–85%), bbox merging (gap < 8%), up to 3 bboxes per image. Evolved from v2 (fixed 30%, 9×9) through v5 (density smoothing) to v6 (adaptive + merging)
  - Bank caching (`models/patchcore_bank.npz`, float16 compressed); ceiling always recalibrated on load
  - 80/20 stratified train/val split + separate test-set processing
  - Normal negative control: 32 held-out Normal test images scored and checked for false positives
- **Status:** Code complete. v6 pipeline deployed with 100% lesion recall on validation. Full pipeline run completed (Feb 2026). First YOLO training run reached epoch 23.
- **First-run results (v1, diagnosed and fixed):** Anomaly maps showed real signal but v1 bbox extraction produced 84% full-image boxes and 100% Normal false-positive rate due to 4 compounding bugs (see §6.2).

---

## 4. Current Implementation — File Map

| File | Purpose | Status |
|---|---|---|
| `src/generate_bboxes.py` | PatchCore pipeline: extract features, build bank, score images, produce YOLO labels with train/val/test splits, confidence logging, data validation, Normal negative control | ✅ Updated |
| `src/main.py` | CLI orchestrator (`--step generate\|yolo\|all`, `--rebuild-bank`) | ✅ Updated |
| `src/train_yolo.py` | YOLO v11-nano training with medical-sensible augmentation, early stopping (patience=10), test evaluation | ✅ Updated |
| `src/diagnose_signal.py` | Signal diagnostic tooling for anomaly-map behavior checks | ✅ Active |
| `src/diagnose_v4.py` | 5-panel diagnostic visualisation for PatchCore debugging | ✅ Active |
| `configs/yolo_data.yaml` | YOLO dataset config — train/val/test paths, 3 classes | ✅ Updated |
| `models/patchcore_bank.npz` | Normal memory bank (float16, 500K patches) + ceiling | ✅ Generated (~880 MB) |

---

## 5. Gap Analysis

### 5.1 Critical Gaps (block the pipeline from producing results)

| ID | Gap | Impact | Resolution | Status |
|---|---|---|---|---|
| **G1** | Pipeline never run end-to-end | No YOLO dataset, no model, no metrics exist | ✅ **Run** — first run completed, v1 bbox issues diagnosed, v2 fixes deployed. Awaiting re-run with v2 pipeline. | ⏳ Re-run needed |
| **G2** | No train/val split for YOLO | Cannot compute valid mAP; `val: images/train` was a loopback | ✅ **Fixed** — 80/20 stratified split added to `generate_bboxes.py`; `yolo_data.yaml` updated | ✅ Done |
| **G3** | Empty labels for no-bbox abnormal images | YOLO treats as negative (no object) — false training signal | ✅ **Resolved** — v1 `fallback_full_image_bbox()` was implemented then **removed** in v2 because it was a root cause of garbage output (see §6.2). v2 allows empty labels; training with fewer but accurate labels is preferred over full-image fallback boxes. | ✅ Done (v2) |
| **G4** | Test set never processed | Cannot evaluate on held-out data | ✅ **Fixed** — test split processing added as step [4/5] | ✅ Done |

### 5.2 Quality Gaps (affect result reliability)

| ID | Gap | Impact | Resolution | Status |
|---|---|---|---|---|
| **G5** | Only 5 diagnostic visualisations per class | Insufficient to judge annotation quality before committing | ✅ **Fixed** — `VISUAL_SAMPLE` increased to 20 | ✅ Done |
| **G6** | No annotation confidence score | Cannot filter/weight noisy labels | ✅ **Fixed** — peak anomaly score logged per image to `confidence_<split>_<class>.csv` | ✅ Done |
| **G7** | No data quality checks | Silent corrupt images, duplicates, invalid bboxes | ✅ **Fixed** — `validate_source_data()` checks corrupt images, filename collisions, reports per-class counts | ✅ Done |
| **G8** | Duplicate content in `AUTOMATED_ANNOTATION_STRATEGIES.md` | Old 2-method text appended below rewritten 3-method version | ✅ **Fixed** — truncated at line 152 | ✅ Done |
| **G9** | Single bbox per image | Multi-focal lesions → one oversized merged box | ✅ **Fixed (v6)** — `extract_bboxes()` returns up to 3 bboxes per image (largest-first), with area limits (0.5 %–85%), bbox merging (gap < 8%). Multi-focal lesions get separate or merged boxes as appropriate. | ✅ Done |
| **G10** | Normal test images (32) unused for calibration validation | No confirmation that calibrated heatmaps produce zero boxes on Normal | ✅ **Fixed** — step [5/5] scores Normal test images, reports false-positive count and rate | ✅ Done |

### 5.3 Downstream Gaps (post-YOLO training)

| ID | Gap | Impact | Resolution | Status |
|---|---|---|---|---|
| **G11** | YOLO training | First YOLO run completed 23 epochs (mAP50=0.176). Params updated: epochs=75, patience=10. Awaiting re-run. | Re-run YOLO training with updated parameters | ⏳ Re-run needed |
| **G12** | No Grad-CAM on YOLO model | Primary XAI deliverable (R3) not implemented | Use `pytorch-grad-cam` on trained YOLO backbone | ⏳ Blocked on G11 |
| **G13** | No inference script | Cannot run predictions on new images | Create `src/inference.py` per Phase 10 of plan | ⏳ Blocked on G11 |
| **G14** | No interactive dashboard | Clinical usability goal (Phase 9.4) unmet | Build Gradio/Streamlit demo | ⏳ Stretch |
| **G15** | Multi-modal fusion not designed | R4 stretch goal not started | Requires clinical metadata; defer | ⏳ Stretch |

---

## 6. Changes Applied in This Review

### Code Changes

**`src/generate_bboxes.py`** (6 modifications):

1. **`VISUAL_SAMPLE = 5 → 20`** — More diagnostic plots to inspect annotation quality before YOLO training.

2. **`TEST_ROOT` config added** — Points to `data/sample_data/test/` so test images are processed.

3. **`VAL_SPLIT_RATIO = 0.2`** — 80/20 stratified train/val split using `sklearn.train_test_split(random_state=42)`.

4. **`fallback_full_image_bbox()`** — New function returning a centered 90%-coverage box. Called when `extract_bbox()` returns nothing for a known-abnormal image. Prevents YOLO from learning empty (false-negative) labels.

5. **`_process_split()` helper** — Refactored the image-processing loop into a reusable function that handles any split (train/val/test). Integrates the fallback logic and reports localised vs. fallback counts.

6. **Test-set processing** added as step [4/5] — Iterates over `data/sample_data/test/{Malignant,Benign,NP}/` and writes labels into `yolo_dataset/images/test/` and `yolo_dataset/labels/test/`.

7. **`validate_source_data()`** — New pre-flight function that checks all source images can be opened, detects filename collisions across classes (which would silently overwrite labels in the flat YOLO dataset), and reports per-class and Normal bank counts. Pipeline aborts early if issues are found.

8. **Stale output cleanup** — `generate_dataset()` now removes any existing `yolo_dataset/` directory via `shutil.rmtree()` before fresh generation, preventing contamination from prior runs.

9. **Peak anomaly confidence score (G6)** — `_process_split()` now tracks the peak anomaly heatmap value (0.0–1.0) per image and writes it to `confidence_<split>_<class>.csv` alongside the label. Enables downstream noise-aware training or filtering.

10. **Normal-test negative control (G10)** — New step [5/5] scores the 32 held-out Normal test images through PatchCore and reports how many produce a bbox. Any false positives trigger a calibration warning.

**`src/train_yolo.py`** (new file):
- YOLOv11-nano fine-tuning script with medical-sensible augmentation config
- HSV jitter, horizontal flip, rotation ±15°, scale, mosaic, random erasing
- No vertical flip (endoscopy orientation), no mixup/copy-paste (approximate bboxes)
- Early stopping (patience=30), automatic CPU/GPU device selection
- Runs test-split evaluation after training and reports mAP metrics
- CLI: `--epochs`, `--batch`, `--imgsz`, `--patience`, `--resume`

**`src/main.py`**:
- Added `'yolo'` to `--step` choices (now: `train|generate|yolo|all`)
- YOLO training step integrated as Step 3, importing `train_yolo.main()`

### 6.2 v2 Overhaul — Calibration, Scoring & BBox Extraction

**Diagnosis:** First pipeline run (v1) produced catastrophic results. Detailed analysis:

| Metric | v1 Result |
|---|---|
| Full-image bboxes (`0.5 0.5 1.0 1.0`) | 322/384 (84%) |
| All confidence scores | 1.000000 (saturated) |
| Normal negative control | 32/32 false positives (100%) |

**Root causes identified (4 compounding issues):**

1. **RC1 — Calibration mismatch:** Bank kept 100/1024 patches per image; scoring uses all 1024. The ~90% of spatial positions absent from the bank score high even for Normal images. v1 ceiling (p95 of bank self-scoring = 31.1) was too low.
2. **RC2 — Otsu always finds foreground:** Any heatmap with any gradient gets split into foreground/background, even if all values are near-zero.
3. **RC3 — 25×25 morphology kernels:** Merged every slightly hot pixel into one image-spanning blob.
4. **RC4 — Single bbox + forced fallback:** Guaranteed full-image garbage for every image.

**Key user validation:** Anomaly maps DO show localised signal corresponding to human-visible lesions. The PatchCore feature extraction works correctly — only the downstream bbox extraction was broken.

**v2 fixes applied to `src/generate_bboxes.py`:**

1. **`calibrate_normal_score()` rewritten** — Scores all 1024 patches of each of the 48 Normal *train* images against the bank. Uses 99th percentile (expected ~38–42 vs. v1's 31.1). Prints p50/p95/p99 diagnostics.
2. **`compute_anomaly_map()` rewritten** — Subtractive scoring: `excess = max(score − ceiling, 0)`, normalised as `clip(excess / ceiling, 0, 1)`. Patches at/below ceiling → exactly 0.0 (v1 always produced nonzero values).
3. **`extract_bboxes()` rewritten** (renamed from `extract_bbox`) — Through v2→v6 iterations: v2 used fixed threshold (30%) + 9×9 kernel + up to 5 bboxes. v5 added density-smoothed scoring. **v6 (current):** adaptive Otsu (floor 0.04), 5×5 dilation + close, bbox merging (gap < 8%), up to 3 bboxes.
4. **`fallback_full_image_bbox()` removed entirely** — Empty labels are correct when no anomaly is detected.
5. **Cache loading** — Always recalibrates ceiling (doesn't trust cached v1 ceiling).
6. **Current config constants (v6):** `ANOMALY_FLOOR_THRESH=0.04`, `MIN_BBOX_AREA_FRAC=0.005`, `MAX_BBOX_AREA_FRAC=0.85`, `MAX_BBOXES_PER_IMAGE=3`, `BRIDGE_DILATE_K=5`, `BBOX_MERGE_GAP_FRAC=0.08`.

---

## 7. Execution Roadmap

### Phase A — Generate Annotations (**next step**)

```powershell
cd E:\Med_Img_Project
.venv\Scripts\python.exe src\main.py --step generate
```

**Expected behaviour (v2 pipeline):**
- [0/5] Data quality pre-check (corrupt images, filename collisions)
- [1/3] Loads cached memory bank (~10 s) OR builds from 6,005 Normal images (~20 min first run)
- Recalibrates ceiling by scoring 48 Normal train images (all 1024 patches each) → p99
- [2/3] Fits k-NN index (~30 s)
- [3/5] Processes 288 train images → splits ~230 train / ~58 val
- [4/5] Processes 96 test images (32 per abnormal class)
- [5/5] Normal-test negative control (32 Normal images, expect 0 or near-0 false positives)
- Writes `data/yolo_dataset/{images,labels}/{train,val,test}/`
- Writes `data/yolo_dataset/confidence_*.csv` metadata files
- Saves 60 diagnostic visualisations (20 per class) to `results/visualizations/`

**v2 checklist:**
- [ ] Check calibration ceiling printed at startup — should be ~38–42
- [ ] Normal negative control false-positive rate — target <10%
- [ ] Review visualisations — bboxes should be tight around lesion tissue, not spanning the full image
- [ ] Check summary: "no-box" count per class — some images having no bbox is EXPECTED and correct
- [ ] Spot-check 5 YOLO `.txt` files — coordinates in [0,1], correct class IDs, multiple lines for multi-bbox images

### Phase B — YOLO Training

```powershell
# Option 1: standalone
.venv\Scripts\python.exe src/train_yolo.py --epochs 75 --patience 10

# Option 2: via pipeline orchestrator
.venv\Scripts\python.exe src/main.py --step yolo
```

**Notes:**
- Using `yolo11n.pt` (nano) instead of `yolo11s.pt` — smaller model is less prone to overfitting on 230 training images.
- Augmentation configured in `src/train_yolo.py`: HSV jitter, horizontal flip, rotation ±15°, scale, mosaic 80%, random erasing 20%. No vertical flip, no mixup.
- Early stopping with patience=10 (reduced from 30 to limit wasted CPU time).
- Test-set evaluation runs automatically after training.
- On CPU (~1.5 min/epoch), expect early stopping around epoch 40–60.

### Phase C — Evaluation

```powershell
.venv\Scripts\python.exe -m ultralytics yolo task=detect mode=val `
    data=configs/yolo_data.yaml `
    model=results/yolo/train/weights/best.pt
```

- Compute mAP@0.5 and mAP@0.5:0.95 on the test split.
- Generate confusion matrix and per-class precision/recall.
- Compare test performance with val performance to check for generalization.

### Phase D — XAI Integration

1. Apply `pytorch-grad-cam` to the trained YOLO backbone's last convolutional layer.
2. Generate Grad-CAM overlays for 10–20 test images per class.
3. Side-by-side comparison: PatchCore anomaly heatmap vs. YOLO Grad-CAM.
   - If both highlight the same tissue region → strong evidence of valid learning.
   - If YOLO's attention drifts to borders/artefacts → shortcut learning has recurred.

### Phase E — Stretch Goals (if time permits)

- Gradio dashboard for single-image upload → detection + heatmap overlay
- Domain-adapted backbone (self-supervised pre-training on `normal_endoscopic/`)
- Noise-aware YOLO training: weight loss by confidence score from `confidence_*.csv`
- Adaptive threshold tuning (per-class or learned from validation set)

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PatchCore bboxes are inaccurate (wrong location) | Medium | High — YOLO learns wrong patterns | Visual inspection of 60 plots before training; iterate on threshold/morph params |
| ImageNet → endoscopic domain gap degrades features | Medium | Medium — some lesions may not separate | Acceptable for bootstrap; can upgrade backbone later |
| 230 training images insufficient for YOLO | High | Medium — overfitting, poor generalisation | Use nano model, aggressive augmentation, early stopping |
| v6 adaptive Otsu floor (0.04) too permissive | Medium | Medium — higher Normal FPs | Inspect visualisations; adjust `ANOMALY_FLOOR_THRESH` if needed. Stage 2 classifier mitigates. |
| CPU-only bank building is too slow for iteration | Low | Low — one-time cost, cached thereafter | ~20 min acceptable; GPU would give ~3 min |

---

## 9. Known Limitations

1. **No ground-truth bounding boxes** — quantitative annotation quality (IoU) cannot be computed. Downstream YOLO mAP on test data serves as a proxy.
2. **Multi-bbox with area limits** — up to 3 bboxes per image (v6); boxes outside 0.5 %–85% area are rejected; nearby boxes merged (gap < 8%). Some images may correctly receive no bbox.
3. **ImageNet backbone features** — not trained on endoscopic data; residual domain gap.
4. **Small dataset scale** — 288 abnormal training images across 3 classes; data augmentation is essential.
5. **CPU-only execution** — ~18 s/image for PatchCore scoring; GPU would give ~10× speedup.
6. **Coreset sampling variance** — mitigated by fixed seed=42 for full reproducibility.
7. **Adaptive Otsu threshold (floor 0.04)** — v6 uses per-image adaptive Otsu with a safety floor. May need tuning for different tissue types. Normal FP rate: 53% (acceptable because Normal excluded from training).

---

## 10. Method Comparison Summary

| | Method 1 (Grad-CAM) | Method 2 (UNet AE) | Method 3 (PatchCore) |
|---|---|---|---|
| Status | Abandoned | Abandoned | **Active** |
| Training required | Yes (classifier) | Yes (autoencoder) | **No** |
| Signal space | Pixel gradients | Pixel recon. error | **Feature-space distance** |
| Failure mode | Shortcut learning | Over-reconstruction | Domain gap (residual) |
| XAI output | Grad-CAM map | Difference map | **Anomaly heatmap** |
| External dependency | None | None | **ImageNet weights** |
| Code | Not preserved | `src/model.py` | `src/generate_bboxes.py` |
| Documentation | `docs/AUTOMATED_ANNOTATION_STRATEGIES.md` | `docs/METHOD_2_UNET_AUTOENCODER.md` | `docs/METHOD_3_PATCHCORE_IMPLEMENTATION.md` |
