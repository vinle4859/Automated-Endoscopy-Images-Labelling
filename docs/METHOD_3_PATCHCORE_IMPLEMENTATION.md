# Method 3: PatchCore Feature-Space Anomaly Detection

**Project:** Trustworthy Endoscopic AI with YOLOv11
**Status:** Active Implementation
**Adopted:** After Method 2 (UNetAutoencoder) was fully trained and evaluated but failed due to
over-reconstruction of pathological tissue (see `docs/METHOD_2_UNET_AUTOENCODER.md`).

Reference: *Towards Total Recall in Industrial Anomaly Detection*, Roth et al., CVPR 2022.

---

## Motivation: Why Method 2 Failed and Why This Solves It

Method 2 operated in **pixel space**. The fundamental assumption was that a model trained only
on Normal images would produce high reconstruction error at the location of a tumour. This
assumption broke down because nasal endoscopic tumour tissue and normal nasal mucosa are
visually indistinguishable at the pixel level — same colour spectrum, same lighting, same camera.
The U-Net skip connections compounded this by passing exact spatial detail from encoder to
decoder, enabling near-perfect reconstruction of any content.

Method 3 operates in **semantic feature space**. A frozen ResNet50 (trained on 1.2M ImageNet
images) encodes each image patch as a 1536-dimensional vector capturing tissue organisation,
surface topology, and structural patterns — not pixel values. Normal endoscopic tissue forms a
tight cluster in this space. Pathological tissue, whose structural organisation differs from
normal at a semantic level even when pixels look alike, appears as a distant outlier.

| Dimension | Method 2 — UNetAutoencoder | Method 3 — PatchCore |
|---|---|---|
| Signal source | Pixel reconstruction error | Feature-space k-NN distance |
| Failure mode | Tumours share pixel texture → near-zero error | None observed (domain gap is a residual risk) |
| Training required | Yes — 50 epochs on Normal images | No — frozen ImageNet weights |
| Inference time | Fast (single forward pass) | Moderate (~1 s/image, k-NN over 600K patches) |
| Explainability output | Difference map (unreliable) | Anomaly heatmap (globally calibrated, reliable) |

---

## Architecture

### Feature Extractor — `PatchCoreExtractor`

```
Input image (256 × 256 × 3)
        │
   ResNet50 frozen backbone (ImageNet weights, no gradient updates)
        ├── stem + layer1 + layer2  →  f2: (512, 32, 32)    mid-level semantics
        ├── stem + … + layer3      →  f3: (1024, 16, 16)   deep semantics
        │                                    ↓ bilinear upsample to (32×32)
        └── stem + … + layer4      →  f4: (2048, 8, 8)     morphological semantics (v9)
                                              ↓ bilinear upsample to (32×32)
        concatenate f2 + f3 + f4  →  (3584, 32, 32)
```

Each of the 32×32 = **1024 spatial positions** carries a 3584-dim feature vector describing a
~8×8 pixel patch of the original image.

**Why layer2 + layer3 + layer4?** Layer2 preserves spatial localisation; layer3 captures
semantic context; layer4 (added in v9) encodes deeper structural/morphological patterns
(tissue organisation, gland architecture) that help differentiate NP (structural anomaly)
from Normal mucosa — the primary weakness in v6–v8. The original PatchCore paper used
layer2+layer3; we extend to layer4 because endoscopic lesion detection requires
morphological features that only emerge at the deepest levels.

### Memory Bank

All images from Normal directories are processed through `PatchCoreExtractor`. Each produces
1024 patch vectors. To stay within RAM limits:
- **Per-image cap:** 100 patches randomly sampled per image (seed=42)
- **Resulting bank:** 300,000 patches × 3584 features (v9: capped from ~600K raw) ≈ 4.1 GB (float32) → ~2.0 GB on disk (float16 compressed, npz)
- **k-NN algorithm:** `brute` — mandatory for 3584-dim; tree-based methods degrade above ~15 dims

The bank is saved to `models/patchcore_bank.npz` after first build. Subsequent runs load in
seconds instead of rebuilding from scratch.

### Score Calibration — `calibrate_normal_score` (v2)

**Problem with naive per-image normalisation:** `score = (s - min) / (max - min)` maps every
image to [0, 1] regardless of whether any anomaly is present. This always produces a hot region
in the heatmap — on normal tissue, on endoscope borders, on specular reflections.

**v1 approach (deprecated):** Score 5,000 randomly sampled bank patches against the bank itself
(k+1 neighbours). The 95th percentile self-score (∱31.1) was used as `score_ceiling`. This
failed because the bank keeps only 100/1024 patches per image — the ~90% of spatial positions
not in the bank produce high k-NN distances even for Normal images.

**v2 approach (current):** Score ALL 1024 patches of each of the 48 Normal *train* images
against the full bank (k=5 neighbours). The **99th percentile** of all 48×1024 = 49,152 patch
scores becomes `score_ceiling` (expected ~38–42). This captures the true maximum distance
that Normal tissue exhibits when scored at full spatial resolution.

**v2 anomaly map scoring (subtractive):**
```
excess = max(raw_knn_distance − score_ceiling, 0)
score_map = clip(excess / score_ceiling, 0, 1)
```

Normal patches (distance ≤ ceiling) → exactly 0.0. Patches at 2× ceiling saturate to 1.0.
This eliminates false heat on Normal tissue, which v1's `clip(score / ceiling, 0, 1)` could not.

`score_ceiling` is stored in `patchcore_bank.npz` alongside the bank, but is always
recalibrated on load (the calibration method may have changed between versions).

### Bounding Box Extraction — `extract_bboxes` (v6)

```
Anomaly heatmap (uint8, original resolution)
    │
    ├── Downsample to feature-map resolution (32×32, INTER_AREA)
    ├── Adaptive threshold:
    │     ├── Compute Otsu on nonzero pixels (above-ceiling anomaly signal)
    │     └── Floor at ANOMALY_FLOOR_THRESH (0.04) to prevent noise
    ├── Dilation (5×5 kernel) — bridges hotspot gaps within same lesion
    ├── Morphological CLOSE (5×5 kernel) — fills internal holes
    ├── connectedComponentsWithStats — one component per anomaly cluster
    ├── Per-component: convert to normalised coords + padding (5% each side)
    ├── Area filter: reject < 0.5% or > 85% of image
    ├── Bbox merging: fuse boxes whose edges are within 8% gap
    │     └── Iterative: keeps merging until no more pairs are close
    └── Sort by area (largest first), keep up to 3 bboxes
```

### Anomaly Map — `compute_anomaly_map` (v7)

```
Input image (256×256)
    │
    ├── ResNet50 feature extraction → (1536, 32, 32)
    ├── k-NN scoring (k=5, brute, euclidean) → raw score map (32×32)
    ├── Subtractive baseline: excess = max(score - ceiling, 0)
    ├── Edge suppression: zero 2-px border
    ├── **Specular suppression (v7):** zero patches where >25% of receptive
    │         field is specular (HSV V>240 & S<30) BEFORE smoothing
    ├── Gaussian smooth (σ=2.0, k=7) at feature-map resolution
    ├── Normalize: clip(excess / (ceiling × 0.15), 0, 1)
    └── Upscale to original resolution → uint8 heatmap
```

**v7 specular suppression rationale (2026-02-24):**
Diagnostic analysis revealed that specular reflections on Normal tissue
push Normal peak scores to 41.9 (vs ceiling 37.7), narrowing the gap to
the weakest true anomalies (NP at 42.5) to only 0.6 raw score units.
v5 relied on density smoothing alone to dilute speculars, but diagnostic
showed 33% of above-ceiling patches in Normal images are specular.
v7 zeroes these patches in the excess map *before* smoothing, so they
cannot contaminate the density estimate. This operates at feature-map
resolution (32×32) with a 25% coverage threshold, avoiding the
over-suppression problems of v4/v5-alpha (which operated at full pixel
resolution).

**v1–v4 progression:**
- v1: Otsu + 25×25 morph + single-bbox → 84% full-image garbage
- v2: Fixed 30% threshold + 9×9 morph + multi-bbox + area filter
- v3: Same logic but full-resolution contours → sparse, failed at scale
- v4: Feature-map-level extraction (32×32 CC analysis) → better detection
  but fragmented bboxes on same tumor
- v5: Density-smoothed scoring (Gaussian blur σ=2.0 at feature level)
  → large central bboxes, but still fragmented on large lesions

**v6 design rationale:**
- **Adaptive threshold** — Otsu on nonzero pixels finds the natural split between
  background and anomaly signal, adapting per-image.  This captures broader
  low-intensity anomaly regions (like ACC-1) that v5's fixed 0.08 threshold missed.
  Floor at 0.04 prevents noise activation on Normal images.
- **Larger kernels** — 5×5 dilation + 5×5 close at 32×32 resolution bridges wider
  gaps between hotspot clusters that belong to the same tumor mass.
- **Bbox merging** — After CC extraction, nearby bboxes (edge gap < 8%) are fused.
  A single tumor with 2–3 hotspot peaks now gets ONE bounding box, not 2–3.
  This is critical for YOLO training quality.
- **No fallback** — empty label files for images with no detection.

**v7 design rationale (current):**
- **Specular suppression before smoothing** — Expert-validated fix for the "tight
  signal gap" problem.  Diagnostic (2026-02-24) showed specular reflections on
  Normal tissue (peak@spec=41.9) push the Normal baseline to within 0.6 of the
  weakest true anomalies (NP at 42.5).  Zeroing specular patches from the excess
  map *before* Gaussian smoothing removes this contamination without over-suppression.
- **Shared `get_specular_mask()` function** — Used by both the pipeline and the
  diagnostic script, ensuring consistent specular detection across all tools.
- **No bank rebuild required** — Specular suppression operates on the excess map
  at runtime; the memory bank and features are unchanged.

---

## Verification & Validation

### V1 — Feature Extractor Slicing
`children = list(resnet50.children())` gives:
`[conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]`

- `children[:6]` = conv1 + bn1 + relu + maxpool + layer1 + **layer2** ✅
- `children[:7]` = above + **layer3** ✅

Both paths share the same stem — no duplicate computation because PyTorch traces through
`self.layer2` (which includes the stem) and `self.layer3` (which also includes the stem).
This is intentional: `layer3` simply extends `layer2`'s computation graph.

### V2 — Memory Bank RAM Budget
```
6,005 images × 100 patches = 600,500 raw → capped to 500,000 (MAX_BANK_PATCHES)
500,000 × 1536 dims × 4 bytes = 3.07 GB peak (float32)
Stored as float16 compressed → ~880 MB on disk
```
✅ Within 5 GB constraint.

### V3 — k-NN Algorithm Choice
scikit-learn documentation: *"For large D [dimensions], brute force is preferred."*
Tree-based methods (ball_tree, kd_tree) split space using hyperplanes; in 1536 dimensions, the
curse of dimensionality makes every point approximately equidistant from query points, so tree
traversal provides no pruning benefit and is slower than brute force.
✅ `algorithm='brute'` is correct.

### V4 — Calibration Statistical Validity
**v2 calibration:** 48 Normal train images × 1024 patches = 49,152 distance samples.
The 99th percentile of 49,152 samples is statistically robust.
This approach matches the actual scoring pipeline (all 1024 patches per image) — unlike v1
which sampled 5,000 patches from the bank and used k+1 neighbours to avoid self-hits.
✅ v2 calibration methodology is sound.

### V5 — Global vs Per-Image Normalisation
Per-image min-max normalization was the **initial implementation** and was identified as a
fundamental flaw after inspecting the first visualisation batch: every image — normal or
pathological — produced a full-range heatmap, always triggering a bounding box. This was the
mirror problem to Method 2 (which never fired). Global calibration was implemented in response.
✅ Confirmed fix. Normal tissue now produces an all-zero heatmap (v2 subtractive scoring);
anomalous regions produce targeted hot zones only when they exceed the Normal ceiling.

### V6 — v2→v6 BBox Extraction Evolution
v1 used Otsu + 25×25 morphological kernels + single-bbox. Diagnosed as a root cause of 84%
full-image garbage output. v2 introduced fixed threshold (30%) + 9×9 kernel + multi-bbox + area
limits. v5 added density-smoothed scoring. **v6 (current)** replaced the fixed threshold with
adaptive Otsu (floor=0.04), reduced kernels to 5×5, added bbox merging (gap < 8%), and
capped to 3 bboxes per image.
✅ v6 bbox extraction addresses all root causes from v1–v5 failure analyses.

### V7 — v5 Density Smoothing
Gaussian blur (σ=2.0, kernel=7) at 32×32 feature-map resolution acts as spatial density
estimation. Large tumor regions (many nearby moderate-anomaly patches) maintain strong signal
after smoothing, while isolated artifacts (specular, edge, texture noise) are diluted.
This replaced explicit specular-mask suppression (which over-suppressed high-specular images).
✅ Normal images: 0% false positive rate. Malignant: large central bboxes matching tumor location.

### V8 — v6 Adaptive Threshold + Bbox Merging
Fixed threshold (v5: 0.08) was too rigid — missed broad low-intensity anomaly regions
(Malignant_ACC-1) and produced fragmented bboxes on the same tumor mass.
v6 fixes:
- **Adaptive Otsu** on nonzero pixels per-image, floored at 0.04
- **5×5 dilation + 5×5 close** at feature resolution (bridges wider gaps)
- **Bbox merging** (edge gap < 8%) fuses nearby boxes into one cohesive annotation
✅ Addresses: (a) fragmentation, (b) partial border captures, (c) rigid threshold.

**v6 Validation Results (Feb 2026):**
- Lesion recall: 9/9 (100%) — all Malignant, Benign, NP images produce at least 1 bbox
- Normal false positives: 2/3 (67%) — peaks overlap with true positives (fundamental indistinguishability)
- Key wins: H01_IP_001 (Benign, peak=0.059) and H01_NP_001 (NP, peak=0.071) both caught — previously missed by v5
- Trade-off: higher sensitivity → Normal FPs acceptable because Normal images are excluded from YOLO training
- Validates the need for the two-stage architecture (Stage 2 classifier filters Normal FPs at inference time)

### V9 — v7 Specular Suppression (Expert-Validated Fix)

**Problem diagnosed (2026-02-24):** Signal diagnostic on 10 representative images revealed
the "tight signal gap" problem — Normal images with specular reflections score up to 41.9
(4.2 above ceiling 37.7), while the weakest true anomalies (NP) score 42.5 (gap of only 0.6).
Specular reflections inflate Normal baselines because ImageNet ResNet50 has never seen
bright desaturated patches on mucosal surfaces.

**Diagnostic evidence:**

| Image | Class | peak@spec | peak@tissue | Gap to ceiling |
|---|---|---|---|---|
| H16_JNA_01_crop | Benign | 43.0 | 45.5 | +7.8 |
| H01_SCC_011 | Malignant | 42.4 | 43.2 | +5.5 |
| H01_NP_041 | NP | 0.0 | 42.5 | +4.8 |
| normal 039 | Normal | **41.9** | 41.5 | +3.8 (FP risk) |
| normal 055 | Normal | 35.3 | 41.6 | +3.9 (FP risk) |

**Fix:** `get_specular_mask()` detects specular patches at feature-map resolution
(HSV V>240 & S<30, >25% receptive-field coverage). These patches are zeroed in
the excess map *before* Gaussian smoothing. Result: specular reflections cannot
contaminate the density estimate, lowering the Normal baseline and widening the
gap to true anomalies.

**Expert validation:** Independent domain expert confirmed the diagnosis and
recommended specular suppression as the highest-priority immediate fix.
See [expert roadmap](#expert-validated-improvement-roadmap) below.

✅ v7 specular suppression implemented in `compute_anomaly_map()`.  No bank
rebuild required — operates on the runtime excess map only.

### V10 — v8 Hair Suppression + Run Grouping + YOLO Visualizations

**Problem (2026-02-25):** Full v7 pipeline run revealed three issues:

1. **Hair artifacts score as highly anomalous.** Hair strands are thin dark
   lines on bright mucosal surfaces, rarely present in Normal bank images.
   PatchCore has never seen them → scores them as anomalous → bboxes anchor
   on hair instead of tumors.
   
2. **`--step all` creates 2-3 separate timestamped folders** making it hard
   to identify which generate run corresponds to which YOLO training.
   
3. **YOLO training lacks prediction visualizations** — hard to assess what
   the detector actually learned without seeing overlay images.

**v7→v8 detection rate comparison:** TBD (v8 not yet run). v7 produced
identical rates to v6 (Malignant 83%, Benign 80%, NP 82%, Normal FP 53.1%),
confirming that specular suppression changed bbox *positioning* but not
detection *rates*.

**Fixes implemented:**

1. **Hair artifact suppression** — Morphological blackhat transform on
   grayscale detects thin dark structures.  Hair-dominated patches (>15%
   coverage at feature-map resolution) are zeroed in the excess map before
   smoothing, same mechanism as specular suppression.
   Parameters: `HAIR_BLACKHAT_KSIZE=15`, `HAIR_INTENSITY_THRESH=30`,
   `HAIR_COVERAGE=0.15`.

2. **Session-based run grouping** — `--step all` now creates a shared
   session directory (`results/runs/<timestamp>_session/`) containing
   subdirectories `generate/` and `yolo/` with a `session_info.json`.
   Individual step runs still get their own timestamped directories.

3. **YOLO prediction visualizations** — After test evaluation, runs
   inference on test images and saves side-by-side plots (Original |
   Ground Truth | YOLO Prediction) as individual images and grid summaries
   to `<run_dir>/test_predictions/`.

### V11 — v9 Tier 2: Layer 4 Features (2026-02-28)

**Problem (2026-02-27):** Comparative study across all YOLO runs showed v8
(retuned hair suppression) mAP50 of 0.075 — significantly below v6 baseline
of 0.181.  Hair suppression reduced Normal FPs (53.1%→46.9%) but cost
~5-10% detection on abnormal classes, especially NP (82%→72%).  The root
cause: ImageNet ResNet50 layer2+layer3 features lack the morphological depth
needed to distinguish NP polyps from normal mucosa at a structural level.

**Tier 2 upgrade implemented:**

1. **Layer 4 feature extraction** — `PatchCoreExtractor` now extracts
   ResNet50 layer4 features (2048-dim, 8×8 spatial) in addition to layer2
   (512-dim, 32×32) and layer3 (1024-dim, 16×16).  Layer4 is upsampled to
   32×32 and concatenated, giving 3584-dim features per patch.
   Layer4 encodes tissue organisation, gland architecture, and structural
   patterns that emerge only at the deepest ResNet levels.

2. **Bank size reduction** — `MAX_BANK_PATCHES` reduced from 500K to 300K
   to offset the 2.3× increase in feature dimensionality.  Net RAM:
   300K × 3584 × 4B ≈ 4.1 GB vs old 500K × 1536 × 4B ≈ 2.9 GB.

3. **Bank dimension auto-check** — On cache load, the extractor’s output
   dimension is probed.  If the cached bank has a different feature dim
   (e.g. 1536 from v8), the bank is automatically rebuilt.  This prevents
   silent dimension mismatch errors when switching between versions.

**Key parameters unchanged:** All bbox extraction, smoothing, specular, and
hair suppression parameters remain identical to v8-retuned.  The only change
is the feature backbone depth.

### V12 — v10 Tier 3: DINO Self-Supervised Backbone (2026-02-28)

**Problem (2026-02-28):** v9 Tier 2 crashed on the 16 GB local machine during
calibration — the 3584-dim bank requires ~12 GB peak RAM for k-NN operations.
Furthermore, even with Layer 4 features, the fundamental domain gap remains:
ImageNet ResNet50 has never seen endoscopic tissue and groups pathological
tissue (NP, Benign) close to Normal in feature space.

**Tier 3 upgrade implemented:**

1. **DINO self-supervised fine-tuning** (`src/finetune_backbone.py`)
   - Fine-tunes ResNet50 on 5,957+ Normal endoscopic images using DINO
     (Self-Distillation with No Labels, Caron et al., ICCV 2021)
   - Teacher-Student architecture with EMA updates
   - Projection head: 2048 → 2048 → 256 → 65536 (discarded after training)
   - Only the backbone weights are saved for PatchCore use
   - 100 epochs, cosine LR schedule, momentum 0.996→1.0

2. **Custom backbone support in PatchCoreExtractor**
   - `PatchCoreExtractor(backbone_path=None)` — new optional argument
   - When provided, loads DINO-finetuned weights instead of ImageNet defaults
   - Handles fc layer shape mismatch (DINO uses Identity, ImageNet uses Linear)
   - Feature dimensions remain 3584 (same architecture, different weights)

3. **Separate bank cache** — DINO features stored as `patchcore_bank_dino.npz`
   to avoid overwriting the ImageNet bank (needed for v9 comparison)

4. **Cloud orchestration** (`run_cloud.py`)
   - Runs full comparative study: DINO fine-tuning → v9 → v10 → comparison
   - Session-grouped outputs for each version
   - Summary JSON with timing and results

**Expected impact:**
- **Tighter Normal cluster:** DINO learns domain-specific invariances
  (lighting, specular, camera angle) → Normal patches become MORE similar
- **Wider anomaly gap:** Pathological tissue is pushed further from Normal
- **Better NP detection:** Structural differences invisible to ImageNet
  features become visible after domain adaptation
- **Reduced Normal FPs:** Backbone invariance to specular reflections
  reduces false anomaly signal without explicit suppression

**CLI usage:**
```bash
# Fine-tune backbone:
python src/main.py --step finetune --finetune-epochs 100

# Run full pipeline with DINO backbone:
python src/main.py --step all --backbone dino

# Or use cloud orchestration:
python run_cloud.py --all
```

---

## Expert-Validated Improvement Roadmap

Based on diagnostic results (2026-02-24), independently validated by domain expert.
Ordered by implementation priority.

### Tier 1 — Immediate Pipeline Fixes (v7-v8, implemented)

1. **Specular suppression before smoothing** ✅ Done (v7)
   - Zero specular patches in excess map before Gaussian smoothing
   - Drops Normal baseline by removing spec-inflated above-ceiling patches
   - Result: detection rates unchanged but bbox positioning improved

2. **Hair artifact suppression before smoothing** ✅ Done (v8)
   - Morphological blackhat detects thin dark structures (hair, fibres)
   - Hair-dominated patches zeroed alongside speculars
   - Expected: fewer false bboxes on hair artifacts

3. **Threshold floor tuning** — Monitor after v8 full pipeline evaluation
   - Current `ANOMALY_FLOOR_THRESH=0.04` may need slight increase if v8
     alone doesn't sufficiently separate Normal from weak NP

### Tier 2 — Feature Extraction Improvements (v9, implemented)

4. **Layer 4 features** ✅ Implemented (v9)
   - ResNet layer4 (2048 dims) added, total 3584-dim features per patch
   - Bank reduced to 300K patches to offset RAM increase
   - Bank auto-rebuild on dimension mismatch
   - Requires bank rebuild (~30 min with deeper features)
   - **OOM on 16GB local** — needs ≥32GB RAM (cloud server)

5. **Multi-resolution ensembling** — Separate banks per feature scale,
   require anomaly to register across multiple scales
   - Reduces single-scale artifact false positives
   - More complex implementation

### Tier 3 — Backbone/Architecture Changes (v10, implemented)

6. **Domain adaptation via DINO** ✅ Implemented (v10)
   - `src/finetune_backbone.py`: DINO self-supervised learning on 5,957 Normal images
   - `src/generate_bboxes.py`: `PatchCoreExtractor(backbone_path=...)` loads DINO weights
   - `run_cloud.py`: Full comparative study orchestration
   - Expert's "real solution" — bypasses the domain gap entirely
   - Requires GPU compute for fine-tuning, ≥32GB RAM for bank operations

7. **Two-stage "Detect & Classify" architecture** — Expert-endorsed approach
   - Optimize PatchCore entirely for **100% recall** (accept all FPs)
   - Use PatchCore as a recall-optimized Region Proposal Network
   - Pass cropped bboxes to Stage 2 WSOL Classifier (EfficientNet)
   - Stage 2 easily filters Normal FPs during inference because it's
     trained to recognize specific pathology classes
   - **This is the planned architecture** — see [Future Work](#future-work-two-stage-detect--classify-architecture) below

### Expert's Key Insight

> "ResNet50 (trained on ImageNet) groups smooth, edematous pathological tissue
> (like polyps) very closely to healthy normal mucosa in its feature space,
> while specular reflections on normal tissue artificially push normal scores
> higher."

This explains why v5 density smoothing alone was insufficient — it dilutes
*both* specular and tissue signals proportionally, rather than selectively
removing the specular contamination that inflates the Normal baseline.

---

## Known Limitations

| # | Limitation | Impact | Mitigation |
|---|---|---|---|
| L1 | ImageNet → endoscopic domain gap | ResNet50 features may not perfectly separate nasal tissue subtypes (especially NP/polyps which are structurally similar to normal mucosa) | **Short-term:** v7 specular suppression widens the gap by lowering Normal baselines; **Mid-term:** add Layer 4 features for deeper structural semantics; **Long-term:** domain adaptation via SSL (SimCLR/DINO) on the 5K+ normal dataset |
| L2 | No ground-truth bboxes for quantitative IoU evaluation | Cannot compute mAP for annotation quality | Visual inspection + downstream YOLO mAP on held-out data serves as proxy |
| L3 | Up to 3 bboxes per image (v6) | Multi-focal lesions may get merged; images with no anomaly get empty labels | Correct behaviour — bbox merging is intentional; empty labels are valid YOLO negatives |
| L4 | Adaptive threshold floor (4%) | If floor is too low, residual noise may trigger on some Normal images | v6 validation: 2/3 Normal images produce FPs (peaks 0.051–0.071). Acceptable because Normal is excluded from YOLO training; Stage 2 classifier resolves this at inference |
| L5 | Coreset sampling variance | Different random seed → slightly different bank → slightly different ceiling | Fixed seed=42 ensures full reproducibility |
| L6 | CPU-only inference | ~18 s/image on CPU for k-NN scoring | GPU acceleration transparent — code auto-detects CUDA |
| L7 | PatchCore cannot differentiate anomaly types | All deviations from Normal are flagged identically — cannot distinguish malignant from benign from polyp | Current: class label comes from source directory. Future: transition to WSOL (classification CNN + Grad-CAM) for class-specific localization |

---

## Data Inventory

| Directory | Role | Count | Notes |
|---|---|---|---|
| `data/sample_data/train/Normal/` | Memory bank (primary) | 48 images | Original labelled dataset |
| `data/normal_endoscopic/` | Memory bank (extended) | 5,957 images | Clinical 2021–2024; confirmed Normal |
| `data/sample_data/train/Malignant/` | Annotation target | 96 images | YOLO class 0 |
| `data/sample_data/train/Benign/` | Annotation target | 96 images | YOLO class 1 |
| `data/sample_data/train/NP/` | Annotation target | 96 images | YOLO class 2 |
| ~~`data/normal_endoscopic/dataKho/`~~ | Deleted | ~~200 images~~ | 2017–2018 data; redundant, removed |
| ~~`data/abnormal_endoscopic/`~~ | Deleted | ~~750+ images~~ | No class labels; leakage risk, removed |
| `models/patchcore_bank.npz` | Bank cache | — | float16 compressed; contains bank + ceiling |

---

## Runtime Reference

### First run (no cache)
```
[1/3] No cache found — building PatchCore Normal memory bank…
      ~20 min (6,005 images × 5 img/s)
      → saves models/patchcore_bank.npz
      Calibrates ceiling on 48 Normal train images (all 1024 patches)  ~1 min
[2/3] Fitting k-NN…                ~30 sec
[3/5–5/5] Generating bboxes…       ~5 min (incl. Normal negative control)
```

### Subsequent runs (cache hit)
```
[1/3] Loading cached memory bank…  ~10 sec
      Recalibrating ceiling…       ~1 min
[2/3] Fitting k-NN…                ~30 sec
[3/5–5/5] Generating bboxes…       ~5 min
```

### Rebuild bank (after adding new Normal data)
```
python src/main.py --step generate --rebuild-bank
```

---

## Future Work: Two-Stage "Detect & Classify" Architecture

### Motivation

PatchCore (Method 3) is fundamentally a **1-class unsupervised anomaly detector**. It answers
"is this patch not Normal?" — it cannot distinguish Malignant from Benign from NP. The class
label currently comes from the source directory, not from the model. This creates two
structural problems:

1. **Smooth anomaly blindspot:** NP/polyps are edematous normal tissue — structural anomalies,
   not textural ones. A frozen ImageNet ResNet50 focuses on texture/gradients (good for bloody
   malignant tumors) but struggles with smooth mucosal deviations (peak ~0.07 vs threshold 0.04).

2. **No class-discriminative localization:** All deviations from Normal get the same heatmap.
   If expansions add Mucus detection, PatchCore cannot differentiate mucus from tumor.

### Architecture: Decouple Localization from Classification

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: PatchCore "Finder" (Unsupervised Region Proposal)  │
│  Goal: HIGH RECALL — find anything that deviates from Normal │
│  Input:  Full endoscope image                                │
│  Output: Bounding box coordinates (class-agnostic)           │
│  Model:  Frozen ResNet50 + Normal memory bank (current code) │
└──────────────────────────────┬───────────────────────────────┘
                               │  crop + 15% padding
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: EfficientNet "Expert" (Supervised Classifier)      │
│  Goal: HIGH PRECISION — separate Benign / Malignant / NP     │
│  Input:  Cropped anomaly region (224×224), with margin        │
│  Output: Multi-label probabilities per class                 │
│  Model:  EfficientNet-B0, fine-tuned on cropped regions      │
│  Loss:   BCE per class (multi-label, not mutually exclusive) │
└──────────────────────────────────────────────────────────────┘
```

**Why this works:**
- PatchCore evaluates patch-by-patch against a feature bank → cannot use background shortcuts
- Cropping strips away nasal cavity structure, background artifacts → classifier is forced to
  learn actual textural/morphological differences between pathology types
- Multi-label setup handles overlapping conditions (e.g., tumor + overlying mucus)

### Critical Bottlenecks

| # | Bottleneck | Requirement | Current Status |
|---|---|---|---|
| B1 | Stage 1 recall must be ~100% | If PatchCore misses a polyp, Stage 2 never sees it | ✅ v6 validation: 9/9 abnormal images detected (100% recall), including NP. H01_NP_001 now caught (peak=0.071) |
| B2 | Context padding on crops | Pathologists need surrounding tissue margin to judge invasiveness | Implement 10-15% padding beyond bbox edges |
| B3 | Sufficient crop training data | Need enough crops per class for supervised training | 96 train images × 3 classes = 288 crops (tight but viable with heavy augmentation) |
| B4 | Multi-label vs multi-class | Anomaly region may contain overlapping conditions | Use BCE loss, not softmax |

### Implementation Plan

**Phase A — Improve Stage 1 Recall (current priority):**
- v6 adaptive threshold + bbox merging (implemented)
- v7 specular suppression (implemented 2026-02-24)
- Validate v7 on full dataset: target ≥90% detection rate on all abnormal classes
- **Expert-endorsed philosophy:** Optimize PatchCore entirely for recall, accept Normal FPs.
  Let PatchCore act as a recall-optimized Region Proposal Network (RPN).
  Stage 2 classifier handles precision.

**Phase B — Build Stage 2 Classifier:**
1. Generate crops: for each training image with PatchCore bbox, crop with 15% padding
2. Train EfficientNet-B0 (pretrained ImageNet):
   - Input: 224×224 crops
   - Output: 3 sigmoid heads (Malignant, Benign, NP)
   - Loss: `BCEWithLogitsLoss` per class
   - Augmentation: heavy (HSV jitter, flip, rotation, scale, random erasing)
   - Epochs: 50-100, early stopping on val loss
3. Evaluation: per-class AUC, sensitivity, specificity on test crops

**Phase C — End-to-End Pipeline:**
```
Full image → PatchCore bbox → crop + 15% pad → EfficientNet → class label + confidence
```
XAI: Grad-CAM on EfficientNet reveals *what features* drive the class decision within the crop.

### Data Inventory for Stage 2

| Split | Malignant | Benign | NP | Total |
|---|---|---|---|---|
| Train | 96 | 96 | 96 | 288 |
| Test | 32 | 32 | 32 | 96 |

### Alternatives Considered

**Layer4 features:** Adding ResNet layer4 (2048 dims → 3584 total) would marginally improve
structural anomaly detection for NP but: (a) exceeds 5 GB RAM constraint without reducing
bank quality, (b) increases k-NN query time 2.3×, (c) does not solve the fundamental
class-differentiation problem. **Verdict:** not worth the cost.

**WSOL (Grad-CAM on classifier):** Using a classification CNN + Grad-CAM for localization
instead of PatchCore. Better for class-specific heatmaps but lower localization precision
than PatchCore's patch-level scoring. The two-stage approach gets the best of both:
PatchCore's localization + classifier's differentiation.

**Domain-adapted features (SimCLR/DINO):** Self-supervised pre-training on the 5,957-image
Normal dataset before building the k-NN bank. Would improve the domain relevance of features
but requires significant compute and implementation effort. **Verdict:** worthwhile as a
long-term improvement after the two-stage architecture is validated.

---

## Options Not Yet Done — Expert Suggestion Registry

Comprehensive record of all improvement paths identified during expert reviews. Organised by
category to provide clear forward options when hitting roadblocks.

### A. PatchCore / Stage 1 Improvements

| ID | Option | Expected Benefit | Effort | Status | When to Pursue |
|---|---|---|---|---|---|
| A1 | **Domain-adapted backbone (SimCLR/DINO)** — Self-supervised pre-training on 5,957 Normal images before k-NN bank | Better feature separation for endoscopic tissue; may reduce Normal FP rate | High (GPU training, new code) | **✅ Implemented (v10)** | DINO fine-tuning + PatchCore integration complete, awaiting cloud evaluation |
| A2 | **Layer4 features** — Add ResNet layer4 (2048 dims → 3584 total) | Deeper structural/morphological semantics for NP differentiation | Medium (bank rebuild + RAM) | **✅ Implemented (v9)** | Complete, crashed on 16GB local — needs cloud server |
| A3 | **Adaptive floor threshold tuning** — Per-class or learned from validation set | Reduce Normal FPs while maintaining lesion recall | Low | Not started | If v7 specular suppression + current floor=0.04 still causes too many FPs |
| A4 | **GPU-accelerated k-NN (FAISS)** — Replace scikit-learn brute-force with FAISS GPU | ~10× scoring speedup (18s → ~2s per image) | Medium | Not started | If CPU scoring time becomes a bottleneck for iteration |
| A5 | **v7 Specular suppression** — Zero specular patches before Gaussian smoothing | Drops Normal baseline, widens gap to true anomalies by ~4 raw score points | Low (~20 lines) | **✅ Implemented** | Done (2026-02-24) |
| A6 | **Multi-resolution ensembling** — Separate banks per feature scale, require consensus | Reduces false positives from single-scale artifacts | Medium-High | Not started | After v7 full pipeline evaluation |

### B. Two-Stage Architecture (Post-PatchCore)

| ID | Option | Expected Benefit | Effort | Status | When to Pursue |
|---|---|---|---|---|---|
| B1 | **Stage 2 EfficientNet-B0 classifier** — Supervised classifier on PatchCore crops | Class-discriminative detection (Malignant vs Benign vs NP) | Medium-High | Documented, not started | After YOLO baseline established; this is the primary next architecture step |
| B2 | **Multi-label BCE loss** — Sigmoid per class, not softmax | Handle overlapping conditions (tumor + mucus) | Low (part of B1) | Documented | Part of B1 implementation |
| B3 | **Crop context padding (15%)** — Pathologist-informed margin around bboxes | Better classification accuracy; mimics diagnostic viewing | Low | Documented | Part of B1 implementation |

### C. YOLO Training Improvements

| ID | Option | Expected Benefit | Effort | Status | When to Pursue |
|---|---|---|---|---|---|
| C1 | **Noise-aware YOLO training** — Weight loss by PatchCore confidence score from `confidence_*.csv` | Down-weight noisy/uncertain annotations during training | Medium | Not started (Phase E stretch) | If YOLO mAP plateaus due to annotation noise |
| C2 | **Class-balanced sampling** — Oversample minority class + weighted loss | Prevent bias toward majority class | Low | Planned in train_yolo.py | During YOLO training tuning |
| C3 | **Medical-specific augmentation tuning** — Adjust HSV jitter, mosaic, rotation ranges | More realistic training distribution | Low | Configured; first YOLO run completed (23 epochs), results not yet fully analysed | During YOLO training tuning |

### D. Evaluation & XAI

| ID | Option | Expected Benefit | Effort | Status | When to Pursue |
|---|---|---|---|---|---|
| D1 | **Grad-CAM on trained YOLO** — Verify model attention aligns with PatchCore heatmap | Detect shortcut learning (if YOLO attends to borders/artefacts instead of tissue) | Medium | Not started (G12, Phase D) | After YOLO training completes |
| D2 | **PatchCore vs YOLO side-by-side** — Compare heatmaps for same images | Scientific validation that YOLO learned from tissue, not artefacts | Low (after D1) | Not started | After D1 |
| D3 | **Inference script** — `src/inference.py` for running predictions on new images | Usable end-to-end system | Medium | Not started (G13) | After YOLO model exists |
| D4 | **Gradio/Streamlit dashboard** — Interactive clinical demo | Clinical trust; upload image → detection + heatmap overlay | Medium | Not started (G14, Phase E) | Stretch goal, after core pipeline works |

### E. Data & Scope Expansion

| ID | Option | Expected Benefit | Effort | Status | When to Pursue |
|---|---|---|---|---|---|
| E1 | **Add Mucus class** — Expand YOLO to detect mucus alongside lesions | More clinically complete detection | High (needs labelled Mucus data + PatchCore retune) | Not started | Requires class-discriminative Stage 2 (B1) to differentiate from lesions |
| E2 | **Additional lesion subtypes** — e.g., specific tumor types as subclasses | Finer-grained diagnosis | High (needs data + hierarchical labels) | Not started | After baseline pipeline validated |
| E3 | **Grounding DINO + MedSAM** — Zero-shot foundation model alternative | Potentially higher bbox precision, no dataset-specific setup | High (external model dependency) | Documented, not pursued | If PatchCore recall proves insufficient on full dataset |
| E4 | **Multi-modal late fusion** — Combine image features with clinical metadata | Enhanced diagnostic context | High (needs metadata, architecture redesign) | Not started (G15, R4 stretch) | Long-term research goal |
| E5 | **Public dataset augmentation (Kvasir)** — Add 8,000 pre-labelled endoscopic images | More training data for YOLO | Medium (download, format conversion) | Planned in IMPLEMENTATION_PLAN.md but not executed | If 288 images prove insufficient for YOLO generalisation |

### F. Infrastructure

| ID | Option | Expected Benefit | Effort | Status | When to Pursue |
|---|---|---|---|---|---|
| F1 | **GPU acceleration** — Enable CUDA for PyTorch and k-NN | 10× faster PatchCore scoring, faster YOLO training | Low-Medium (hardware dependent) | CPU-only currently | If iteration speed becomes a bottleneck |
| F2 | **Experiment tracking (W&B/MLflow)** — Log runs, hyperparams, metrics | Reproducibility, comparison across experiments | Low | Not started | During YOLO training iteration |

---

## Phase 4: YOLOv11 Training

After `generate_dataset()` completes:

```
data/yolo_dataset/
├── images/train/   ← copies of all Malignant/Benign/NP images
└── labels/train/   ← YOLO format .txt files (one per image)
```

```bash
python src/train_yolo.py --epochs 75 --patience 10
# Or via main.py:
python src/main.py --step yolo
```

Defaults: `yolo11n.pt` (nano), `imgsz=640`, `batch=16`, `patience=10`, `epochs=75`.
Output: `results/yolo/train/weights/best.pt`

**Evaluation metrics:** mAP@0.5, mAP@0.5:0.95 per class.
**XAI layer 1:** PatchCore anomaly heatmap — explains *why* the region was flagged.
**XAI layer 2:** Grad-CAM on trained YOLOv11 — verifies model attention aligns with heatmap.
