# Automated Annotation Strategies for Endoscopic AI

**Project:** Trustworthy Endoscopic AI with YOLOv11
**Objective:** Generate high-quality bounding boxes for YOLO object detection without
manual medical expert annotation.
**Context:** The dataset contains only image-level classification labels (Normal, Benign,
Malignant, NP). Bounding boxes required for detection training do not exist.

This document records all annotation strategies attempted in chronological order, including
failed approaches with full documentation of their failure modes. This is intentional: the
failure of Methods 1 and 2 provides scientific justification for the design choices in Method 3.

---

## Method 1: Weakly Supervised Object Detection via Grad-CAM

**Status: Abandoned â€” shortcut learning**

### Concept
Train a classification CNN on the image-level labels (Normal / Benign / Malignant / NP), then
use Gradient-weighted Class Activation Mapping (Grad-CAM) to extract the spatial regions that
the classifier used to make its decision. These activation regions are converted to bounding
boxes under the Weakly Supervised Object Detection (WSOD) paradigm.

### Failure Mode
The classifier learned to distinguish classes using background artefacts rather than
pathological tissue features â€” a phenomenon known as **shortcut learning**. Endoscopic images
contain consistent imaging artefacts per acquisition session (vignetting patterns, instrument
shadows, specular highlights, light positioning). These artefacts are easier correlation targets
for a supervised classifier than the tissue itself, especially with a small dataset.

Grad-CAM activations fired predominantly on:
- Image borders and vignetting patterns
- Bright specular highlights from the endoscope light source
- Instrument shadows

Not on the pathological tissue.

### Why Documented
This failure motivated the shift to unsupervised methods (Methods 2 and 3) that make no use of
class labels during the localization step and cannot exploit spurious correlations between
image-level labels and background artefacts.

---

## Method 2: Reconstruction-Based Anomaly Detection (UNetAutoencoder)

**Status: Fully implemented and trained. Abandoned â€” over-reconstruction failure.**
**Detailed documentation:** `docs/METHOD_2_UNET_AUTOENCODER.md`

### Concept
Train a U-Net convolutional autoencoder exclusively on Normal images. The model learns to
reconstruct healthy tissue. When presented with an abnormal image, it should reconstruct
"what healthy tissue looks like there", producing high pixel-level error at the site of
the pathology. The pixel difference map becomes the anomaly localization signal.

### Implementation Summary
- **Architecture:** UNetAutoencoder with 4-level encoder, bottleneck (512ch), 4-level decoder,
  and skip connections at each level
- **Training:** 50 epochs on 48 Normal images, MSE loss, Adam optimizer, ReduceLROnPlateau
- **Outcome:** Loss converged successfully; model weights saved to `models/autoencoder_best.pth`

### Failure Mode
U-Net skip connections pass fine-grained spatial detail directly from each encoder stage to
the corresponding decoder, allowing the model to reconstruct **any** spatial content without
compressing it through the bottleneck. The bottleneck â€” intended to be the information filter
that prevents reconstruction of unseen patterns â€” is bypassed. Additionally, nasal tumour
tissue and normal nasal mucosa share the same pixel-level texture (colour, reflectance), so
even an MSE loss provides no differential signal between normal and pathological regions.

Result: The reconstruction error map was near-zero everywhere, including over tumours. The
method produced no useful localisation signal.

### Lesson
Pixel-space anomaly detection methods depend on a reconstruction bottleneck that is strict
enough to fail on unseen patterns. Skip connections are inherently incompatible with this
requirement. For texturally homogeneous tissue, pixel-space methods are insufficient regardless
of architecture. The signal must come from **semantic features**, not pixels.

---

## Method 3: PatchCore Feature-Space Anomaly Detection (ResNet50 + k-NN)

**Status: Active implementation**
**Detailed documentation:** `docs/METHOD_3_PATCHCORE_IMPLEMENTATION.md`
**Reference:** *Towards Total Recall in Industrial Anomaly Detection*, Roth et al., CVPR 2022

### Concept
Instead of comparing pixels, compare **semantic patch features** in the embedding space of a
frozen ImageNet backbone. A frozen ResNet50 encodes each image patch as a 1536-dimensional
vector (concatenated layer2 + layer3 outputs). All Normal image patches form a memory bank.
For each abnormal image, per-patch distances to the k nearest Normal bank neighbours form a
spatial anomaly score map. High distance â†’ the feature pattern is absent from Normal tissue â†’
anomaly. No training is required beyond the ImageNet-pretrained backbone.

### Why This Resolves the Previous Failures

| Failure | Method 2 cause | Method 3 fix |
|---|---|---|
| Over-reconstruction | Skip connections let any content through | No reconstruction at all; distance in feature space |
| Pixel texture homogeneity | MSE sees no difference between normal and tumour pixels | Semantic features encode tissue organisation, not pixel values |
| Shortcut learning (Method 1) | Classifier exploits spurious correlations | Artefacts present in Normal images are encoded in the bank; they score near-zero distance |

### Implementation
- **Backbone:** Frozen ResNet50 (ImageNet weights, `layer2` + `layer3` concatenated = 1536-dim)
- **Memory bank:** 500,000 patches (capped from ~600,500 raw) from 6,005 Normal images (100 patches/image, seed=42)
- **k-NN:** `NearestNeighbors(k=5, algorithm='brute')` â€” brute-force required for 1536-dim
- **Calibration (v2):** Score all 1024 patches of 48 Normal train images against the bank; 99th percentile → `score_ceiling` (~38–42)
- **Scoring (v2):** Subtractive: `excess = max(score − ceiling, 0)`, normalised as `clip(excess / ceiling, 0, 1)`. Normal patches → exactly 0.
- **Bbox extraction (v6, current):** Density-smoothed scoring (Gaussian σ=2.0, kernel=7 at 32×32) → adaptive Otsu (floor 0.04) → dilation 5×5 + close 5×5 → area filter (0.5 %–85%) → bbox merging (gap < 8%) → up to 3 bboxes → YOLO format. No fallback.
- **Cache:** `models/patchcore_bank.npz` (float16 compressed, ~880 MB); ceiling always recalibrated on load

### Validation
- ✅ Anomaly maps localise genuine lesion tissue (confirmed by human visual inspection of v1 output)
- ✅ No training required; pipeline is deterministic (fixed seeds)
- ✅ Anomaly heatmap provides first-layer XAI before YOLO training
- ✅ Memory bank enrichable with additional Normal data at any time (`--rebuild-bank`)
- ✅ v2 subtractive scoring eliminates false heat on Normal tissue
- ✅ Multi-bbox support (up to 3 per image, v6) with area-based noise filtering and bbox merging
- ✅ Normal negative control: 32 held-out Normal test images checked for false positives
- ⚠️ ImageNet → endoscopic domain gap is a residual uncertainty
- ⚠️ No ground-truth bboxes available for quantitative IoU validation
- ⚠️ Adaptive Otsu threshold (floor 0.04) produces 53% Normal FPs — acceptable because Normal excluded from YOLO training

---

## Zero-Shot Foundation Model Alternative (Not Implemented)

For completeness, a fourth candidate strategy was evaluated but not implemented:
**Grounding DINO + MedSAM** â€” text-prompted lesion detection followed by medical image
segmentation to refine the bbox.

- **Pros:** Leverages large-scale medical and vision-language pre-training; no dataset-specific
  setup required; potentially higher bbox precision if prompts are well-chosen
- **Cons:** Requires two large models (~1â€“2 GB each); depends on Grounding DINO correctly
  interpreting the prompt ("lesion", "polyp") in an unfamiliar endoscopic context; introduces
  external model dependency and reproducibility concerns
- **Reason not pursued:** Method 3 (PatchCore) is fully self-contained, uses only the
  project's own Normal data as reference, and provides a cleaner scientific narrative for
  a Trustworthy AI pipeline

---

## Method Comparison Summary

| | Method 1 (Grad-CAM) | Method 2 (UNetAutoencoder) | Method 3 (PatchCore) | Alternative (Foundation) |
|---|---|---|---|---|
| Status | Abandoned | Abandoned | **Active** | Not implemented |
| Training required | Yes (classifier) | Yes (autoencoder) | **No** | No |
| Signal space | Pixel gradients | Pixel reconstruction error | **Feature-space distance** | Vision-language embeddings |
| Failure mode | Shortcut learning | Over-reconstruction | Domain gap (residual) | Prompt sensitivity |
| XAI output | Grad-CAM map | Difference map (unreliable) | **Anomaly heatmap (calibrated)** | Segmentation mask |
| External dependency | None | None | **ImageNet weights (cached)** | Grounding DINO + MedSAM |

See individual method documents for full implementation details and failure analysis.

