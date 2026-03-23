# Automated Endoscopy Image Labelling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A fully automated annotation pipeline that generates bounding-box labels for endoscopic images **without any manual annotation**. The system uses [PatchCore](https://arxiv.org/abs/2106.08265) anomaly detection with a ResNet50 backbone to localise pathological regions, then trains a [YOLOv11](https://docs.ultralytics.com/) object detector on the auto-generated labels.

> **Status:** v14 draft mode is now implemented as a governance upgrade over the v13 extraction stack. It adds gate-v2 constraints (Normal FP and bbox-to-signal control) so high-coverage but unsafe pseudo-label sets no longer pass by default.

---

## Key Features

- **Zero Manual Annotation** — PatchCore anomaly detection generates YOLO bounding boxes from image-level class labels alone
- **Domain-Adapted Backbone** — Optional DINO self-supervised fine-tuning adapts ResNet50 features to endoscopic tissue (v10)
- **Multi-Tier Architecture** — 13 iterative versions across 5 tiers: baseline → artifact suppression → deeper features → backbone adaptation → calibration governance
- **Built-in Explainability** — Anomaly heatmap overlays and diagnostic visualisations show why each region was flagged
- **Reproducible** — Fixed seeds, cached memory banks, timestamped run directories

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: PatchCore Anomaly Detection                           │
│                                                                 │
│  Normal Images ──► ResNet50 ──► Feature Bank (300K patches)     │
│                    (L2+L3+L4)   k-NN anomaly scoring            │
│                                                                 │
│  Test Image ──► Feature Extraction ──► Excess Score Map         │
│             ──► Gaussian Smoothing ──► Adaptive Threshold       │
│             ──► Bbox Extraction ──► YOLO-format labels          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: YOLOv11-nano Object Detection                         │
│                                                                 │
│  Auto-generated labels ──► YOLOv11 training ──► Detector        │
└─────────────────────────────────────────────────────────────────┘
```

**Backbone options:**
- **ImageNet** (default) — Frozen ResNet50 with ImageNet weights
- **DINO** (v10) — Self-supervised fine-tuned ResNet50 on 5,957 Normal endoscopic images

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with ≥8 GB VRAM for DINO fine-tuning
- ≥32 GB RAM recommended for full pipeline (16 GB sufficient for v6–v8)

### Installation

```bash
git clone https://github.com/vinle4859/Automated-Endoscopy-Images-Labelling.git
cd Automated-Endoscopy-Images-Labelling

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# CPU
pip install -r requirements.txt

# GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Usage

```bash
# Run the full pipeline (generate bboxes + train YOLO)
python src/main.py --step all

# Run v14 draft gate-v2 mode (v13 extraction + stricter governance)
python src/main.py --step generate --v14-draft

# Run with DINO backbone (v10 — requires GPU for fine-tuning)
python src/main.py --step finetune --finetune-epochs 100
python src/main.py --step all --backbone dino

# Cloud comparative study (v9 vs v10)
python run_cloud.py --all
```

## Project Structure

```
Automated-Endoscopy-Images-Labelling/
├── src/                        # Source code
│   ├── main.py                 #   CLI entry point
│   ├── generate_bboxes.py      #   PatchCore anomaly detection + bbox pipeline
│   ├── finetune_backbone.py    #   DINO self-supervised backbone training
│   ├── train_yolo.py           #   YOLOv11-nano training
│   ├── run_manager.py          #   Timestamped run directory management
│   ├── diagnose_signal.py      #   Signal diagnostic visualisations
│   ├── diagnose_v4.py          #   Heatmap overlay diagnostics
│   └── validate_bboxes.py      #   Automated bbox quality validation
├── run_cloud.py                # Cloud orchestration (v9 vs v10 comparison)
├── configs/
│   └── yolo_data.yaml          # YOLO dataset paths + class names
├── docs/                       # Technical documentation
├── data/                       # Datasets (git-ignored, see below)
├── models/                     # Weights and feature banks (git-ignored)
├── results/                    # Training outputs (git-ignored)
├── VERSION_REGISTRY.md         # Complete version history (v1–v13)
├── requirements.txt
└── LICENSE
```

## Version History

The pipeline evolved through 13 versions across 5 tiers. See [VERSION_REGISTRY.md](VERSION_REGISTRY.md) for the complete record with parameters, results, and failure analysis.

| Version | Tier | Key Change | YOLO mAP50 | Status |
|---------|------|------------|------------|--------|
| v1-v5 | 0 | Core PatchCore development | — | Superseded |
| v6 | 0 | Bbox merging + adaptive threshold | **0.181** | Baseline |
| v7-v8 | 1 | Specular + hair suppression | 0.075 | Evaluated |
| v9 | 2 | Layer 4 features (3584-dim) | OOM | Needs ≥32 GB RAM |
| v10 | 3 | DINO self-supervised backbone | Train failed | Local run showed near-zero lesion coverage |
| v11 | 3 | ROI-aware glare rejection | Blocked by gate | Edge artifacts removed, lesion coverage still low |
| v12 | 4 | Auto-calibration + backbone A/B | Blocked by gate | ImageNet selected; gate still fails (coverage/FP trade-off) |
| v13 | 4 | Soft-border + dual-threshold extraction | Gate passed, clinically risky | 88.6% train non-empty labels but 68.8% Normal FP |
| v14 | 5 | Gate-v2 draft over v13 extraction stack | Draft implemented | Enforce Normal-control and pseudo-label expansion constraints |
| v15 | 5 | Hybrid PatchCore + Med-SAM refinement | Planned | Improve localization while controlling false positives |

**Current finding:** v13 fixed under-coverage but over-expanded detections into Normal space. The pipeline now has a gate-design failure mode: it can pass quality checks while still generating high Normal false positives. Next iterations must tighten gate definitions (include Normal FP constraints) before any trusted YOLO retraining.

## Documentation

| Document | Description |
|----------|-------------|
| [VERSION_REGISTRY.md](VERSION_REGISTRY.md) | Complete version history with parameters, results, and failure analysis |
| [METHOD_3_PATCHCORE_IMPLEMENTATION.md](docs/METHOD_3_PATCHCORE_IMPLEMENTATION.md) | Core PatchCore algorithm documentation |
| [PROJECT_REVIEW_AND_ROADMAP.md](docs/PROJECT_REVIEW_AND_ROADMAP.md) | Gap analysis and execution roadmap |
| [AUTOMATED_ANNOTATION_STRATEGIES.md](docs/AUTOMATED_ANNOTATION_STRATEGIES.md) | Methodology evolution narrative |
| [METHOD_2_UNET_AUTOENCODER.md](docs/METHOD_2_UNET_AUTOENCODER.md) | Failed U-Net approach (negative result) |
| [BBOX_MERGE_VS_SPLIT.md](docs/BBOX_MERGE_VS_SPLIT.md) | Bbox merging design decisions |
| [FAILURE_MODE_STUDY_READINESS.md](docs/FAILURE_MODE_STUDY_READINESS.md) | Paper-readiness checklist for failure-mode study artifacts |
| [PUBLICATION_STRATEGY.md](docs/PUBLICATION_STRATEGY.md) | Venue-tier strategy and evidence quality targets |
| [NEXT_STEPS_PRIORITIZATION.md](docs/NEXT_STEPS_PRIORITIZATION.md) | Ranked execution plan for post-v13 work |

## Data Privacy

All data in `data/`, `models/`, and `results/` directories are excluded from version control. Medical images may be subject to NDA restrictions.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
