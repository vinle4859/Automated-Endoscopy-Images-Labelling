# Medical Imaging AI Pipeline

**Project:** Trustworthy Endoscopic AI with YOLOv11  
**Status:** Initial Setup  
**Python Version:** 3.12.6

## Quick Start

1. **Activate Virtual Environment:**
   ```powershell
   venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify Installation:**
   ```powershell
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   ```

## Project Structure

```
Med_Img_Project/
├── data/              # All datasets (git-ignored, NDA protected)
├── models/            # Model weights and checkpoints (git-ignored)
├── src/               # Source code
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks for exploration
├── results/           # Training results and metrics (git-ignored)
├── docs/              # Documentation
└── IMPLEMENTATION_PLAN.md  # Detailed implementation guide
```

## Key Features

- **Semi-Automated Labeling:** AI-assisted annotation workflow
- **Hierarchical Classification:** Multi-level diagnostic taxonomy
- **Quality-First Approach:** Automated data validation before training
- **Built-in Explainability:** Grad-CAM and attention visualizations
- **Multi-Modal Fusion:** Support for combining image + clinical data

## Next Steps

Refer to [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed implementation guide.

## Data Privacy

All data in `data/`, `models/`, and `results/` directories are excluded from Git due to potential NDA restrictions.
