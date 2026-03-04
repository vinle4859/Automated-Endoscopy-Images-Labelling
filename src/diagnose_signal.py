"""
Signal Diagnostic: Where Does PatchCore Actually See Anomaly?
=============================================================
Generates 5-panel diagnostic plots per image showing:
  1. Original image
  2. RAW k-NN scores (before ceiling subtraction) — shows what ResNet sees
  3. Specular mask (HSV V>240 & S<30) — identifies reflections
  4. Excess after ceiling (what actually drives bbox generation)
  5. Specular-masked excess — what remains after removing specular signal

Also prints per-image statistics:
  - % of above-ceiling patches that are specular vs non-specular
  - Peak score at specular locations vs peak outside specular
  - Raw score percentiles (p50/p75/p95/max) to understand distribution

Usage:
    python src/diagnose_signal.py
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reuse config from generate_bboxes ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from generate_bboxes import (
    PatchCoreExtractor, build_memory_bank, calibrate_normal_score,
    IMG_SIZE, K_NEIGHBORS, IMAGENET_MEAN, IMAGENET_STD,
    SCORE_SMOOTH_SIGMA, SCORE_SMOOTH_KERNEL, EDGE_SUPPRESS_PX,
    ANOMALY_MARGIN_FRAC, ANOMALY_FLOOR_THRESH, FEATURE_MAP_RES,
    BANK_CACHE_PATH, DATA_ROOT,
    get_specular_mask,  # v7: shared specular mask function
)
from torchvision import transforms

try:
    from run_manager import create_run_dir
    _HAS_RUN_MANAGER = True
except ImportError:
    _HAS_RUN_MANAGER = False

# Fallback path (overridden in main() if run_manager is available)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'signal_diagnostic')

# Images to diagnose — pick a mix of classes and signal strengths
DIAG_IMAGES = [
    # Benign — the image the user flagged
    ("Benign", "H16_JNA_01_crop.png"),
    # Malignant — pick images with varying signal
    ("Malignant", "H01_Salivary gland type carcinoma_001.png"),
    ("Malignant", "H09_ONB_001_crop.png"),
    ("Malignant", "H01_SCC_011.png"),
    # Benign — varying
    ("Benign", "H01_IP_016.png"),
    ("Benign", "H17_hemangioma_012_crop.png"),
    # NP — typically subtle
    ("NP", "H01_NP_041.png"),
    ("NP", "H06_NP_072_crop.png"),
    # Normal — should be all dark
    ("Normal", "normal 039.PNG"),
    ("Normal", "normal 055.PNG"),
]


def get_specular_mask_at_feature_res(img_path, fH=32, fW=32):
    """
    Detect specular highlights at FEATURE MAP resolution.
    Delegates to the shared get_specular_mask() in generate_bboxes.py
    so diagnostic and pipeline use identical specular detection.
    """
    return get_specular_mask(img_path, fH, fW)


def diagnose_image(img_path, class_name, extractor, nbrs, score_ceiling,
                   transform, device):
    """
    Produce diagnostic data for a single image.
    Returns dict with all analysis results.
    """
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = extractor(tensor)
    _, C, fH, fW = feat.shape
    patches = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()

    dists, _ = nbrs.kneighbors(patches)
    raw_scores = dists.mean(axis=1).reshape(fH, fW).astype(np.float32)

    # Excess after ceiling subtraction (what drives v6 bboxes)
    excess = np.maximum(raw_scores - score_ceiling, 0.0)

    # Edge suppression
    if EDGE_SUPPRESS_PX > 0:
        n = EDGE_SUPPRESS_PX
        excess[:n, :] = 0.0
        excess[-n:, :] = 0.0
        excess[:, :n] = 0.0
        excess[:, -n:] = 0.0

    # Density smoothing
    excess_smooth = cv2.GaussianBlur(
        excess, (SCORE_SMOOTH_KERNEL, SCORE_SMOOTH_KERNEL), SCORE_SMOOTH_SIGMA
    )

    # Normalized (same as pipeline)
    margin = score_ceiling * ANOMALY_MARGIN_FRAC
    normed = np.clip(excess_smooth / (margin + 1e-8), 0.0, 1.0)

    # Specular mask at feature resolution
    spec_mask = get_specular_mask_at_feature_res(img_path, fH, fW)

    # Specular-masked excess
    excess_no_spec = excess.copy()
    excess_no_spec[spec_mask] = 0.0
    excess_no_spec_smooth = cv2.GaussianBlur(
        excess_no_spec, (SCORE_SMOOTH_KERNEL, SCORE_SMOOTH_KERNEL),
        SCORE_SMOOTH_SIGMA
    )
    normed_no_spec = np.clip(excess_no_spec_smooth / (margin + 1e-8), 0.0, 1.0)

    # ── Statistics ─────────────────────────────────────────────────────────
    above_ceiling = raw_scores > score_ceiling
    n_above = above_ceiling.sum()
    n_spec_above = (above_ceiling & spec_mask).sum()
    n_nonspec_above = (above_ceiling & ~spec_mask).sum()

    peak_raw = float(raw_scores.max())
    peak_at_spec = float(raw_scores[spec_mask].max()) if spec_mask.any() else 0.0
    peak_outside_spec = float(raw_scores[~spec_mask].max()) if (~spec_mask).any() else 0.0

    return {
        'raw_scores': raw_scores,
        'excess': excess,
        'excess_smooth': excess_smooth,
        'normed': normed,
        'spec_mask': spec_mask,
        'excess_no_spec': excess_no_spec,
        'normed_no_spec': normed_no_spec,
        'orig_img': np.array(img),
        'orig_w': orig_w, 'orig_h': orig_h,
        'peak_raw': peak_raw,
        'peak_at_spec': peak_at_spec,
        'peak_outside_spec': peak_outside_spec,
        'n_above_ceiling': int(n_above),
        'n_spec_above': int(n_spec_above),
        'n_nonspec_above': int(n_nonspec_above),
        'n_spec_total': int(spec_mask.sum()),
        'score_ceiling': score_ceiling,
        'p50': float(np.percentile(raw_scores, 50)),
        'p75': float(np.percentile(raw_scores, 75)),
        'p95': float(np.percentile(raw_scores, 95)),
    }


def plot_diagnostic(results, class_name, fname, output_path):
    """5-panel diagnostic plot."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'{class_name} -- {fname}', fontsize=14, fontweight='bold')

    # Panel 1: Original
    axes[0].imshow(results['orig_img'])
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Panel 2: Raw k-NN scores (before ceiling subtraction)
    raw_up = cv2.resize(results['raw_scores'],
                        (results['orig_w'], results['orig_h']))
    im2 = axes[1].imshow(raw_up, cmap='hot', vmin=0,
                         vmax=results['score_ceiling'] * 1.5)
    axes[1].set_title(f'Raw k-NN scores\n'
                      f'ceiling={results["score_ceiling"]:.1f}\n'
                      f'peak={results["peak_raw"]:.1f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Panel 3: Specular mask overlay
    overlay = results['orig_img'].copy()
    spec_up = cv2.resize(results['spec_mask'].astype(np.uint8),
                         (results['orig_w'], results['orig_h']),
                         interpolation=cv2.INTER_NEAREST)
    overlay[spec_up > 0] = [255, 0, 0]  # red = specular
    axes[2].imshow(overlay)
    axes[2].set_title(f'Specular mask (red)\n'
                      f'{results["n_spec_total"]} patches\n'
                      f'{results["n_spec_above"]}/{results["n_above_ceiling"]} '
                      f'above-ceiling are specular')
    axes[2].axis('off')

    # Panel 4: Current pipeline output (with speculars)
    amap = cv2.resize((results['normed'] * 255).astype(np.uint8),
                      (results['orig_w'], results['orig_h']))
    axes[3].imshow(amap, cmap='hot', vmin=0, vmax=255)
    axes[3].set_title(f'Pipeline output (v6)\npeak normed={results["normed"].max():.3f}')
    axes[3].axis('off')

    # Panel 5: After specular removal
    amap_ns = cv2.resize((results['normed_no_spec'] * 255).astype(np.uint8),
                         (results['orig_w'], results['orig_h']))
    axes[4].imshow(amap_ns, cmap='hot', vmin=0, vmax=255)
    axes[4].set_title(f'Specular-removed\npeak normed={results["normed_no_spec"].max():.3f}')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    global OUTPUT_DIR
    if _HAS_RUN_MANAGER:
        OUTPUT_DIR = create_run_dir('signal_diag', version='v6', params={
            'ANOMALY_MARGIN_FRAC': ANOMALY_MARGIN_FRAC,
            'ANOMALY_FLOOR_THRESH': ANOMALY_FLOOR_THRESH,
            'SCORE_SMOOTH_SIGMA': SCORE_SMOOTH_SIGMA,
            'SCORE_SMOOTH_KERNEL': SCORE_SMOOTH_KERNEL,
            'K_NEIGHBORS': K_NEIGHBORS,
        })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")

    # ── Load bank ─────────────────────────────────────────────────────────
    if os.path.exists(BANK_CACHE_PATH):
        print("Loading cached memory bank...")
        data = np.load(BANK_CACHE_PATH)
        bank = data['bank'].astype(np.float32)
    else:
        print("ERROR: No bank cache found. Run main pipeline first.")
        sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────────────
    extractor = PatchCoreExtractor().to(device)
    extractor.eval()
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Fitting k-NN...")
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='brute',
                            metric='euclidean', n_jobs=-1)
    nbrs.fit(bank)

    print("Calibrating ceiling...")
    # Use known ceiling from previous calibration run to save ~14 min
    # (48 Normal images × 1024 patches scored against 500K bank)
    ceiling = 37.7454
    print(f"  Using known ceiling = {ceiling:.4f} (from full pipeline run)")
    print(f"  (Skip recalibration to save time; delete this override to recalibrate)")

    # ── Process diagnostic images ─────────────────────────────────────────
    print(f"\nGenerating diagnostics for {len(DIAG_IMAGES)} images...")
    print(f"{'='*90}")

    summary_lines = []

    for i, (cls, fname) in enumerate(DIAG_IMAGES):
        if cls == "Normal":
            img_path = os.path.join(DATA_ROOT, cls, fname)
        else:
            img_path = os.path.join(DATA_ROOT, cls, fname)

        if not os.path.exists(img_path):
            # Try test split
            test_root = os.path.join(PROJECT_ROOT, 'data', 'sample_data', 'test')
            img_path = os.path.join(test_root, cls, fname)

        if not os.path.exists(img_path):
            print(f"  SKIP: {cls}/{fname} not found")
            continue

        results = diagnose_image(img_path, cls, extractor, nbrs, ceiling,
                                 tfm, device)

        # Print statistics
        spec_pct = (results['n_spec_above'] / max(results['n_above_ceiling'], 1)) * 100
        line = (f"  {cls:12s} | {fname:45s} | "
                f"raw_peak={results['peak_raw']:6.1f} | "
                f"ceiling={ceiling:5.1f} | "
                f"above_ceil={results['n_above_ceiling']:3d}/1024 | "
                f"spec_above={results['n_spec_above']:3d} ({spec_pct:4.1f}%) | "
                f"peak@spec={results['peak_at_spec']:6.1f} | "
                f"peak@tissue={results['peak_outside_spec']:6.1f}")
        print(line)
        summary_lines.append(line)

        # Save plot
        out_name = f"diag_{i:02d}_{cls}_{fname}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plot_diagnostic(results, cls, fname, out_path)

    print(f"\n{'='*90}")
    print(f"Diagnostics saved to: {OUTPUT_DIR}")

    # ── Key findings ──────────────────────────────────────────────────────
    print("\n== KEY DIAGNOSTIC QUESTION ==")
    print("If 'peak@tissue' (non-specular peak) is close to or below the ceiling,")
    print("then PatchCore fundamentally CANNOT detect tumors regardless of threshold tuning.")
    print("The signal simply isn't there in ResNet50 feature space.")
    print("Only specular suppression + feature backbone change (DINOv2) or")
    print("a fundamentally different approach would help.")


if __name__ == '__main__':
    main()
