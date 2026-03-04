"""
Diagnostic: overlay PatchCore anomaly heatmap on actual images to see
WHERE the algorithm is detecting vs where tumors actually are.

Creates 5-panel visualizations per image:
  1. Original image
  2. Raw anomaly map (before edge suppression & smoothing)
  3. Final anomaly map (after edge suppression & density smoothing)
  4. Anomaly overlay on original (hot regions)
  5. BBox result

Processes 5 images per class to give a comprehensive picture.
"""
import os, sys, time
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from generate_bboxes import (
    PatchCoreExtractor, extract_bboxes,
    calibrate_normal_score,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, K_NEIGHBORS, BANK_CACHE_PATH,
    ANOMALY_FLOOR_THRESH, ANOMALY_MARGIN_FRAC, EDGE_SUPPRESS_PX,
    FEATURE_MAP_RES, SCORE_SMOOTH_SIGMA, SCORE_SMOOTH_KERNEL
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'sample_data', 'train')
TEST_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'sample_data', 'test')
OUT_DIR      = os.path.join(PROJECT_ROOT, 'results', 'v4_diagnosis')
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLE = 5


def compute_raw_and_final_amap(extractor, img_path, nbrs, score_ceiling,
                                transform, device, orig_w, orig_h):
    """Return both RAW and FINAL anomaly maps + the per-patch score grid."""
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(tensor)
    _, C, fH, fW = feat.shape
    patches = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()

    dists, _ = nbrs.kneighbors(patches)
    scores   = dists.mean(axis=1).reshape(fH, fW).astype(np.float32)

    # RAW: subtractive only, no suppression
    excess = np.maximum(scores - score_ceiling, 0.0)
    margin = score_ceiling * ANOMALY_MARGIN_FRAC
    normed_raw = np.clip(excess / (margin + 1e-8), 0.0, 1.0)

    # FINAL: with edge suppression, density smoothing, specular suppression
    excess_final = excess.copy()
    if EDGE_SUPPRESS_PX > 0:
        n = EDGE_SUPPRESS_PX
        excess_final[:n, :]  = 0.0
        excess_final[-n:, :] = 0.0
        excess_final[:, :n]  = 0.0
        excess_final[:, -n:] = 0.0

    # Density smoothing at feature level
    excess_smooth = cv2.GaussianBlur(
        excess_final.astype(np.float32),
        (SCORE_SMOOTH_KERNEL, SCORE_SMOOTH_KERNEL),
        SCORE_SMOOTH_SIGMA
    )
    normed_final = np.clip(excess_smooth / (margin + 1e-8), 0.0, 1.0)

    amap_raw   = cv2.resize((normed_raw * 255).astype(np.uint8),
                            (orig_w, orig_h))
    amap_final = cv2.resize((normed_final * 255).astype(np.uint8),
                            (orig_w, orig_h))

    return amap_raw, amap_final, scores


def main():
    device = torch.device('cpu')
    extractor = PatchCoreExtractor().to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Loading bank...")
    cached = np.load(BANK_CACHE_PATH)
    bank = cached['bank'].astype(np.float32)
    print(f"  Bank: {bank.shape[0]:,} patches")

    print("Calibrating ceiling...")
    score_ceiling = calibrate_normal_score(bank, K_NEIGHBORS, extractor,
                                           transform, device)
    margin = score_ceiling * ANOMALY_MARGIN_FRAC
    print(f"  Ceiling: {score_ceiling:.4f}  Margin: {margin:.4f}")

    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='brute',
                            metric='euclidean', n_jobs=-1)
    nbrs.fit(bank)

    test_cases = [
        ('Normal',    os.path.join(TEST_ROOT, 'Normal')),
        ('Malignant', os.path.join(DATA_ROOT, 'Malignant')),
        ('Benign',    os.path.join(DATA_ROOT, 'Benign')),
        ('NP',        os.path.join(DATA_ROOT, 'NP')),
    ]

    for cls_name, cls_dir in test_cases:
        if not os.path.isdir(cls_dir):
            print(f"  {cls_name}: not found, skip")
            continue
        files = sorted(f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        # Sample: first, 25%, mid, 75%, last
        indices = [0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]
        indices = sorted(set(min(i, len(files)-1) for i in indices))
        sample = [files[i] for i in indices][:N_SAMPLE]

        print(f"\n=== {cls_name} ({len(sample)} images) ===")
        for fname in sample:
            img_path = os.path.join(cls_dir, fname)
            orig_bgr = cv2.imread(img_path)
            if orig_bgr is None:
                continue
            orig_h, orig_w = orig_bgr.shape[:2]
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            amap_raw, amap_final, patch_scores = \
                compute_raw_and_final_amap(
                    extractor, img_path, nbrs, score_ceiling,
                    transform, device, orig_w, orig_h)
            bboxes = extract_bboxes(amap_final, orig_w, orig_h)
            dt = time.time() - t0

            peak_raw = float(amap_raw.max()) / 255.0
            peak_final = float(amap_final.max()) / 255.0
            n_anomalous = int((patch_scores > score_ceiling).sum())
            total_patches = patch_scores.size

            print(f"  {fname[:45]:45s}  raw_peak={peak_raw:.3f}  "
                  f"final_peak={peak_final:.3f}  "
                  f"anomalous_patches={n_anomalous}/{total_patches}  "
                  f"bboxes={len(bboxes)}  ({dt:.1f}s)")

            # Print raw score statistics
            print(f"    Raw scores: min={patch_scores.min():.1f}  "
                  f"median={np.median(patch_scores):.1f}  "
                  f"p90={np.percentile(patch_scores,90):.1f}  "
                  f"max={patch_scores.max():.1f}  "
                  f"ceiling={score_ceiling:.1f}")

            # Create overlay: blend anomaly heatmap onto original
            amap_color = cv2.applyColorMap(amap_final, cv2.COLORMAP_JET)
            amap_color_rgb = cv2.cvtColor(amap_color, cv2.COLOR_BGR2RGB)
            # Only blend where anomaly > 0
            alpha = (amap_final.astype(np.float32) / 255.0)[:, :, np.newaxis]
            overlay = (orig_rgb * (1 - alpha * 0.6) +
                       amap_color_rgb * alpha * 0.6).astype(np.uint8)

            # Draw bboxes on overlay and on clean copy
            vis_bbox = orig_rgb.copy()
            for (xc, yc, bw, bh) in bboxes:
                x1 = int((xc - bw/2) * orig_w)
                y1 = int((yc - bh/2) * orig_h)
                x2 = int((xc + bw/2) * orig_w)
                y2 = int((yc + bh/2) * orig_h)
                cv2.rectangle(vis_bbox, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 5-panel figure
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(orig_rgb)
            axes[0].set_title('Original')
            axes[1].imshow(amap_raw, cmap='hot', vmin=0, vmax=255)
            axes[1].set_title(f'Raw anomaly\n(peak={peak_raw:.3f})')
            axes[2].imshow(amap_final, cmap='hot', vmin=0, vmax=255)
            axes[2].set_title(f'Final anomaly\n(peak={peak_final:.3f})')
            axes[3].imshow(overlay)
            axes[3].set_title(f'Overlay\n({n_anomalous} anom patches)')
            axes[4].imshow(vis_bbox)
            axes[4].set_title(f'BBoxes ({len(bboxes)})')
            for ax in axes: ax.axis('off')

            plt.suptitle(f"{cls_name}: {fname}\n"
                         f"Score range: {patch_scores.min():.1f}–"
                         f"{patch_scores.max():.1f}  "
                         f"Ceiling: {score_ceiling:.1f}",
                         fontsize=10)
            plt.tight_layout()
            safe_name = fname.replace(' ', '_')
            plt.savefig(os.path.join(OUT_DIR, f"{cls_name}_{safe_name}.png"),
                        dpi=100, bbox_inches='tight')
            plt.close()

    print(f"\nDiagnostic visualizations saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
