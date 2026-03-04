"""
Automated BBox Quality Validation — no expert required
=======================================================
Runs after the pipeline to assess whether detected bounding boxes are covering
genuine pathological tissue or imaging artefacts (specular highlights, borders, etc.)

Three self-validation strategies that do NOT require expert annotation:

1. COLOUR CONTRAST TEST
   If a bbox covers real pathology, the tissue inside should have different
   colour statistics from the tissue immediately outside (the surrounding ring).
   Specular reflections also have high colour contrast (bright white vs. tissue),
   but specular suppression removes them before this stage.

2. TEXTURE COMPLEXITY TEST
   Pathological tissue tends to have irregular texture (higher Laplacian variance).
   A bbox on featureless mucosa or a black border would score low.

3. CROSS-IMAGE CONSISTENCY TEST
   If PatchCore repeatedly detects anomalies in similar spatial regions for the
   same pathology subtype, that suggests the signal is real (pathology-driven)
   not random (noise-driven).

Usage: python src/validate_bboxes.py
  (requires the pipeline to have been run first — reads from data/yolo_dataset/)
"""

import os, csv, sys
import numpy as np
import cv2
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_DIR     = os.path.join(PROJECT_ROOT, 'data', 'yolo_dataset')
IMG_DIRS     = {s: os.path.join(YOLO_DIR, 'images', s) for s in ['train', 'val', 'test']}
LBL_DIRS     = {s: os.path.join(YOLO_DIR, 'labels', s) for s in ['train', 'val', 'test']}
OUT_DIR      = os.path.join(PROJECT_ROOT, 'results', 'bbox_validation')
CLASS_NAMES  = {0: 'Malignant', 1: 'Benign', 2: 'NP'}


def compute_colour_contrast(img_bgr, x1, y1, x2, y2, ring_width=20):
    """
    Compare mean LAB colour inside the bbox vs a ring surrounding it.
    Returns the Euclidean distance in LAB space (perceptually uniform).
    Higher = more contrast = more likely to be a distinct region.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Inside bbox
    inner = lab[y1:y2, x1:x2]
    if inner.size == 0:
        return 0.0
    inner_mean = inner.reshape(-1, 3).mean(axis=0)

    # Ring around bbox
    rx1 = max(0, x1 - ring_width)
    ry1 = max(0, y1 - ring_width)
    rx2 = min(w, x2 + ring_width)
    ry2 = min(h, y2 + ring_width)

    # Mask: ring region minus inner region
    mask = np.zeros((h, w), dtype=bool)
    mask[ry1:ry2, rx1:rx2] = True
    mask[y1:y2, x1:x2] = False

    ring_pixels = lab[mask]
    if len(ring_pixels) == 0:
        return 0.0
    ring_mean = ring_pixels.mean(axis=0)

    # Euclidean distance in LAB
    return float(np.linalg.norm(inner_mean - ring_mean))


def compute_texture_score(img_bgr, x1, y1, x2, y2):
    """
    Laplacian variance inside the bbox — measures texture complexity.
    Higher = more textured = more likely to be real tissue.
    Very low = featureless region or edge artefact.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    return float(lap.var())


def is_edge_bbox(xc, yc, bw, bh, margin=0.05):
    """
    Check if bbox is touching the image edge (within margin fraction).
    Edge bboxes are more likely to be border artefacts.
    """
    return (xc - bw/2 < margin or xc + bw/2 > 1 - margin or
            yc - bh/2 < margin or yc + bh/2 > 1 - margin)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Automated BBox Quality Validation ===\n")

    all_results = []
    per_class_results = defaultdict(list)
    per_class_positions = defaultdict(list)  # for consistency test

    n_images_with_box = 0
    n_images_total = 0
    n_empty = 0

    for split in ['train', 'val', 'test']:
        img_dir = IMG_DIRS[split]
        lbl_dir = LBL_DIRS[split]
        if not os.path.isdir(lbl_dir):
            continue

        for lbl_fname in sorted(os.listdir(lbl_dir)):
            if not lbl_fname.endswith('.txt'):
                continue
            n_images_total += 1

            lbl_path = os.path.join(lbl_dir, lbl_fname)
            content = open(lbl_path).read().strip()
            if not content:
                n_empty += 1
                continue

            img_stem = os.path.splitext(lbl_fname)[0]
            # Find corresponding image (try common extensions)
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                candidate = os.path.join(img_dir, img_stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            n_images_with_box += 1

            for line in content.split('\n'):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Convert YOLO normalised to pixel coords
                x1 = max(0, int((xc - bw/2) * w))
                y1 = max(0, int((yc - bh/2) * h))
                x2 = min(w, int((xc + bw/2) * w))
                y2 = min(h, int((yc + bh/2) * h))

                if x2 <= x1 or y2 <= y1:
                    continue

                colour_contrast = compute_colour_contrast(img_bgr, x1, y1, x2, y2)
                texture = compute_texture_score(img_bgr, x1, y1, x2, y2)
                area_pct = bw * bh * 100
                on_edge = is_edge_bbox(xc, yc, bw, bh)

                result = {
                    'file': img_stem,
                    'split': split,
                    'class': CLASS_NAMES.get(cls_id, str(cls_id)),
                    'class_id': cls_id,
                    'xc': xc, 'yc': yc, 'bw': bw, 'bh': bh,
                    'area_pct': area_pct,
                    'colour_contrast': colour_contrast,
                    'texture': texture,
                    'on_edge': on_edge,
                }

                all_results.append(result)
                per_class_results[cls_id].append(result)
                per_class_positions[cls_id].append((xc, yc))

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"Dataset: {n_images_total} images, {n_images_with_box} with bboxes, "
          f"{n_empty} empty labels")
    print(f"Total bboxes analysed: {len(all_results)}\n")

    if not all_results:
        print("No bounding boxes found — nothing to validate.")
        print("Run the pipeline first: python src/main.py --step generate")
        return

    # ── Test 1: Colour Contrast ─────────────────────────────────────────────
    print("─── Test 1: Colour Contrast (LAB distance, inner vs surrounding ring) ───")
    print("  > 10: good contrast (distinct region)")
    print("  5-10: moderate (might be real)")
    print("  < 5:  poor (probably noise or artefact)\n")

    for cls_id in sorted(per_class_results.keys()):
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        results = per_class_results[cls_id]
        contrasts = [r['colour_contrast'] for r in results]
        good = sum(1 for c in contrasts if c > 10)
        moderate = sum(1 for c in contrasts if 5 <= c <= 10)
        poor = sum(1 for c in contrasts if c < 5)
        print(f"  {cls_name:12s}: n={len(contrasts):3d}  "
              f"mean={np.mean(contrasts):5.1f}  median={np.median(contrasts):5.1f}  "
              f"good={good}  moderate={moderate}  poor={poor}")

    all_contrasts = [r['colour_contrast'] for r in all_results]
    print(f"  {'OVERALL':12s}: n={len(all_contrasts):3d}  "
          f"mean={np.mean(all_contrasts):5.1f}  median={np.median(all_contrasts):5.1f}")

    # ── Test 2: Texture Complexity ──────────────────────────────────────────
    print("\n─── Test 2: Texture Complexity (Laplacian variance) ────────────────────")
    print("  > 500: complex texture (tissue-like)")
    print("  100-500: moderate")
    print("  < 100: smooth/featureless (likely artefact)\n")

    for cls_id in sorted(per_class_results.keys()):
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        results = per_class_results[cls_id]
        textures = [r['texture'] for r in results]
        complex_t = sum(1 for t in textures if t > 500)
        moderate_t = sum(1 for t in textures if 100 <= t <= 500)
        smooth_t = sum(1 for t in textures if t < 100)
        print(f"  {cls_name:12s}: n={len(textures):3d}  "
              f"mean={np.mean(textures):7.1f}  median={np.median(textures):7.1f}  "
              f"complex={complex_t}  moderate={moderate_t}  smooth={smooth_t}")

    # ── Test 3: Edge artefact analysis ──────────────────────────────────────
    print("\n─── Test 3: Edge Placement Analysis ───────────────────────────────────")
    print("  Edge bboxes (within 5% of image border) are more likely artefacts.\n")

    for cls_id in sorted(per_class_results.keys()):
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        results = per_class_results[cls_id]
        on_edge = sum(1 for r in results if r['on_edge'])
        total = len(results)
        print(f"  {cls_name:12s}: {on_edge}/{total} on edge ({100*on_edge/max(total,1):.0f}%)")

    all_edge = sum(1 for r in all_results if r['on_edge'])
    print(f"  {'OVERALL':12s}: {all_edge}/{len(all_results)} on edge "
          f"({100*all_edge/max(len(all_results),1):.0f}%)")

    # ── Test 4: Cross-image spatial consistency ─────────────────────────────
    print("\n─── Test 4: Spatial Consistency (same class → similar regions?) ─────────")
    print("  Low std of bbox centers → consistent targeting (pathology-driven)")
    print("  High std → scattered detections (noise-driven)\n")

    for cls_id in sorted(per_class_positions.keys()):
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        positions = np.array(per_class_positions[cls_id])
        if len(positions) < 3:
            print(f"  {cls_name:12s}: too few bboxes ({len(positions)}) for consistency analysis")
            continue
        xc_std = positions[:, 0].std()
        yc_std = positions[:, 1].std()
        print(f"  {cls_name:12s}: n={len(positions):3d}  "
              f"center x_std={xc_std:.3f}  y_std={yc_std:.3f}  "
              f"{'(scattered)' if xc_std > 0.3 and yc_std > 0.3 else '(some consistency)'}")

    # ── Quality score per bbox ──────────────────────────────────────────────
    print("\n─── Per-BBox Quality Scores ────────────────────────────────────────────")
    print("  Each bbox gets a 0-3 quality score:")
    print("    +1 if colour contrast > 5")
    print("    +1 if texture > 100")
    print("    +1 if NOT on edge\n")

    quality_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in all_results:
        q = 0
        if r['colour_contrast'] > 5:
            q += 1
        if r['texture'] > 100:
            q += 1
        if not r['on_edge']:
            q += 1
        quality_counts[q] += 1
        r['quality'] = q

    for q in [3, 2, 1, 0]:
        label = {3: 'High', 2: 'Medium', 1: 'Low', 0: 'Suspect'}[q]
        count = quality_counts[q]
        print(f"  Score {q} ({label:7s}): {count}/{len(all_results)} "
              f"({100*count/max(len(all_results),1):.0f}%)")

    # ── Worst bboxes (for manual spot-check) ────────────────────────────────
    suspect = [r for r in all_results if r['quality'] == 0]
    if suspect:
        print(f"\n  ⚠ {len(suspect)} suspect bboxes (score=0, likely artefacts):")
        for r in suspect[:10]:
            print(f"    {r['file']}: class={r['class']}  "
                  f"contrast={r['colour_contrast']:.1f}  "
                  f"texture={r['texture']:.1f}  edge={r['on_edge']}")

    # ── Save detailed CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'bbox_quality_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'split', 'class', 'xc', 'yc', 'bw', 'bh', 'area_pct',
            'colour_contrast', 'texture', 'on_edge', 'quality'
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\nDetailed report saved to: {csv_path}")

    # ── Summary plot ────────────────────────────────────────────────────────
    if len(all_results) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Colour contrast distribution
        for cls_id in sorted(per_class_results.keys()):
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            vals = [r['colour_contrast'] for r in per_class_results[cls_id]]
            if vals:
                axes[0].hist(vals, bins=20, alpha=0.5, label=cls_name)
        axes[0].axvline(x=5, color='r', linestyle='--', label='threshold')
        axes[0].set_xlabel('Colour Contrast (LAB)')
        axes[0].set_title('Test 1: Colour Contrast')
        axes[0].legend()

        # Texture distribution
        for cls_id in sorted(per_class_results.keys()):
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            vals = [r['texture'] for r in per_class_results[cls_id]]
            if vals:
                axes[1].hist(vals, bins=20, alpha=0.5, label=cls_name)
        axes[1].axvline(x=100, color='r', linestyle='--', label='threshold')
        axes[1].set_xlabel('Texture (Laplacian Var)')
        axes[1].set_title('Test 2: Texture Complexity')
        axes[1].legend()

        # Bbox position scatter
        for cls_id in sorted(per_class_positions.keys()):
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            positions = np.array(per_class_positions[cls_id])
            axes[2].scatter(positions[:, 0], positions[:, 1],
                          alpha=0.5, label=cls_name, s=30)
        axes[2].set_xlim(0, 1); axes[2].set_ylim(1, 0)
        axes[2].set_xlabel('x_center'); axes[2].set_ylabel('y_center')
        axes[2].set_title('Test 4: BBox Position Distribution')
        axes[2].legend()
        axes[2].set_aspect('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'bbox_quality_summary.png'), dpi=150)
        plt.close()
        print(f"Summary plot saved to: {os.path.join(OUT_DIR, 'bbox_quality_summary.png')}")


if __name__ == '__main__':
    main()
