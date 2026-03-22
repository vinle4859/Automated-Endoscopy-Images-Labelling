"""
YOLO v11 Training Script — Endoscopic Lesion Detector
======================================================
Trains a YOLOv11-nano detector on the auto-annotated dataset produced by
src/generate_bboxes.py.

Key design decisions
--------------------
* **Model size**: yolo11n (nano) — the training set is ~230 images after
  splitting, far too small for larger backbones.
* **Augmentation**: heavy, medically-sensible augmentation via Ultralytics'
  built-in pipeline — HSV jitter, horizontal flip, scale, translate, mosaic,
  random erasing.  Vertical flip is disabled (endoscopy images have a
  consistent up/down orientation).
* **Data-leak safety**: all augmentation is applied on-the-fly exclusively to
  the training split. Validation/test images are evaluated without any
  augmentation — this is handled internally by Ultralytics.
* **Early stopping**: patience = 10 epochs of no val-mAP improvement.

Usage
-----
    python src/train_yolo.py                    # defaults
    python src/train_yolo.py --epochs 200       # override epochs
    python src/train_yolo.py --resume           # resume from last.pt
"""

import argparse
import json
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML    = os.path.join(PROJECT_ROOT, 'configs', 'yolo_data.yaml')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'yolo')

def _init_yolo_run_dir(args, session=None) -> str:
    """Create a timestamped run dir for YOLO training, or fall back to RESULTS_DIR."""
    try:
        from run_manager import create_run_dir
        return create_run_dir('yolo', version='yolo11n', params={
            'model': args.model,
            'epochs': args.epochs,
            'patience': args.patience,
            'imgsz': args.imgsz,
            'batch': args.batch,
        }, session=session)
    except ImportError:
        return RESULTS_DIR

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = 'yolo11n.pt'         # Ultralytics YOLOv11-nano
DEFAULT_IMGSZ   = 640
DEFAULT_EPOCHS  = 75
DEFAULT_BATCH   = 16
DEFAULT_PATIENCE = 10
MIN_NON_EMPTY_PCT = 30.0
MAX_EDGE_BOX_PCT = 20.0


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv11 on endoscopic data')
    p.add_argument('--model',    default=DEFAULT_MODEL,   help='Pretrained model to fine-tune')
    p.add_argument('--imgsz',    type=int, default=DEFAULT_IMGSZ)
    p.add_argument('--epochs',   type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--batch',    type=int, default=DEFAULT_BATCH)
    p.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    p.add_argument('--resume',   action='store_true', help='Resume from last.pt')
    p.add_argument('--force-bad-labels', action='store_true',
                   help='Bypass quality gate and force training even with bad labels.')
    return p.parse_args()


def _check_label_quality_gate():
    """Read generated labels and enforce no-go gate for resource savings."""
    gate_json = os.path.join(PROJECT_ROOT, 'data', 'yolo_dataset', 'quality_gate.json')
    if os.path.isfile(gate_json):
        try:
            with open(gate_json, encoding='utf-8') as f:
                payload = json.load(f)
            gate = payload.get('gate', {})
            metrics = payload.get('metrics', {})
            thresholds = gate.get('thresholds', {})
            return {
                'passed': bool(gate.get('overall_pass', False)),
                'mode': gate.get('mode', 'gate_v1'),
                'train_non_empty_pct': float(metrics.get('train', {}).get('non_empty_pct', 0.0)),
                'overall_edge_box_pct': float(metrics.get('overall', {}).get('edge_box_pct', 100.0)),
                'normal_fp_pct': float(metrics.get('normal_negative_control', {}).get('fp_pct', 100.0)),
                'mean_bbox_to_signal_ratio': float(metrics.get('overall', {}).get('mean_bbox_to_signal_ratio', 0.0)),
                'thresholds': {
                    'min_non_empty_pct': thresholds.get('min_non_empty_pct', MIN_NON_EMPTY_PCT),
                    'max_edge_box_pct': thresholds.get('max_edge_box_pct', MAX_EDGE_BOX_PCT),
                    'max_normal_fp_pct': thresholds.get('max_normal_fp_pct'),
                    'max_mean_bbox_to_signal_ratio': thresholds.get('max_mean_bbox_to_signal_ratio'),
                }
            }
        except Exception:
            # Fall back to legacy gate computation if the JSON is missing fields
            # or malformed.
            pass

    yolo_root = os.path.join(PROJECT_ROOT, 'data', 'yolo_dataset', 'labels')
    splits = ['train', 'val', 'test']
    total_boxes = 0
    total_edge = 0
    train_non_empty_pct = 0.0

    for split in splits:
        lbl_dir = os.path.join(yolo_root, split)
        if not os.path.isdir(lbl_dir):
            continue
        txts = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
        non_empty = 0
        boxes = 0
        edge = 0
        for fn in txts:
            content = open(os.path.join(lbl_dir, fn)).read().strip()
            if content:
                non_empty += 1
            for line in content.splitlines():
                s = line.split()
                if len(s) >= 5:
                    boxes += 1
                    xc = float(s[1]); yc = float(s[2])
                    if xc < 0.2 or xc > 0.8 or yc < 0.2 or yc > 0.8:
                        edge += 1
        if split == 'train':
            train_non_empty_pct = 100.0 * non_empty / max(len(txts), 1)
        total_boxes += boxes
        total_edge += edge

    overall_edge_pct = 100.0 * total_edge / max(total_boxes, 1)
    passed = (train_non_empty_pct >= MIN_NON_EMPTY_PCT and
              overall_edge_pct <= MAX_EDGE_BOX_PCT)
    return {
        'passed': passed,
        'mode': 'gate_v1',
        'train_non_empty_pct': train_non_empty_pct,
        'overall_edge_box_pct': overall_edge_pct,
        'normal_fp_pct': None,
        'mean_bbox_to_signal_ratio': None,
        'thresholds': {
            'min_non_empty_pct': MIN_NON_EMPTY_PCT,
            'max_edge_box_pct': MAX_EDGE_BOX_PCT,
            'max_normal_fp_pct': None,
            'max_mean_bbox_to_signal_ratio': None,
        }
    }


def main(session=None):
    args = parse_args()

    gate = _check_label_quality_gate()
    if not gate['passed'] and not args.force_bad_labels:
      print("ERROR: Label quality gate failed. Blocking YOLO training to save resources.")
      print(f"  Gate mode: {gate.get('mode', 'gate_v1')}")
      print(f"  Train non-empty labels: {gate['train_non_empty_pct']:.2f}% "
          f"(min {gate['thresholds']['min_non_empty_pct']}%)")
      print(f"  Edge-centered boxes: {gate['overall_edge_box_pct']:.2f}% "
          f"(max {gate['thresholds']['max_edge_box_pct']}%)")
      if gate.get('thresholds', {}).get('max_normal_fp_pct') is not None:
          print(f"  Normal FP: {gate.get('normal_fp_pct', 100.0):.2f}% "
                f"(max {gate['thresholds']['max_normal_fp_pct']}%)")
      if gate.get('thresholds', {}).get('max_mean_bbox_to_signal_ratio') is not None:
          print("  Mean bbox-to-signal ratio: "
                f"{gate.get('mean_bbox_to_signal_ratio', 0.0):.3f} "
                f"(max {gate['thresholds']['max_mean_bbox_to_signal_ratio']})")
      print("  Regenerate labels with improved v11/v12 settings before training.")
      sys.exit(1)

    # Validate dataset exists
    yolo_img_train = os.path.join(
        PROJECT_ROOT, 'data', 'yolo_dataset', 'images', 'train'
    )
    if not os.path.isdir(yolo_img_train):
        print("ERROR: YOLO training images not found at", yolo_img_train)
        print("       Run `python src/main.py --step generate` first.")
        sys.exit(1)

    n_train = len([f for f in os.listdir(yolo_img_train)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Training images found: {n_train}")
    if n_train == 0:
        print("ERROR: Training split is empty. Aborting.")
        sys.exit(1)

    # ── Import Ultralytics (after validation so errors are clear) ─────────
    from ultralytics import YOLO

    # ── Load model ────────────────────────────────────────────────────────
    if args.resume:
        last_pt = os.path.join(RESULTS_DIR, 'train', 'weights', 'last.pt')
        if not os.path.exists(last_pt):
            print(f"ERROR: --resume specified but {last_pt} not found.")
            sys.exit(1)
        model = YOLO(last_pt)
        print(f"Resuming from {last_pt}")
    else:
        model = YOLO(args.model)
        print(f"Fine-tuning from pretrained {args.model}")

    # ── Augmentation hyper-parameters ─────────────────────────────────────
    # Ultralytics exposes these as training kwargs. We pick medically
    # sensible values — strong colour and spatial jitter to combat the
    # tiny dataset, but no vertical flip (nasal images have fixed vertical
    # orientation) and moderate mosaic (too aggressive mosaic can destroy
    # subtle lesion signals).
    aug_kwargs = dict(
        hsv_h=0.015,          # hue jitter   — mild (endoscopy colour matters)
        hsv_s=0.5,            # saturation   — moderate
        hsv_v=0.3,            # brightness   — moderate
        degrees=15.0,         # rotation ±15°
        translate=0.1,        # translate ±10%
        scale=0.4,            # scale 0.6–1.4×
        shear=0.0,            # no shear (unnecessary for endoscopy)
        perspective=0.0,      # no perspective warp
        flipud=0.0,           # NO vertical flip
        fliplr=0.5,           # horizontal flip 50%
        mosaic=0.8,           # mosaic on 80% of batches
        mixup=0.0,            # no mixup (bboxes are approximate; mixing adds noise)
        copy_paste=0.0,       # not useful with approximate bboxes
        erasing=0.2,          # random erasing 20% — regularisation for small dataset
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Model   : {args.model}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Patience: {args.patience}")
    yolo_run_dir = _init_yolo_run_dir(args, session=session)
    print(f"  Output  : {yolo_run_dir}")
    print(f"{'-'*60}\n")

    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=yolo_run_dir,
        name='train',
        exist_ok=True,
        device='0' if __import__('torch').cuda.is_available() else 'cpu',
        workers=2,                    # keep low on CPU machines
        amp=True,                     # mixed-precision where supported
        resume=args.resume,
        **aug_kwargs,
    )

    print("\nTraining complete. Best weights saved to:")
    best_pt = os.path.join(yolo_run_dir, 'train', 'weights', 'best.pt')
    print(f"  {best_pt}")

    # ── Quick validation on test split ────────────────────────────────────
    yolo_img_test = os.path.join(
        PROJECT_ROOT, 'data', 'yolo_dataset', 'images', 'test'
    )
    if os.path.isdir(yolo_img_test):
        print("\nRunning evaluation on held-out test split...")
        test_model = YOLO(best_pt)
        metrics = test_model.val(
            data=DATA_YAML,
            split='test',
            imgsz=args.imgsz,
            batch=args.batch,
            project=yolo_run_dir,
            name='test_eval',
            exist_ok=True,
        )
        print(f"\n  Test mAP@50    : {metrics.box.map50:.4f}")
        print(f"  Test mAP@50-95 : {metrics.box.map:.4f}")

        # ── Post-training prediction visualizations ───────────────────────
        _visualize_predictions(test_model, yolo_img_test, yolo_run_dir,
                               args.imgsz)
    else:
        print("\n  WARNING: No test split found -- skipping test evaluation.")


def _visualize_predictions(model, test_img_dir, run_dir, imgsz,
                            max_images=30, conf_thresh=0.25):
    """
    Run inference on test images and save side-by-side visualizations:
    Original | Ground-truth boxes | Predicted boxes.

    Saves up to *max_images* plots to <run_dir>/test_predictions/.
    Also produces a summary grid for quick review.
    """
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    pred_dir = os.path.join(run_dir, 'test_predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # Map class IDs → names
    CLASS_NAMES = {0: 'Malignant', 1: 'Benign', 2: 'NP'}
    CLASS_COLORS_PRED = {0: (255, 0, 0), 1: (0, 200, 0), 2: (0, 100, 255)}
    CLASS_COLORS_GT = {0: (180, 0, 0), 1: (0, 130, 0), 2: (0, 60, 180)}

    # Get test images
    test_imgs = sorted([
        os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])[:max_images]

    if not test_imgs:
        print("  No test images found for visualization.")
        return

    # Ground-truth labels directory
    gt_label_dir = os.path.join(
        os.path.dirname(os.path.dirname(test_img_dir)),
        'labels', 'test'
    )

    print(f"\n  Generating prediction visualizations for {len(test_imgs)} test images...")

    # Run batch inference
    results = model.predict(
        source=test_imgs,
        imgsz=imgsz,
        conf=conf_thresh,
        save=False,
        verbose=False,
    )

    grid_images = []
    for img_path, result in zip(test_imgs, results):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]

        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        h, w = orig_rgb.shape[:2]

        # ── Ground-truth overlay ──────────────────────────────────────────
        gt_img = orig_rgb.copy()
        gt_label_path = os.path.join(gt_label_dir, stem + '.txt')
        if os.path.exists(gt_label_path):
            with open(gt_label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        color = CLASS_COLORS_GT.get(cls_id, (128, 128, 128))
                        cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                        label = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
                        cv2.putText(gt_img, f'GT:{label}', (x1, max(y1-5, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── Prediction overlay ────────────────────────────────────────────
        pred_img = orig_rgb.copy()
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = CLASS_COLORS_PRED.get(cls_id, (128, 128, 128))
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                label = f'{CLASS_NAMES.get(cls_id, f"cls{cls_id}")} {conf:.2f}'
                cv2.putText(pred_img, label, (x1, max(y1-5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── Side-by-side plot ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(gt_img)
        axes[1].set_title('Ground Truth (PatchCore)')
        axes[1].axis('off')
        axes[2].imshow(pred_img)
        n_preds = len(boxes) if boxes is not None else 0
        axes[2].set_title(f'YOLO Predictions ({n_preds} boxes)')
        axes[2].axis('off')
        plt.suptitle(fname, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(pred_dir, f'pred_{stem}.png'), dpi=100)
        plt.close()

        # Collect for grid
        grid_images.append((fname, orig_rgb, gt_img, pred_img))

    # ── Summary grid (4×3 per page) ───────────────────────────────────────
    n_per_page = 4
    for page_idx in range(0, len(grid_images), n_per_page):
        batch = grid_images[page_idx:page_idx + n_per_page]
        n_rows = len(batch)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        for row, (fname, orig, gt, pred) in enumerate(batch):
            axes[row, 0].imshow(orig)
            axes[row, 0].set_ylabel(fname, fontsize=8, rotation=0,
                                     labelpad=60, va='center')
            axes[row, 0].axis('off')
            axes[row, 1].imshow(gt)
            axes[row, 1].axis('off')
            axes[row, 2].imshow(pred)
            axes[row, 2].axis('off')
        axes[0, 0].set_title('Original', fontsize=10)
        axes[0, 1].set_title('Ground Truth', fontsize=10)
        axes[0, 2].set_title('YOLO Predictions', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(pred_dir, f'grid_page_{page_idx // n_per_page + 1}.png'),
                    dpi=120)
        plt.close()

    print(f"  Prediction visualizations saved to {pred_dir}")
    print(f"  ({len(grid_images)} images, "
          f"{(len(grid_images) + n_per_page - 1) // n_per_page} grid pages)")


if __name__ == '__main__':
    main()
