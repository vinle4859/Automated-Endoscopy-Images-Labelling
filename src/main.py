import os
import sys
import argparse
from generate_bboxes import generate_dataset, run_v11_gate_sweep, run_v12_advanced

# Default path for DINO-finetuned backbone weights
DINO_BACKBONE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'dino_resnet50.pth'
)

V15_DRAFT_MAX_NORMAL_FP_PCT = 20.0
V15_DRAFT_MAX_BBOX_TO_SIGNAL_RATIO = 1.80


def main():
    parser = argparse.ArgumentParser(description="Endoscopic AI Pipeline")
    parser.add_argument('--step', type=str,
                        choices=['finetune', 'generate', 'yolo', 'all'],
                        default='all',
                        help="Which step to run: 'finetune' (DINO backbone), "
                             "'generate' (BBoxes via PatchCore), "
                             "'yolo' (YOLO training), or 'all'.")
    parser.add_argument('--rebuild-bank', action='store_true',
                        help="Force re-extraction of the PatchCore Normal memory bank, "
                             "ignoring any existing cache at models/patchcore_bank.npz. "
                             "Use this when new Normal images have been added.")
    parser.add_argument('--backbone', type=str, default=None,
                        help="Path to a DINO-finetuned backbone (.pth). "
                             "If omitted, uses default ImageNet ResNet50 weights. "
                             "Use 'dino' as shorthand for the default DINO path "
                             f"({DINO_BACKBONE_PATH}).")
    parser.add_argument('--finetune-epochs', type=int, default=100,
                        help="Number of DINO fine-tuning epochs (default: 100).")
    parser.add_argument('--anom-floor', type=float, default=None,
                        help="Override ANOMALY_FLOOR_THRESH for generation.")
    parser.add_argument('--anom-margin', type=float, default=None,
                        help="Override ANOMALY_MARGIN_FRAC for generation.")
    parser.add_argument('--calibration-percentile', type=float, default=99,
                        help="Normal calibration percentile (default: 99).")
    parser.add_argument('--sweep-gates', action='store_true',
                        help="Run v11 gate sensitivity sweep before generation.")
    parser.add_argument('--v12-advanced', action='store_true',
                        help="Run v12 advanced A/B sweep across backbones, then generate with best config.")
    parser.add_argument('--disable-v11-roi', action='store_true',
                        help="Disable v11 ROI mask and border rejection.")
    parser.add_argument('--v15-draft', action='store_true',
                        help="Enable v15 draft gate-v2 constraints (Normal FP + "
                             "bbox-to-signal controls) while using current v13 extraction stack.")

    args = parser.parse_args()

    # Resolve backbone path shorthand
    backbone_path = args.backbone
    if backbone_path is not None:
        if backbone_path.lower() == 'dino':
            backbone_path = DINO_BACKBONE_PATH

    # Determine version tag for session
    if args.v15_draft:
        version = 'v15'
    elif args.v12_advanced:
        version = 'v13'
    else:
        version = 'v10' if backbone_path else 'v9'

    print("==================================================")
    print("   Trustworthy Endoscopic AI Pipeline Started     ")
    print(f"   Version: {version}  |  Backbone: "
          f"{'DINO' if backbone_path else 'ImageNet'}")
    print("==================================================")

    # ── Session grouping (v8) ─────────────────────────────────────────────
    # When --step all, create a shared session so generate + yolo outputs
    # are grouped under one timestamped directory.
    session = None
    if args.step == 'all':
        try:
            from run_manager import start_session
            session = start_session(version=version)
            print(f"\n  Session directory: {session.session_dir}")
        except ImportError:
            pass  # fallback: no session grouping

    if args.step == 'finetune':
        print("\n--- STEP 0: DINO Self-Supervised Backbone Fine-Tuning ---")
        print("    (Fine-tunes ResNet50 on Normal endoscopic images)")
        from finetune_backbone import train_dino
        train_dino(epochs=args.finetune_epochs, session=session)
        return  # finetune is standalone; user runs generate next with --backbone dino

    if args.step in ['generate', 'all']:
        print("\n--- STEP 1: Generating Bounding Boxes via PatchCore Feature-Space Anomaly Detection ---")
        backbone_label = f"DINO ({backbone_path})" if backbone_path else "ImageNet (frozen)"
        print(f"    (Backbone: {backbone_label})")
        if args.v12_advanced:
            run_v12_advanced(
                rebuild_bank=args.rebuild_bank,
                session=session,
                dino_backbone_path=DINO_BACKBONE_PATH if os.path.isfile(DINO_BACKBONE_PATH) else None,
            )
        else:
            best_cfg = None
            if args.sweep_gates:
                best_cfg, _ = run_v11_gate_sweep(backbone_path=backbone_path,
                                                 rebuild_bank=args.rebuild_bank,
                                                 session=session)
            generate_dataset(
                rebuild_bank=args.rebuild_bank,
                session=session,
                backbone_path=backbone_path,
                anom_floor=(best_cfg['floor'] if best_cfg else args.anom_floor),
                anom_margin=(best_cfg['margin'] if best_cfg else args.anom_margin),
                calibration_percentile=(best_cfg['calibration_percentile']
                                        if best_cfg else args.calibration_percentile),
                v11_mode=not args.disable_v11_roi,
                v13_mode=True,
                v15_mode=bool(args.v15_draft),
                max_normal_fp_pct=(V15_DRAFT_MAX_NORMAL_FP_PCT
                                   if args.v15_draft else None),
                max_bbox_to_signal_ratio=(V15_DRAFT_MAX_BBOX_TO_SIGNAL_RATIO
                                          if args.v15_draft else None),
            )

    if args.step in ['yolo', 'all']:
        print("\n--- STEP 2: Training YOLOv11 Detector ---")
        from train_yolo import main as train_yolo_main
        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            train_yolo_main(session=session)
        finally:
            sys.argv = saved_argv

    print("\n==================================================")
    print("   Pipeline Execution Complete!                   ")
    print("==================================================")


if __name__ == "__main__":
    main()
