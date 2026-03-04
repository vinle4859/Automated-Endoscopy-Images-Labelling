import os
import sys
import argparse
from generate_bboxes import generate_dataset

# Default path for DINO-finetuned backbone weights
DINO_BACKBONE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'dino_resnet50.pth'
)


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

    args = parser.parse_args()

    # Resolve backbone path shorthand
    backbone_path = args.backbone
    if backbone_path is not None:
        if backbone_path.lower() == 'dino':
            backbone_path = DINO_BACKBONE_PATH

    # Determine version tag for session
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
        generate_dataset(rebuild_bank=args.rebuild_bank, session=session,
                         backbone_path=backbone_path)

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
