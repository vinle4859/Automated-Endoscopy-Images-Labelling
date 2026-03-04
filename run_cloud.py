"""
Cloud Orchestration Runner — Full Comparative Study
====================================================
Runs the complete Tier 2 + Tier 3 comparative study on a cloud server.

This script orchestrates ALL pipeline versions in sequence, producing a
side-by-side comparison that can be analysed for the failure mode study.

Versions run:
  v9  (Tier 2): Layer 2+3+4 features (3584-dim), ImageNet backbone, 300K bank
  v10 (Tier 3): Layer 2+3+4 features (3584-dim), DINO-finetuned backbone, 300K bank

Each version runs the full pipeline:
  1. Generate bounding boxes via PatchCore anomaly detection
  2. Train YOLOv11 detector on generated annotations
  3. Record all metrics to results/runs/ with full version tracking

Prerequisites (cloud server setup):
  1. Python 3.10+ with venv
  2. GPU with ≥8 GB VRAM (NVIDIA, CUDA-capable)
  3. ≥32 GB RAM (v9 bank + k-NN requires ~12 GB peak)
  4. Install dependencies:
       pip install -r requirements.txt

Usage:
  # Run full comparative study (DINO fine-tuning → v9 → v10):
  python run_cloud.py --all

  # Run only DINO fine-tuning (Tier 3 backbone):
  python run_cloud.py --finetune-only

  # Run only v9 (Tier 2, ImageNet backbone):
  python run_cloud.py --v9-only

  # Run only v10 (Tier 3, requires DINO backbone at models/dino_resnet50.pth):
  python run_cloud.py --v10-only

  # Skip DINO fine-tuning (if backbone already exists):
  python run_cloud.py --all --skip-finetune

  # Custom DINO epochs:
  python run_cloud.py --all --finetune-epochs 200

Outputs:
  results/runs/<timestamp>_session/    — grouped run directories per version
  models/dino_resnet50.pth             — DINO-finetuned backbone weights
  models/patchcore_bank.npz            — v9 bank (ImageNet backbone)
  models/patchcore_bank_dino.npz       — v10 bank (DINO backbone)

Reference:
  VERSION_REGISTRY.md — full documentation of all versions and results
  docs/METHOD_3_PATCHCORE_IMPLEMENTATION.md — technical architecture details
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Ensure src/ is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DINO_BACKBONE_PATH = os.path.join(PROJECT_ROOT, 'models', 'dino_resnet50.pth')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def banner(text, char='='):
    """Print a banner line."""
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def check_environment():
    """Pre-flight check: verify dependencies and hardware."""
    banner("Environment Check")

    import torch
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {'YES — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO (CPU only)'}")

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM:      {vram:.1f} GB")
    else:
        print("  ⚠ WARNING: No GPU detected. DINO fine-tuning will be EXTREMELY slow.")
        print("           PatchCore + YOLO will still work but slower than GPU.")

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"  RAM:       {ram_gb:.1f} GB")
        if ram_gb < 24:
            print("  ⚠ WARNING: <24 GB RAM. v9/v10 3584-dim bank may be tight.")
            print("           Consider reducing MAX_BANK_PATCHES if OOM occurs.")
    except ImportError:
        print("  RAM:       (install psutil for RAM check: pip install psutil)")

    print(f"  Project:   {PROJECT_ROOT}")
    print(f"  Backbone:  {'EXISTS' if os.path.exists(DINO_BACKBONE_PATH) else 'NOT FOUND'}")
    print()
    return True


def run_finetune(epochs=100):
    """Run DINO self-supervised fine-tuning on Normal endoscopic images."""
    banner("DINO Self-Supervised Fine-Tuning (Tier 3)")

    from finetune_backbone import train_dino
    start = time.time()
    backbone_path = train_dino(epochs=epochs)
    elapsed = time.time() - start

    print(f"\n  DINO fine-tuning completed in {elapsed/60:.1f} minutes")
    print(f"  Backbone saved to: {backbone_path}")
    return backbone_path


def run_version(version_tag, backbone_path=None, description=""):
    """Run full generate + YOLO pipeline for a specific version."""
    banner(f"{version_tag}: {description}")

    from run_manager import start_session
    from generate_bboxes import generate_dataset
    from train_yolo import main as train_yolo_main

    session = start_session(version=version_tag)
    print(f"  Session directory: {session.session_dir}")

    # ── Step 1: Generate bounding boxes ───────────────────────────────────
    print(f"\n  [{version_tag}] Step 1/2: Generating bounding boxes…")
    start = time.time()
    generate_dataset(
        rebuild_bank=True,  # Always rebuild for clean comparison
        session=session,
        backbone_path=backbone_path,
    )
    gen_elapsed = time.time() - start
    print(f"\n  [{version_tag}] Generate completed in {gen_elapsed/60:.1f} minutes")

    # ── Step 2: Train YOLOv11 ─────────────────────────────────────────────
    print(f"\n  [{version_tag}] Step 2/2: Training YOLOv11 detector…")
    start = time.time()
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        train_yolo_main(session=session)
    finally:
        sys.argv = saved_argv
    yolo_elapsed = time.time() - start
    print(f"\n  [{version_tag}] YOLO training completed in {yolo_elapsed/60:.1f} minutes")

    return {
        'version': version_tag,
        'description': description,
        'session_dir': session.session_dir,
        'backbone': backbone_path or 'ImageNet (default)',
        'generate_time_min': round(gen_elapsed / 60, 1),
        'yolo_time_min': round(yolo_elapsed / 60, 1),
    }


def run_comparative_study(args):
    """Run full comparative study across versions."""
    banner("COMPARATIVE STUDY — Full Pipeline", char='#')
    total_start = time.time()
    results = []

    # ── Phase 0: DINO fine-tuning (if needed) ─────────────────────────────
    if not args.skip_finetune and not args.v9_only:
        if os.path.exists(DINO_BACKBONE_PATH) and not args.force_finetune:
            print(f"  DINO backbone already exists at {DINO_BACKBONE_PATH}")
            print(f"  Use --force-finetune to retrain. Skipping.")
        else:
            run_finetune(epochs=args.finetune_epochs)

    # ── Phase 1: v9 (Tier 2 — ImageNet backbone) ─────────────────────────
    if not args.v10_only and not args.finetune_only:
        v9_result = run_version(
            'v9',
            backbone_path=None,
            description='Tier 2 — Layer 2+3+4, ImageNet backbone, 300K bank'
        )
        results.append(v9_result)

    # ── Phase 2: v10 (Tier 3 — DINO backbone) ────────────────────────────
    if not args.v9_only and not args.finetune_only:
        if not os.path.exists(DINO_BACKBONE_PATH):
            print(f"\n  ⚠ DINO backbone not found at {DINO_BACKBONE_PATH}")
            print(f"  Run with --all or --finetune-only first.")
        else:
            v10_result = run_version(
                'v10',
                backbone_path=DINO_BACKBONE_PATH,
                description='Tier 3 — Layer 2+3+4, DINO backbone, 300K bank'
            )
            results.append(v10_result)

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    banner("COMPARATIVE STUDY — Complete")

    print(f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print()

    if results:
        summary_path = os.path.join(RESULTS_DIR, 'cloud_study_summary.json')
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_min': round(total_elapsed / 60, 1),
            'versions': results,
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to: {summary_path}")

        print("\n  Version Results:")
        print(f"  {'Version':<8} {'Generate':<12} {'YOLO':<12} {'Session Dir'}")
        print(f"  {'-------':<8} {'--------':<12} {'----':<12} {'-----------'}")
        for r in results:
            dirname = os.path.basename(r['session_dir'])
            print(f"  {r['version']:<8} {r['generate_time_min']:>6.1f} min   "
                  f"{r['yolo_time_min']:>6.1f} min   {dirname}")

    print("\n  Next steps:")
    print("  1. Check results/runs/ for per-version outputs")
    print("  2. Compare YOLO mAP50 across versions in the training logs")
    print("  3. Update VERSION_REGISTRY.md with actual results")
    print("  4. Review visualizations/ for qualitative comparison")


def main():
    parser = argparse.ArgumentParser(
        description='Cloud Comparative Study — Tier 2 + Tier 3 Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cloud.py --all                  # Full study: DINO → v9 → v10
  python run_cloud.py --all --skip-finetune  # v9 + v10 (DINO already trained)
  python run_cloud.py --finetune-only        # Only DINO fine-tuning
  python run_cloud.py --v9-only              # Only v9 (Tier 2)
  python run_cloud.py --v10-only             # Only v10 (Tier 3, needs DINO)
        """)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all', action='store_true',
                      help='Run full comparative study (DINO + v9 + v10)')
    mode.add_argument('--finetune-only', action='store_true',
                      help='Run only DINO fine-tuning')
    mode.add_argument('--v9-only', action='store_true',
                      help='Run only v9 (Tier 2, ImageNet backbone)')
    mode.add_argument('--v10-only', action='store_true',
                      help='Run only v10 (Tier 3, DINO backbone)')

    parser.add_argument('--finetune-epochs', type=int, default=100,
                        help='DINO fine-tuning epochs (default: 100)')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Skip DINO fine-tuning (use existing backbone)')
    parser.add_argument('--force-finetune', action='store_true',
                        help='Force DINO re-training even if backbone exists')

    args = parser.parse_args()

    check_environment()
    run_comparative_study(args)


if __name__ == '__main__':
    main()
