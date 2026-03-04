"""
DINO Self-Supervised Backbone Fine-Tuning for Endoscopic Feature Extraction
============================================================================
Tier 3 upgrade: Fine-tune the ResNet50 backbone on 5,957+ Normal endoscopic
images using DINO (Self-Distillation with No Labels, Caron et al., ICCV 2021).

Why DINO?
---------
The core limitation of Method 3 (PatchCore) is that the frozen ImageNet ResNet50
has never seen endoscopic tissue.  Its feature space groups pathological tissue
(especially smooth anomalies like nasal polyps) close to Normal mucosa, because
both look "fleshy" to an ImageNet backbone trained on dogs, cars, and landscapes.

DINO self-supervised learning on the 5,957 Normal endoscopic images will:
  1. Tighten the Normal cluster in feature space (Normal images become MORE
     similar to each other, as the backbone learns domain-specific invariances)
  2. Widen the gap to pathological tissue (anomalies that are structurally
     different but pixel-similar will now be farther from Normal)
  3. Learn endoscope-specific invariances (lighting, specular reflections,
     camera angle) as augmentation-invariant features

This is the expert-endorsed "real solution" for the tight signal gap problem
that caused Normal FP rates of 47-53% across v6-v8.

Architecture
------------
  Teacher: ResNet50 (EMA-updated copy of student)
  Student: ResNet50 (trained via gradient descent)
  Projection head: MLP (2048 → 256 → 256 → 65536)
  Loss: Cross-entropy on centering + sharpening (self-distillation)

  The backbone weights (everything before the projection head) are saved and
  used as drop-in replacement for ImageNet weights in PatchCoreExtractor.

Usage
-----
    # From main.py:
    python src/main.py --step finetune --finetune-epochs 100

    # Standalone:
    python src/finetune_backbone.py --epochs 100 --batch-size 32

Output
------
    models/dino_resnet50.pth  — Fine-tuned backbone state_dict (no proj head)

Reference: Caron et al., "Emerging Properties in Self-Supervised Vision
Transformers", ICCV 2021.  We adapt the framework for ResNet50 (CNN backbone).
"""

import os
import sys
import argparse
import math
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Normal image directories (same as PatchCore bank sources)
NORMAL_DIRS = [
    os.path.join(PROJECT_ROOT, "data", "sample_data", "train", "Normal"),
    os.path.join(PROJECT_ROOT, "data", "normal_endoscopic"),
]

# Output path for fine-tuned weights
BACKBONE_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "dino_resnet50.pth")
FINETUNE_LOG_PATH  = os.path.join(PROJECT_ROOT, "models", "dino_training_log.json")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EPOCHS      = 100
DEFAULT_BATCH_SIZE  = 32      # adjust for GPU VRAM; 32 fits in ~6 GB
DEFAULT_LR          = 5e-4    # base learning rate (cosine schedule)
DEFAULT_WARMUP      = 10      # warmup epochs
DEFAULT_IMG_SIZE    = 224     # DINO standard
DEFAULT_OUT_DIM     = 65536   # projection head output dimension
DEFAULT_HIDDEN_DIM  = 2048    # projection head hidden dimension
DEFAULT_BOTTLENECK  = 256     # projection head bottleneck
DEFAULT_MOMENTUM    = 0.996   # EMA momentum start (increases to 1.0)
DEFAULT_TEMP_S      = 0.1     # student temperature
DEFAULT_TEMP_T      = 0.04    # teacher temperature (lower = sharper)
DEFAULT_CENTER_MOM  = 0.9     # centering momentum
DEFAULT_NUM_WORKERS = 4


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class NormalEndoscopicDataset(Dataset):
    """
    Loads all Normal endoscopic images from the configured directories.
    Each __getitem__ returns TWO different augmented views of the same image
    (DINO multi-crop is simplified to 2 global crops for ResNet50).
    """
    def __init__(self, image_dirs, img_size=DEFAULT_IMG_SIZE):
        self.paths = []
        for d in image_dirs:
            if not os.path.isdir(d):
                print(f"  Warning: {d} not found, skipping")
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.paths.append(os.path.join(d, f))

        print(f"  DINO dataset: {len(self.paths)} Normal images from {len(image_dirs)} dirs")

        # Global crop augmentation (2 views per image)
        # Heavy augmentation is key to DINO — forces the backbone to learn
        # invariances to lighting, specular reflections, camera angle
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


# ══════════════════════════════════════════════════════════════════════════════
#  DINO Projection Head
# ══════════════════════════════════════════════════════════════════════════════

class DINOHead(nn.Module):
    """
    3-layer MLP projection head: backbone_dim → hidden → bottleneck → out_dim.
    The last layer is weight-normalized (no bias) following DINO paper.
    """
    def __init__(self, in_dim=2048, hidden_dim=DEFAULT_HIDDEN_DIM,
                 bottleneck_dim=DEFAULT_BOTTLENECK, out_dim=DEFAULT_OUT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # Last layer: L2-normalized weights, no bias
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Fix norm to 1
        self.last_layer.parametrizations.weight.original1.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)  # L2-normalize before last layer
        x = self.last_layer(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  DINO Loss
# ══════════════════════════════════════════════════════════════════════════════

class DINOLoss(nn.Module):
    """
    Self-distillation loss: cross-entropy between sharpened teacher output
    and student output, with centering to prevent collapse.
    """
    def __init__(self, out_dim, temp_teacher=DEFAULT_TEMP_T,
                 temp_student=DEFAULT_TEMP_S, center_momentum=DEFAULT_CENTER_MOM):
        super().__init__()
        self.temp_teacher = temp_teacher
        self.temp_student = temp_student
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        student_output: list of 2 tensors (one per view)
        teacher_output: list of 2 tensors (one per view)
        """
        # Teacher: center + sharpen
        teacher_out = [(t - self.center) / self.temp_teacher for t in teacher_output]
        teacher_probs = [F.softmax(t, dim=-1).detach() for t in teacher_out]

        # Student: soften
        student_log_probs = [F.log_softmax(s / self.temp_student, dim=-1)
                             for s in student_output]

        # Cross-entropy: each teacher view supervises the OTHER student view
        total_loss = 0
        n_loss_terms = 0
        for i_t, tp in enumerate(teacher_probs):
            for i_s, slp in enumerate(student_log_probs):
                if i_t == i_s:
                    continue  # skip same-view pairs
                loss = -(tp * slp).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center (EMA of teacher output means)
        with torch.no_grad():
            batch_center = torch.cat([t for t in teacher_output]).mean(dim=0, keepdim=True)
            self.center = (self.center * self.center_momentum +
                           batch_center * (1 - self.center_momentum))

        return total_loss


# ══════════════════════════════════════════════════════════════════════════════
#  Cosine Schedule Helpers
# ══════════════════════════════════════════════════════════════════════════════

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0):
    """Returns a list of values following cosine annealing with warmup."""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs).tolist()
    iters = np.arange(epochs - warmup_epochs)
    cosine = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return warmup_schedule + cosine.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_dino(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
               lr=DEFAULT_LR, warmup_epochs=DEFAULT_WARMUP,
               num_workers=DEFAULT_NUM_WORKERS,
               session=None):
    """
    Train DINO self-supervised on Normal endoscopic images.

    The fine-tuned backbone (ResNet50 without projection head) is saved to
    models/dino_resnet50.pth for use as a drop-in replacement in PatchCore.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  DINO fine-tuning — device: {device}")

    if device.type == 'cpu':
        print("  *** WARNING: DINO training on CPU will be extremely slow. ***")
        print("  *** GPU is strongly recommended (cloud server).           ***")

    # ── Run directory ─────────────────────────────────────────────────────
    try:
        from run_manager import create_run_dir
        run_dir = create_run_dir('finetune', version='v10', params={
            'method': 'DINO',
            'backbone': 'ResNet50',
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'warmup_epochs': warmup_epochs,
            'img_size': DEFAULT_IMG_SIZE,
            'out_dim': DEFAULT_OUT_DIM,
            'momentum_range': f'{DEFAULT_MOMENTUM}→1.0',
            'temp_student': DEFAULT_TEMP_S,
            'temp_teacher': DEFAULT_TEMP_T,
        }, session=session)
    except ImportError:
        run_dir = os.path.join(PROJECT_ROOT, 'results', 'finetune')
        os.makedirs(run_dir, exist_ok=True)

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    dataset = NormalEndoscopicDataset(NORMAL_DIRS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    # ── Student & Teacher networks ────────────────────────────────────────
    # Both are ResNet50 + DINOHead.  Teacher is EMA of student.
    backbone_dim = 2048  # ResNet50 final layer output

    student_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    teacher_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Remove the classification head (avgpool + fc) — we add our own
    student_backbone.fc = nn.Identity()
    teacher_backbone.fc = nn.Identity()

    student_head = DINOHead(in_dim=backbone_dim)
    teacher_head = DINOHead(in_dim=backbone_dim)

    # Move to device
    student_backbone = student_backbone.to(device)
    teacher_backbone = teacher_backbone.to(device)
    student_head = student_head.to(device)
    teacher_head = teacher_head.to(device)

    # Teacher starts as copy of student, no gradients
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    # ── Loss, Optimizer, Schedulers ───────────────────────────────────────
    criterion = DINOLoss(out_dim=DEFAULT_OUT_DIM).to(device)

    # Only optimize student parameters
    params = list(student_backbone.parameters()) + list(student_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.04)

    # Cosine schedules
    lr_schedule = cosine_scheduler(lr, 1e-6, epochs, warmup_epochs)
    momentum_schedule = cosine_scheduler(DEFAULT_MOMENTUM, 1.0, epochs)

    # ── Training ──────────────────────────────────────────────────────────
    training_log = {
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'dataset_size': len(dataset),
            'device': str(device),
        },
        'epoch_losses': [],
    }

    best_loss = float('inf')
    print(f"\n  Training DINO for {epochs} epochs on {len(dataset)} images…")
    print(f"  Batch size: {batch_size}, LR: {lr}, Warmup: {warmup_epochs} epochs")
    print()

    for epoch in range(epochs):
        student_backbone.train()
        student_head.train()

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1:3d}/{epochs}",
                    leave=False, file=sys.stdout)

        for view1, view2 in pbar:
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Student forward
            s_feat1 = student_backbone(view1)
            s_feat2 = student_backbone(view2)
            s_out1 = student_head(s_feat1)
            s_out2 = student_head(s_feat2)

            # Teacher forward (no grad)
            with torch.no_grad():
                t_feat1 = teacher_backbone(view1)
                t_feat2 = teacher_backbone(view2)
                t_out1 = teacher_head(t_feat1)
                t_out2 = teacher_head(t_feat2)

            loss = criterion([s_out1, s_out2], [t_out1, t_out2])

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (DINO standard)
            nn.utils.clip_grad_norm_(params, max_norm=3.0)

            optimizer.step()

            # EMA update teacher
            m = momentum_schedule[epoch]
            with torch.no_grad():
                for ps, pt in zip(student_backbone.parameters(),
                                  teacher_backbone.parameters()):
                    pt.data = m * pt.data + (1 - m) * ps.data
                for ps, pt in zip(student_head.parameters(),
                                  teacher_head.parameters()):
                    pt.data = m * pt.data + (1 - m) * ps.data

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_schedule[epoch]:.2e}")

        avg_loss = epoch_loss / max(n_batches, 1)
        training_log['epoch_losses'].append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': lr_schedule[epoch],
            'momentum': momentum_schedule[epoch],
        })

        # Print every 10 epochs or at start/end
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"lr={lr_schedule[epoch]:.2e}  ema_m={momentum_schedule[epoch]:.4f}")

        # Save best backbone
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_backbone(student_backbone, epoch + 1, avg_loss)

    # ── Save final + log ──────────────────────────────────────────────────
    _save_backbone(student_backbone, epochs, best_loss, tag='final')

    training_log['best_loss'] = best_loss
    training_log['save_path'] = BACKBONE_SAVE_PATH

    log_path = os.path.join(run_dir, 'dino_training_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)

    # Also save to models/ for easy reference
    with open(FINETUNE_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n  DINO training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Backbone saved to: {BACKBONE_SAVE_PATH}")
    print(f"  Training log: {log_path}")

    return BACKBONE_SAVE_PATH


def _save_backbone(backbone, epoch, loss, tag='best'):
    """Save backbone state_dict (without projection head)."""
    os.makedirs(os.path.dirname(BACKBONE_SAVE_PATH), exist_ok=True)

    if tag == 'best':
        save_path = BACKBONE_SAVE_PATH
    else:
        base, ext = os.path.splitext(BACKBONE_SAVE_PATH)
        save_path = f"{base}_{tag}{ext}"

    # Save only the backbone, not the projection head
    # The avgpool layer is preserved, but fc is Identity — we save the
    # state_dict so PatchCoreExtractor can load subsets of it
    state = {
        'backbone_state_dict': backbone.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'tag': tag,
        'timestamp': datetime.now().isoformat(),
        'architecture': 'resnet50',
        'method': 'DINO self-supervised',
        'training_data': 'Normal endoscopic (5,957+ images)',
    }
    torch.save(state, save_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='DINO self-supervised fine-tuning on Normal endoscopic images')
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                   help=f'Training epochs (default: {DEFAULT_EPOCHS})')
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                   help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    p.add_argument('--lr', type=float, default=DEFAULT_LR,
                   help=f'Base learning rate (default: {DEFAULT_LR})')
    p.add_argument('--warmup', type=int, default=DEFAULT_WARMUP,
                   help=f'Warmup epochs (default: {DEFAULT_WARMUP})')
    p.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                   help=f'DataLoader workers (default: {DEFAULT_NUM_WORKERS})')
    return p.parse_args()


def main(session=None):
    args = parse_args()
    train_dino(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup,
        num_workers=args.num_workers,
        session=session,
    )


if __name__ == '__main__':
    main()
