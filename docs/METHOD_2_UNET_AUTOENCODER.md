# Method 2: UNetAutoencoder Reconstruction-Based Anomaly Detection

**Project:** Trustworthy Endoscopic AI with YOLOv11
**Status:** Fully implemented and trained. **Abandoned** â€” fundamental failure mode identified
during evaluation. Superseded by Method 3 (PatchCore).
**Code preserved in:** `src/model.py`, `src/train_autoencoder.py`, `models/autoencoder_best.pth`

---

## Concept

Train a convolutional autoencoder exclusively on Normal endoscopic images. The model learns a
compressed latent representation of healthy tissue. When an abnormal image is passed through
the trained model, the decoder should reconstruct "what healthy tissue would look like there",
producing high pixel-level error at the site of the pathology.

This method was selected as Method 2 because:
- It required no external models or text prompts (unlike Method 1 Grad-CAM WSOD)
- The anomaly detection is inherently explainable (the difference map is direct evidence)
- It is well-established in the industrial defect detection literature
- The dataset's structure (Normal / abnormal split) maps directly onto the training paradigm

---

## Architecture â€” `UNetAutoencoder`

A U-Net style convolutional autoencoder. Skip connections were added to address the known
limitation of plain CAEs: blurry reconstructions that can mask high-frequency texture errors.

```
Encoder                       Bottleneck          Decoder
â”€â”€â”€â”€â”€â”€â”€â”€                      â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€          â”€â”€â”€â”€â”€â”€â”€
DoubleConv(3â†’32)              DoubleConv          up4 + concat(enc4) â†’ DoubleConv
       â†“ MaxPool                512ch             up3 + concat(enc3) â†’ DoubleConv
DoubleConv(32â†’64)         (16Ã—16 spatial)         up2 + concat(enc2) â†’ DoubleConv
       â†“ MaxPool                                  up1 + concat(enc1) â†’ DoubleConv
DoubleConv(64â†’128)                                Conv1Ã—1 + Sigmoid â†’ output
       â†“ MaxPool
DoubleConv(128â†’256)
       â†“ MaxPool
```

Input/Output: (3, 256, 256). Total parameters: ~7.7 M.

**Skip connections:** Each encoder stage is concatenated with the corresponding decoder stage
before the double convolution. This allows high-frequency spatial details to bypass the
bottleneck entirely.

---

## Training

- **Dataset:** `data/sample_data/train/Normal/` (48 images)
- **Loss:** Mean Squared Error (MSE) between input and reconstruction
- **Optimizer:** Adam, lr=1e-3, ReduceLROnPlateau scheduler
- **Epochs:** 50 with early stopping on validation loss
- **Augmentation:** Horizontal/vertical flips, mild colour jitter
- **Result:** Training and validation loss converged. Model saved to `models/autoencoder_best.pth`.

---

## Failure Analysis

### Observed behaviour
When pathological images (Malignant, Benign, NP) were passed through the trained model,
reconstruction quality was **indistinguishable from Normal images**. The pixel difference map
showed near-zero values across the entire image including the tumour region. No localised spike
in reconstruction error was observed.

### Root cause

**Skip connections defeat the purpose of the bottleneck for anomaly detection.**

In a plain autoencoder, the bottleneck acts as an information filter: only patterns seen
frequently enough during training are efficiently encoded. Unusual patterns (tumours) cannot be
encoded, so the decoder reconstructs background instead â€” producing high error at the tumour.

With U-Net skip connections, the encoder does not need to compress spatial detail through the
bottleneck at all. The skip paths pass the original feature maps directly to the decoder,
bypassing the bottleneck. The model can reconstruct **any** spatial content regardless of
whether it was seen during training.

**Pixel texture homogeneity compounds this.** Nasal endoscopic tumour tissue and normal nasal
mucosa share the same colour palette, reflectance profile, and pixel-level texture â€” they are
indistinguishable at the signal level used by the MSE loss. Even a plain CAE (no skip
connections) would likely fail here, because the signal that indicates "this is abnormal"
(semantic tissue organisation, structural topology) is not captured by pixel-level loss.

### Why skip connections were initially chosen
The original motivation was to prevent blurry reconstructions, which is valid for applications
where reconstruction fidelity matters. For anomaly detection in texturally homogeneous tissue,
they are counterproductive. This is a well-known tension in autoencoder-based anomaly detection
and represents a genuine lesson for this project.

---

## Summary Table

| Aspect | Result |
|---|---|
| Architecture implemented | âœ… UNetAutoencoder (~7.7 M params) |
| Training completed | âœ… 50 epochs, loss converged |
| Reconstruction quality on Normal | âœ… High quality |
| Reconstruction quality on Malignant | âŒ Also high quality (failure) |
| Difference map signal | âŒ Near-zero for all tissue types |
| Localisation performance | âŒ Not usable |
| Root cause | Skip connections + pixel-texture homogeneity |
| Resolution | Method 3 (PatchCore): feature-space distance, no training required |

---

## Preserved Artefacts

| File | Description |
|---|---|
| `src/model.py` | `UNetAutoencoder` and `DoubleConv` class definitions |
| `src/train_autoencoder.py` | Training loop with loss plotting |
| `src/dataset.py` | `EndoscopicDataset`, `SubsetWithTransform`, `get_transforms()` |
| `models/autoencoder_best.pth` | Trained weights (kept for reference / future experiments) |


---
