# Medical Imaging AI Pipeline - Implementation Plan

**Project:** Blueprint for a Trustworthy Endoscopic AI Pipeline  
**Created:** January 10, 2026  
**Status:** Planning Phase

---

## Executive Summary

This document outlines the step-by-step implementation plan for building a trustworthy endoscopic AI pipeline based on YOLOv11 architecture with multi-modal late-fusion and explainability features.

---

## Why PyTorch Instead of TensorFlow/Keras?

**PyTorch is the recommended framework for this project for the following reasons:**

### 1. **YOLOv11 Native Support**
- YOLOv11 (Ultralytics) is built natively in PyTorch
- Official implementation and pretrained weights available in PyTorch
- No conversion overhead or compatibility issues

### 2. **Research Flexibility**
- More pythonic and intuitive for custom architectures
- Dynamic computational graphs allow easier debugging
- Better for implementing novel fusion and explainability mechanisms

### 3. **Medical Imaging Community**
- Most medical imaging research papers use PyTorch
- Libraries like MONAI (Medical Open Network for AI) are PyTorch-based
- Better community support for medical imaging tasks

### 4. **Explainability Tools**
- Grad-CAM, Captum, and other XAI libraries have better PyTorch integration
- Easier to access and modify intermediate layers for visualization

### 5. **Production Readiness**
- TorchScript and ONNX export for deployment
- TorchServe for model serving
- Compatible with all major cloud platforms

**Note:** While TensorFlow/Keras are excellent frameworks, PyTorch provides a more streamlined path for this specific YOLOv11-based medical imaging project.

---

## Quick Start Summary

### Core Approach

This pipeline implements a **"Segment-to-Box" strategy** where existing segmentation masks (if available) are automatically converted to bounding boxes, significantly reducing manual annotation effort. The system uses **hierarchical labeling** (e.g., Pathology → Polyp → Adenomatous) enabling both broad and specific classification. A **data quality engine** with automated validation ensures annotation consistency, while **semi-automated labeling workflows** combine AI-assisted pre-annotation with human verification.

### Key Differentiators

- **Semi-Automated Labeling:** Use pre-trained models to generate initial annotations, then human experts refine them (reducing annotation time by 60-70%)
- **Quality-First Data Engine:** Automated checks for blur, brightness, bbox validity, and annotation completeness before training
- **Hierarchical Classification:** Multi-level taxonomy allows detection at different diagnostic granularities
- **Built-in Explainability:** Grad-CAM visualizations and attention maps generated alongside predictions for clinical trust
- **Multi-Modal Fusion:** Architecture supports combining image features with clinical metadata when available

### 5-Phase Fast Track

1. **Environment Setup (1-2 days)**
   - Install PyTorch-based stack (chosen for YOLOv11 native support and medical imaging ecosystem)
   - Setup project structure with separated data/model/results directories
   - Configure GPU acceleration if available

2. **Data Acquisition & Smart Preparation (1-2 weeks)**
   - Download public datasets (Kvasir recommended: 8,000 images, pre-labeled)
   - Implement automatic segment-to-bbox conversion if masks exist
   - Split data: 70% train, 15% validation, 15% test with stratification

3. **Semi-Automated Annotation (2-3 weeks for new data)**
   - Use CVAT or LabelStudio with AI-assisted pre-labeling
   - Apply hierarchical taxonomy (parent-child class relationships)
   - Quality control with inter-annotator agreement checks
   - **Key Detail:** If using pre-labeled public datasets, skip to validation phase

4. **Model Training with Quality Monitoring (1-2 weeks)**
   - Fine-tune YOLOv11 on endoscopic data with medical-specific augmentations
   - Monitor via TensorBoard: track mAP, loss curves, and validation samples
   - Implement early stopping and checkpoint saving

5. **Evaluation & Explainability Integration (1 week)**
   - Generate comprehensive metrics (mAP@0.5:0.95, per-class performance)
   - Add Grad-CAM heatmaps to show what model "sees"
   - Create inference pipeline with confidence thresholds

**Total Timeline:** 6-10 weeks for fully functional MVP with explainability

---

## Pipeline Overview

The pipeline consists of:
1. **Data Preparation Strategy** - "Segment to Box" approach
2. **Hierarchical Labeling Strategy** - Structured annotation system
3. **Data Engine & Quality Control** - Automated quality checks
4. **Model Architecture** - YOLOv11 for Detection
5. **Multi-Modal Late-Fusion & Explainability** - Enhanced interpretability

---

## Detailed Step-by-Step Implementation Plan

### Phase 1: Environment Setup & Preparation

#### Step 1.1: Read and Analyze Requirements
- [x] Read New Plan.pdf to understand pipeline architecture
- [ ] Review technical specifications: YOLOv11 for detection, hierarchical labeling, multi-modal fusion
- [ ] Identify key stakeholders: clinical team for validation, annotators for labeling, developers for implementation
- [ ] Document assumptions (e.g., GPU availability, dataset size) and constraints (time, budget, computational resources)

#### Step 1.2: Setup Development Environment

**Action: Create Python virtual environment**
- Navigate to project directory
- Create virtual environment using `venv` module
- Activate the virtual environment (activation command differs by OS)
- Verify Python version is 3.8 or higher

**Action: Install CUDA for GPU support (if available)**
- Check if NVIDIA GPU is detected using system tools
- Install PyTorch with appropriate CUDA version matching your GPU drivers
- Verify GPU is accessible to PyTorch after installation
- Note: CPU-only installation is acceptable for initial development

**Action: Initialize Git repository**
- Initialize Git in project root directory
- Create .gitignore file to exclude: virtual environment, Python cache files, raw data, model checkpoints, environment variables
- Make initial commit with project structure

**Action: Setup IDE (VS Code)**
- [ ] Install Python extension for VS Code
- [ ] Install Jupyter extension for notebooks
- [ ] Configure Python interpreter to use virtual environment
- [ ] Install Pylance for better code intelligence

#### Step 1.3: Install Required Python Packages

**Action: Create requirements.txt file**

Create a requirements.txt in project root with the following categories:

**Core Deep Learning:**
- PyTorch (2.0.0+)
- TorchVision (0.15.0+)
- Ultralytics (8.0.0+) - for YOLOv11

**Image Processing:**
- OpenCV (4.8.0+)
- NumPy (1.24.0+)
- Pillow (10.0.0+)
- scikit-image (0.21.0+)

**Data Manipulation:**
- Pandas (2.0.0+)
- scikit-learn (1.3.0+)

**Data Augmentation:**
- Albumentations (1.3.0+)

**Annotation Tools:**
- LabelMe (5.3.0+)
- pycocotools (2.0.7+)

**Visualization:**
- Matplotlib (3.7.0+)
- Seaborn (0.12.0+)
- TensorBoard (2.13.0+)

**Explainability:**
- pytorch-grad-cam (1.4.0+)
- Captum (0.6.0+)

**Utilities:**
- tqdm (4.65.0+)
- PyYAML (6.0+)
- jsonschema (4.17.0+)

**Action: Install all packages**
- Activate virtual environment
- Install all requirements from requirements.txt file
- Verify core packages (torch, ultralytics, opencv) import successfully

**Action: Install optional packages for advanced features**
- Weights & Biases (wandb) for experiment tracking
- ONNX and ONNX Runtime for model optimization and deployment

#### Step 1.4: Setup Project Folder Structure

**Action: Create complete directory hierarchy**

Create the following folder structure in your project root:

**Data Directories:**
- data/raw - Original endoscopic images
- data/processed - Preprocessed images
- data/annotations/segments - Segmentation masks
- data/annotations/boxes - Bounding box annotations
- data/train/images & data/train/labels - Training set
- data/val/images & data/val/labels - Validation set
- data/test/images & data/test/labels - Test set

**Model Directories:**
- models/pretrained - Pre-trained weights
- models/checkpoints - Training checkpoints
- models/final - Final trained models

**Source Code Directories:**
- src/data_preparation - Data processing scripts
- src/labeling - Hierarchical labeling tools
- src/quality_control - Data engine & QC scripts
- src/model - YOLOv11 architecture modifications
- src/training - Training scripts
- src/fusion - Multi-modal fusion components
- src/explainability - XAI components
- src/utils - Utility functions

**Supporting Directories:**
- configs - Configuration files
- notebooks - Jupyter notebooks for exploration
- tests - Unit tests
- results/metrics - Performance metrics
- results/visualizations - Result visualizations
- results/logs - Training logs
- docs - Documentation

**Action: Create Python package structure**
- Create empty `__init__.py` files in src/ and all subdirectories to make them importable Python modules

**Expected Structure:**
```
Med_Img_Project/
├── data/
│   ├── raw/                    # Original endoscopic images
│   ├── processed/              # Preprocessed images
│   ├── annotations/            # Label files
│   │   ├── segments/          # Segmentation masks
│   │   └── boxes/             # Bounding box annotations
│   ├── train/                 # Training dataset
│   ├── val/                   # Validation dataset
│   └── test/                  # Test dataset
├── models/
│   ├── pretrained/            # Pre-trained weights
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final trained models
├── src/
│   ├── data_preparation/      # Data processing scripts
│   ├── labeling/              # Hierarchical labeling tools
│   ├── quality_control/       # Data engine & QC
│   ├── model/                 # YOLOv11 architecture
│   ├── training/              # Training scripts
│   ├── fusion/                # Multi-modal fusion
│   ├── explainability/        # XAI components
│   └── utils/                 # Utility functions
├── configs/                   # Configuration files
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
├── results/                   # Experiment results
│   ├── metrics/              # Performance metrics
│   ├── visualizations/       # Result visualizations
│   └── logs/                 # Training logs
├── docs/                      # Documentation
└── requirements.txt           # Python dependencies
```

---

### Phase 2: Data Acquisition & Preparation

#### Step 2.1: Acquire Endoscopic Images

**Action: Download Kvasir dataset (recommended starting point)**
- Visit official Kvasir dataset website at datasets.simula.no/kvasir/
- Download kvasir-dataset-v2.zip (contains ~8,000 labeled GI tract images)
- Extract to data/raw directory
- Alternative: Use command-line tools (wget/curl) if available

**Action: Download HyperKvasir (larger, more comprehensive)**
- Visit datasets.simula.no/hyper-kvasir/
- Note: This is a large dataset (~60GB compressed)
- Download the labeled images subset specifically for object detection tasks
- Contains 10,662 labeled images with bounding boxes and segmentation masks

**Other Public Datasets to Consider:**
- **EDD2020:** Endoscopy Disease Detection Challenge dataset at edd2020.grand-challenge.org
- **CVC-ClinicDB:** Polyp detection dataset at polyp.grand-challenge.org/CVCClinicDB/
- **Additional polyp datasets:** Available through polyp.grand-challenge.org

**Action: Verify downloaded data**
- Navigate to data/raw directory
- Count total image files (both .jpg and .png formats)
- List sample filenames to verify structure
- Check that files are readable and not corrupted

**Action: Document dataset metadata**
- [ ] Create `data/raw/README.md` with dataset source, date downloaded, version
- [ ] Note any licensing restrictions or usage terms
- [ ] Record dataset statistics (number of images, classes, etc.)
- [ ] Document any known issues or limitations with the dataset

#### Step 2.2: Data Exploration & Analysis

**Action: Create exploratory data analysis notebook**

Create `notebooks/01_data_exploration.ipynb` to analyze:

**Image Properties Analysis:**
- Load sample of images from data/raw directory
- Extract properties for each image:
  - Width and height dimensions
  - Number of channels (should be 3 for RGB)
  - Aspect ratio (width/height)
  - File format and size
- Create DataFrame to store image metadata
- Generate summary statistics (mean, median, min, max dimensions)

**Visualization Tasks:**
- Create histogram of image widths
- Create histogram of image heights
- Create histogram of aspect ratios
- Identify common dimensions (most images should have similar sizes)
- Save visualizations to results/visualizations/

**Action: Detect corrupted or unreadable images**
- Iterate through all image files
- Attempt to load each image using image processing library
- Catch and log any files that fail to load
- Record list of corrupted files for exclusion
- Report percentage of corrupted images

#### Step 2.2.1: Split Dataset into Train/Val/Test

**Action: Create data splitting script**

Create `src/data_preparation/split_dataset.py` with the following logic:

**Splitting Strategy:**
- Use 70% for training, 15% for validation, 15% for testing (adjust based on dataset size)
- Ensure splits are stratified if possible (maintain class distribution across splits)
- Set random seed for reproducibility
- Verify that train + val + test ratios sum to 1.0

**Implementation Steps:**
1. Gather all image files from source directory (data/raw)
2. Shuffle images randomly to avoid ordering bias
3. Use scikit-learn's train_test_split for splitting:
   - First split: separate test set from train+val
   - Second split: separate train from val
4. For each split (train/val/test):
   - Copy images to appropriate split/images directory
   - Copy corresponding label files (if they exist) to split/labels directory
   - Create directories if they don't exist
5. Print statistics: number of images in each split

**Action: Execute data split**
- Run the splitting script
- Verify output: check that files were copied correctly
- Document split statistics in a log file

#### Step 2.3: Implement "Segment to Box" Conversion
**Purpose:** Convert segmentation masks to bounding boxes for detection

**Action: Create conversion script**

Create `src/data_preparation/mask_to_bbox.py` with mask-to-bbox conversion logic:

**Conversion Algorithm:**
1. **Read Segmentation Mask:**
   - Load mask as grayscale image
   - Get image dimensions (height, width)

2. **Find Object Contours:**
   - Use contour detection to identify separate objects in mask
   - Use EXTERNAL retrieval mode to get outer contours only
   - Use SIMPLE approximation to reduce contour points

3. **Extract Bounding Boxes:**
   - For each contour, calculate minimum bounding rectangle
   - Get rectangle coordinates (x, y, width, height)

4. **Convert to YOLO Format:**
   - Normalize coordinates by image dimensions
   - Calculate center point: x_center = (x + width/2) / image_width
   - Calculate center point: y_center = (y + height/2) / image_height
   - Normalize dimensions: norm_width = width / image_width
   - Format: [class_id, x_center, y_center, width, height]

5. **Validation & Filtering:**
   - Filter out very small boxes (e.g., < 1% of image area)
   - Ensure coordinates are within [0, 1] range
   - Check that boxes don't have zero width or height

**Batch Processing:**
- Create function to process entire directory of masks
- Map class labels from directory structure or filename patterns
- Save each annotation as .txt file matching image filename
- Output to data/annotations/boxes/
- Log processing statistics (number of objects per image)

**Action: Execute conversion**
- Run the conversion script on your mask directory
- Verify output format matches YOLO requirements
- Spot-check a few conversions by visualizing boxes on images

---

### Phase 3: Hierarchical Labeling Strategy

#### Step 3.1: Define Label Hierarchy

**Methodology:** Work with clinical experts to create a multi-level taxonomy that mirrors diagnostic decision-making

**Approach:**
1. **Consult Medical Literature:** Review endoscopy diagnostic criteria and classification systems (e.g., Paris Classification for polyps, Vienna Classification for dysplasia)
2. **Define Parent-Child Relationships:** Structure labels from general to specific, allowing models to learn at multiple granularities
3. **Create Decision Trees:** Document "how to distinguish" between sibling classes (e.g., adenomatous vs hyperplastic polyps)
4. **Establish Inclusion/Exclusion Criteria:** Define what features must be present for each label

**Example Hierarchy with Diagnostic Logic:**
```
Pathology (Abnormal Finding)
├── Polyps (Raised lesion from mucosa)
│   ├── Adenomatous (Neoplastic potential, irregular surface)
│   └── Hyperplastic (Non-neoplastic, smooth surface)
├── Inflammation (Mucosal redness/swelling)
│   ├── Mild (Localized erythema)
│   ├── Moderate (Edema + friability)
│   └── Severe (Ulceration + bleeding)
└── Tumors (Mass lesion)
    ├── Benign (Well-circumscribed, smooth)
    └── Malignant (Irregular borders, ulcerated)
```

**Deliverable:** Create a 1-2 page annotation guide with visual examples for each class and decision flowchart

#### Step 3.2: Setup Semi-Automated Annotation Infrastructure

**Tool Selection Rationale:**

**Option A: CVAT (Computer Vision Annotation Tool) - RECOMMENDED**
- **Why:** Free, open-source, supports AI-assisted annotation
- **How it helps:** Upload a pre-trained model to generate initial boxes, annotators only correct/refine
- **Setup:** Docker-based deployment, supports team collaboration with task assignment
- **Workflow:** Admin creates projects → AI pre-labels → Annotators review → Export to YOLO format

**Option B: Label Studio**
- **Why:** Modern UI, built-in ML backend for active learning
- **How it helps:** Model learns from corrections, gets better at pre-labeling over time
- **Best for:** Larger teams needing advanced project management

**Option C: LabelImg (Basic, Manual)**
- **Why:** Simple, lightweight, no server needed
- **Limitation:** Fully manual, no AI assistance
- **Best for:** Small datasets (<500 images) or proof-of-concept

**Recommended Workflow:**
1. **Initial Setup:** Deploy CVAT on local server or use cloud instance
2. **Pre-labeling:** If you have ANY existing model (even trained on different medical images), use it to generate initial bounding boxes
3. **Human-in-the-Loop:** Annotators review pre-labels, adjusting boxes and confirming classes
4. **Hierarchical Tagging:** Configure CVAT to show hierarchical dropdown (select parent class, then child)
5. **Quality Gates:** Configure minimum box size, aspect ratio limits to prevent errors

**Time Savings:** Semi-automated approach reduces annotation time from ~2 minutes/image to ~30 seconds/image

#### Step 3.3: Execute Hierarchical Annotation with Quality Control

**Annotation Protocol:**

1. **Batch Assignment:** Divide dataset into batches of 100-200 images per annotator
2. **Dual Annotation for Calibration:** Have 10% of images annotated by 2+ people to measure agreement
3. **Use Hierarchical Dropdowns:** Annotator selects broad category first (e.g., "Polyp"), then specific type (e.g., "Adenomatous")
4. **Capture Uncertainty:** Allow annotators to flag "uncertain" cases for expert review

**Quality Control Workflow:**

1. **Inter-Annotator Agreement (IAA):**
   - Calculate Cohen's Kappa or Fleiss' Kappa for multi-annotator scenarios
   - Target: Kappa > 0.75 (substantial agreement)
   - If Kappa < 0.6, revise annotation guidelines and retrain

2. **Automated Validation Checks:**
   - Bbox size: Flag boxes < 20 pixels (likely errors)
   - Aspect ratio: Flag extreme ratios (>10:1 or <1:10)
   - Class distribution: Alert if any class < 5% of dataset (potential missing labels)
   - Image coverage: Ensure all images have at least 1 annotation

3. **Conflict Resolution Process:**
   - For disagreements, convene annotator + clinical expert
   - Document consensus in annotation log
   - Update guidelines if issue is recurring

4. **Version Control:**
   - Use Git to track annotation file changes
   - Tag versions: v1.0 (initial), v1.1 (post-review), v2.0 (final)
   - Keep changelog documenting corrections made

**Deliverable:** Validated annotation dataset with documented IAA scores and quality metrics report

---

### Phase 4: Data Engine & Quality Control

#### Step 4.1: Build Automated Data Quality Pipeline

**Philosophy:** "Garbage in, garbage out" - catch data issues BEFORE training to avoid wasted compute and poor models

**Quality Check Layers:**

**Layer 1: Image Quality Assessment**
- **Blur Detection:** Use Laplacian variance method to detect out-of-focus images
  - Why: Blurry images provide poor training signal
  - Action: Flag images with variance < threshold, review if >5% of dataset is blurry
  - Tool: OpenCV `cv2.Laplacian()` + variance calculation

- **Brightness/Contrast Analysis:** Calculate histogram statistics
  - Why: Under/overexposed images lose important features
  - Action: Check mean pixel intensity (target: 80-180 for 8-bit images)
  - Remediation: Apply histogram equalization or CLAHE preprocessing

- **Resolution Check:** Ensure minimum dimensions (e.g., 256x256)
  - Why: Tiny images lack detail for detection
  - Action: Upscale or reject images below threshold

**Layer 2: Annotation Validation**
- **Completeness:** Verify every image has corresponding label file
  - Tool: Script to compare image filenames with label filenames
  - Action: Generate "missing labels" report for re-annotation

- **Bounding Box Validity:**
  - Check coordinates within [0, 1] range (YOLO normalized format)
  - Verify box area > minimum (e.g., 0.001 = 0.1% of image)
  - Flag boxes with extreme aspect ratios (>20:1 indicates potential error)
  - Ensure boxes don't extend beyond image boundaries

- **Class Distribution Analysis:**
  - Calculate per-class counts and percentages
  - Identify severe imbalances (e.g., one class <1% of dataset)
  - Decision point: Oversample minority class or collect more data?

**Layer 3: Data Integrity**
- **Duplicate Detection:** Use perceptual hashing (pHash) to find near-identical images
  - Why: Duplicates across train/test sets cause inflated metrics
  - Tool: `imagehash` library
  - Action: Remove duplicates or ensure they're in same split

- **Corrupted File Detection:** Attempt to load each image, catch I/O errors
  - Action: Log corrupted files and exclude from dataset

**Implementation Approach:**
Create a `DataQualityChecker` class that runs all checks and generates HTML report with:
- Overall dataset statistics
- Flagged issues with severity (Critical/Warning/Info)
- Visualizations (class distribution bar chart, blur score histogram)
- Action items ranked by priority

**Frequency:** Run quality checks after every annotation batch and before training

#### Step 4.2: Design Medical-Specific Augmentation Strategy

**Augmentation Philosophy:** Augment realistically to simulate real-world variation without creating unrealistic artifacts

**Geometric Augmentations (Moderate Use):**
- **Horizontal Flip (50% probability):** Simulates different endoscope orientations
- **Rotation (±10-15°):** Small angles only - GI tract has natural orientation
- **Scaling (0.8-1.2x):** Simulates zoom in/out during procedure
- **Translation (±10%):** Mimics slight scope movement
- **NOT recommended:** Vertical flips (unrealistic), extreme rotations (>30°)

**Color/Intensity Augmentations (Aggressive Use):**
- **Brightness Adjustment (±20%):** Different light source intensities
- **Contrast (0.8-1.2x):** Variation in tissue appearance
- **Hue Shift (±5%):** Simulates different white balance settings
- **Saturation (0.7-1.3x):** Variation in blood vessel prominence

**Medical-Specific Augmentations:**
- **Gaussian Noise Addition:** Simulates low-light sensor noise
- **Motion Blur (slight):** Simulates movement during imaging
- **JPEG Compression Artifacts:** Simulates different video quality settings
- **Vignette Effect:** Natural darkening at scope edges

**Advanced: Mosaic/Mixup (Use Carefully):**
- **Mosaic:** Combine 4 images into one (YOLOv11 supports this)
  - Benefit: Exposes model to multiple objects/scales simultaneously
  - Risk: Can create unrealistic compositions
  - Recommendation: Use in first 50% of training only

**Class Balancing Strategy:**
- If class A has 1000 samples and class B has 100:
  - Option 1: Oversample minority class (augment class B 10x)
  - Option 2: Undersample majority class (use 100 from class A)
  - Option 3: Weighted loss function (penalize misclassification of rare class more)
  - **Recommendation:** Combination of light oversampling + weighted loss

**Validation of Augmentations:**
- Manually inspect 50-100 augmented samples
- Ask: "Could this image appear in a real endoscopy?"
- If augmentations look artificial, reduce intensity
- Save augmentation examples to documentation

**Tool Choice:** Use `albumentations` library (faster than PyTorch transforms, supports bboxes)

#### Step 4.3: Establish Continuous Data Validation Framework

**Testing Strategy:**

**Unit Tests for Data Pipeline:**
- Test that DataLoader returns correct batch shapes
- Test that labels match corresponding images
- Test that augmentations don't corrupt bboxes
- Test edge cases: empty labels, multiple objects, single-pixel boxes

**Schema Validation:**
- Define expected data format (e.g., YOLO txt format: `class_id x_center y_center width height`)
- Use `jsonschema` or `pydantic` to validate annotation files
- Reject malformed annotations early

**Automated QC Reports:**
- Generate daily/weekly reports showing:
  - Dataset growth (# new images annotated)
  - Quality metrics trends (average blur score, annotation completeness %)
  - Class distribution evolution
- Set up alerts if metrics fall below thresholds

**Continuous Monitoring:**
- Version control data: Use DVC (Data Version Control) or similar
- Track data lineage: Which images came from which source?
- Log all preprocessing steps: filterings, augmentations applied
- Enable rollback: If new data batch degrades performance, revert

**Quality Gate:** Before training, automated check must pass:
- ✅ No corrupted images
- ✅ <5% missing annotations
- ✅ All classes have >50 examples (or document as limitation)
- ✅ No duplicates across train/val/test
- ✅ Blur rate <10%

If gate fails, training is blocked until issues resolved.

---

### Phase 5: YOLOv11 Model Architecture

#### Step 5.1: Understanding YOLOv11
- [ ] Study YOLOv11 architecture paper/documentation
- [ ] Review key components:
  - [ ] Backbone network
  - [ ] Neck (feature pyramid)
  - [ ] Detection heads
- [ ] Understand anchor-free detection mechanism

#### Step 5.2: Model Configuration
- [ ] Download YOLOv11 base weights
- [ ] Create custom configuration file:
  - [ ] Number of classes
  - [ ] Input image size
  - [ ] Anchor settings (if applicable)
  - [ ] Network depth and width
- [ ] Adapt architecture for medical imaging:
  - [ ] Modify input channels if needed
  - [ ] Adjust network capacity for dataset size

#### Step 5.3: Custom Model Modifications
- [ ] Implement medical-specific features:
  - [ ] Enhanced feature extraction for subtle findings
  - [ ] Multi-scale detection for various polyp sizes
  - [ ] Attention mechanisms for critical regions
- [ ] Integrate hierarchical classification head
- [ ] Add auxiliary outputs for explainability

---

### Phase 6: Data Loading & Preprocessing

#### Step 6.1: Create Custom Dataset Class
- [ ] Implement PyTorch Dataset class
- [ ] Handle image loading and preprocessing
- [ ] Parse annotation files (YOLO/COCO format)
- [ ] Implement data augmentation pipeline
- [ ] Add validation and error handling

#### Step 6.2: Setup Data Loaders
- [ ] Configure training DataLoader
- [ ] Configure validation DataLoader
- [ ] Configure test DataLoader
- [ ] Optimize batch size and num_workers
- [ ] Implement data collation functions

#### Step 6.3: Data Normalization
- [ ] Calculate dataset mean and std
- [ ] Implement normalization transforms
- [ ] Test preprocessing pipeline
- [ ] Validate data shapes and ranges

---

### Phase 7: Model Training

#### Step 7.1: Training Configuration
- [ ] Define hyperparameters:
  - [ ] Learning rate and scheduler
  - [ ] Batch size
  - [ ] Number of epochs
  - [ ] Optimizer (AdamW, SGD)
  - [ ] Loss functions
  - [ ] Regularization (weight decay, dropout)
- [ ] Setup mixed precision training (if using GPU)
- [ ] Configure checkpointing strategy

#### Step 7.2: Loss Function Design
- [ ] Implement detection losses:
  - [ ] Localization loss (IoU, GIoU, CIoU)
  - [ ] Classification loss (CrossEntropy, Focal Loss)
  - [ ] Objectness loss
- [ ] Add hierarchical classification loss
- [ ] Weight loss components appropriately

#### Step 7.3: Training Loop Implementation
- [ ] Build training loop with:
  - [ ] Forward pass
  - [ ] Loss computation
  - [ ] Backward pass
  - [ ] Optimizer step
  - [ ] Gradient clipping (if needed)
- [ ] Implement validation loop
- [ ] Add learning rate scheduling
- [ ] Setup early stopping mechanism

#### Step 7.4: Monitoring & Logging
- [ ] Setup TensorBoard/WandB logging
- [ ] Log training metrics:
  - [ ] Loss values
  - [ ] Learning rate
  - [ ] GPU utilization
- [ ] Log validation metrics:
  - [ ] mAP (mean Average Precision)
  - [ ] Precision, Recall, F1
  - [ ] Per-class performance
- [ ] Save best model checkpoints
- [ ] Create visualization of training progress

#### Step 7.5: Create Training Configuration

**Action: Create YOLO configuration file**

Create `configs/yolo_config.yaml` containing:

**Dataset Paths:**
- Absolute path to data root directory
- Relative paths to train/val/test image folders

**Class Definitions:**
- Map class IDs to class names (e.g., 0: polyp, 1: ulcer, 2: inflammation, 3: tumor)
- Specify total number of classes

**Training Hyperparameters:**
- Number of epochs (e.g., 100)
- Batch size (e.g., 16 - adjust based on GPU memory)
- Input image size (e.g., 640x640)
- Device specification (GPU ID or 'cpu')
- Number of worker threads for data loading

**Model Selection:**
- Specify pretrained model size (yolov11n for nano, yolov11s for small, yolov11m for medium)
- Nano is fastest, medium is most accurate

**Augmentation Parameters:**
- HSV color space adjustments (hue, saturation, value)
- Rotation degrees limit
- Translation percentage
- Scaling factor range
- Horizontal flip probability
- Mosaic augmentation strength
- Mixup augmentation strength

**Action: Create training script**

Create `src/training/train_yolo.py` that:
- Checks for GPU availability and prints device information
- Loads pretrained YOLOv11 model
- Calls training method with configuration file
- Enables automatic checkpointing (save every N epochs)
- Enables validation during training
- Generates training plots automatically
- Returns training results for analysis

#### Step 7.6: Execute Training

**Action: Launch training process**
- Activate virtual environment
- Run training script
- Optionally launch TensorBoard in separate terminal for real-time monitoring
- TensorBoard URL will be http://localhost:6006

**Action: Monitor training progress**
- [ ] Watch terminal output for loss values and learning rate
- [ ] Check TensorBoard for loss curves and metrics graphs
- [ ] Review validation images in results directory (shows predictions on val set)
- [ ] Monitor GPU utilization to ensure GPU is being used efficiently
- [ ] Estimate time remaining based on epoch duration

**Action: Post-training analysis**
- Load the best checkpoint (based on validation mAP)
- Run validation on test set
- Extract key metrics: mAP@0.5, mAP@0.5:0.95
- Review per-class performance
- Identify which classes perform well/poorly

---

### Phase 8: Multi-Modal Late-Fusion

#### Step 8.1: Design Fusion Architecture
- [ ] Identify modalities to fuse:
  - [ ] Image features from YOLOv11
  - [ ] Clinical metadata (if available)
  - [ ] Additional imaging modalities
- [ ] Design fusion module:
  - [ ] Concatenation-based fusion
  - [ ] Attention-based fusion
  - [ ] Gating mechanisms

#### Step 8.2: Implement Fusion Layer
- [ ] Create multi-modal input pipeline
- [ ] Build fusion network architecture
- [ ] Integrate with YOLOv11 backbone
- [ ] Add fusion-specific loss terms

#### Step 8.3: Train Fused Model
- [ ] Prepare multi-modal datasets
- [ ] Train end-to-end fused model
- [ ] Compare with single-modality baseline
- [ ] Analyze fusion contribution

---

### Phase 9: Explainability & Interpretability

**Critical Context:** In medical AI, explainability isn't optional - clinicians need to understand WHY the model makes predictions to trust and validate outputs. This phase makes the "black box" transparent.

#### Step 9.1: Implement Gradient-Based Explanations (Grad-CAM)

**What is Grad-CAM:** Visualizes which image regions the model focused on when making a detection/classification

**How it Works:**
1. Take a prediction (e.g., "Polyp detected")
2. Backpropagate gradients to the last convolutional layer
3. Weight activation maps by gradients → creates heatmap
4. Overlay heatmap on original image → shows "what model sees"

**Implementation Strategy:**
- **Library Choice:** Use `pytorch-grad-cam` (actively maintained, YOLOv11 compatible)
- **Target Layer:** Hook into YOLOv11's backbone final layer (typically before detection head)
- **Per-Class Activation:** Generate separate Grad-CAM for each detected class

**Validation Approach:**
- **Clinical Review:** Have gastroenterologists review 100 Grad-CAM outputs
- **Expected:** Heatmaps should highlight polyp margins, texture, color differences
- **Red Flags:** If model focuses on scope artifacts, image borders, or irrelevant regions → investigate training data issues

**Use Cases:**
- Debugging false positives: Did model focus on reflection/artifact?
- Building trust: Clinician sees model "looking at" the same features they would
- Error analysis: Understand why model missed a lesion

#### Step 9.2: Add Attention Visualization (If Using Attention Mechanisms)

**Context:** If you modify YOLOv11 to include attention modules (e.g., CBAM, SE blocks)

**Approach:**
- **Extract Attention Weights:** During forward pass, save attention map outputs
- **Visualization:** Create side-by-side view of original image + attention overlay
- **Interpretation:** 
  - High attention (bright areas) = features model considers important
  - Low attention (dark areas) = suppressed/ignored regions

**Difference from Grad-CAM:**
- Grad-CAM: Post-hoc explanation (generated after prediction)
- Attention: Built-in mechanism (part of model architecture)
- Use both for complementary insights

**Clinical Value:** Shows not just WHERE model looks, but HOW MUCH it weighs different regions

#### Step 9.3: Feature Attribution with SHAP/LIME

**When to Use:**
- When combining image features with clinical metadata (multi-modal fusion)
- To understand contribution of different input modalities

**SHAP (SHapley Additive exPlanations):**
- **What:** Assigns importance score to each input feature
- **Best for:** Tabular data (patient age, procedure type, etc.) combined with imaging
- **Output:** "This prediction was 60% due to image features, 30% due to patient history, 10% due to lesion location"

**LIME (Local Interpretable Model-agnostic Explanations):**
- **What:** Fits simple model around a specific prediction to explain it
- **Best for:** Understanding individual predictions in detail
- **Output:** "Removing this image region changes prediction from 95% polyp to 30% polyp"

**Implementation Approach:**
- Use `shap` Python library with PyTorch backend
- Generate SHAP values for 50-100 representative test samples
- Create summary plots showing average feature importance across dataset

**Clinical Interpretation Guide:**
Document in user manual:
- "Red regions indicate positive contribution to detection"
- "Blue regions indicate features arguing against detection"
- "Color intensity = strength of contribution"

#### Step 9.4: Build Interactive Explainability Dashboard

**Purpose:** Give clinicians a user-friendly interface to explore model predictions with explanations

**Tool Choice:**

**Option A: Gradio (Recommended for MVP)**
- **Why:** Quick to build, automatic API creation, easy to share
- **Setup time:** 1-2 hours for basic interface
- **Features:** Upload image → Get prediction + Grad-CAM overlay + confidence scores

**Option B: Streamlit**
- **Why:** More customizable layouts, better for complex dashboards
- **Setup time:** 3-5 hours
- **Features:** Multi-page app with data exploration, model comparison, batch processing

**Option C: Custom Flask/FastAPI + React**
- **Why:** Full control, production-ready
- **Setup time:** 1-2 weeks
- **Best for:** Deployment to clinical settings

**Dashboard Components:**

1. **Input Section:**
   - Image upload (drag-and-drop)
   - Batch upload option
   - Optional: Enter patient metadata if multi-modal

2. **Prediction Display:**
   - Bounding boxes overlaid on image
   - Class labels with hierarchical path (Pathology → Polyp → Adenomatous)
   - Confidence scores (both detection and classification)

3. **Explainability Panel:**
   - Grad-CAM heatmap toggle (on/off overlay)
   - Attention map (if available)
   - Slider to adjust heatmap opacity
   - Color-coding: Red (high activation), Blue (low activation)

4. **Detailed Analysis:**
   - List of all detected objects with individual explanations
   - Feature importance chart (if using SHAP)
   - Comparison: "Similar cases from training data"

5. **Export Options:**
   - Download annotated image
   - Generate PDF report with explanations
   - Save predictions to CSV for record-keeping

**Clinical Workflow Integration:**
- **Real-time Mode:** Clinician uploads image during procedure → instant feedback
- **Batch Mode:** Upload day's images → get overnight analysis report
- **Review Mode:** Filter to show only high-confidence detections for quick triage

**Quality Assurance Features:**
- "Disagree" button → logs cases where clinician disagrees with model
- Feedback collection → "Was this explanation helpful? Yes/No"
- Uncertainty flagging → Highlight predictions with confidence 50-70% (borderline cases)

**Deliverable:** Working dashboard that non-technical users (clinicians) can use to:
1. Upload endoscopy images
2. Get AI predictions with visual explanations
3. Understand reasoning behind each detection
4. Provide feedback for model improvement

---

### Phase 10: Model Evaluation

#### Step 10.0: Run Inference on Test Set

**Action: Create inference script**

Create `src/model/inference.py` with two main functions:

**Single Image Inference:**
- Load trained model from checkpoint path
- Accept image path and confidence threshold as parameters
- Run prediction on the image
- Save annotated image with bounding boxes
- Save prediction labels to text file
- Include confidence scores in saved output
- Return results object containing boxes, classes, and scores

**Batch Inference:**
- Load trained model once (for efficiency)
- Iterate through all images in test directory
- Run prediction on each image
- Extract and print detection count per image
- Save all results to organized output directory
- Generate summary statistics (average detections per image, etc.)

**Configuration Options:**
- Confidence threshold (default: 0.25, higher = fewer but more confident detections)
- IOU threshold for NMS (non-maximum suppression)
- Output directory structure
- Whether to save visualizations, labels, or both

**Action: Execute inference**
- Run inference script on test set
- Review saved outputs to verify predictions make sense
- Check prediction text files match YOLO label format

#### Step 10.1: Comprehensive Metrics
- [ ] Calculate detection metrics:
  - [ ] mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - [ ] Precision, Recall, F1-score
  - [ ] Confusion matrix
- [ ] Compute per-class metrics
- [ ] Analyze false positives/negatives

#### Step 10.2: Clinical Validation
- [ ] Perform expert review of predictions
- [ ] Calculate clinical accuracy metrics
- [ ] Assess diagnostic concordance
- [ ] Identify failure cases

#### Step 10.3: Robustness Testing
- [ ] Test on out-of-distribution data
- [ ] Evaluate on different imaging devices
- [ ] Assess performance under various conditions
- [ ] Conduct adversarial robustness tests

---

### Phase 11: Optimization & Deployment

#### Step 11.1: Model Optimization
- [ ] Apply model pruning
- [ ] Implement quantization (INT8)
- [ ] Use knowledge distillation
- [ ] Optimize inference speed

#### Step 11.2: Deployment Preparation
- [ ] Convert model to ONNX/TorchScript
- [ ] Build inference API (FastAPI/Flask)
- [ ] Create Docker container
- [ ] Setup model serving (TorchServe/TensorRT)

#### Step 11.3: Production Pipeline
- [ ] Implement preprocessing pipeline
- [ ] Add post-processing (NMS, filtering)
- [ ] Setup monitoring and logging
- [ ] Create CI/CD pipeline

---

### Phase 12: Documentation & Reporting

#### Step 12.1: Technical Documentation
- [ ] Document architecture and design
- [ ] Write API documentation
- [ ] Create user guides
- [ ] Provide code examples

#### Step 12.2: Model Card
- [ ] Document intended use
- [ ] Describe training data
- [ ] Report performance metrics
- [ ] Disclose limitations and biases

#### Step 12.3: Research Paper/Report
- [ ] Write methodology section
- [ ] Present experimental results
- [ ] Compare with baselines
- [ ] Discuss clinical implications

---

## Tools & Technologies Summary

### Development Tools
- Python 3.8+
- PyTorch
- CUDA/cuDNN (for GPU)
- Git for version control

### Key Libraries
- **Deep Learning:** `torch`, `torchvision`, `ultralytics`
- **Image Processing:** `opencv-python`, `Pillow`, `scikit-image`
- **Data Science:** `numpy`, `pandas`, `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`, `tensorboard`
- **Annotation:** `labelme`, `cvat`
- **Explainability:** `grad-cam`, `captum`, `lime`

### Infrastructure
- GPU: NVIDIA RTX 3090/4090 or cloud GPUs (AWS/GCP)
- Storage: Sufficient space for images and models (100GB+)
- Compute: Multi-core CPU for data processing

---

## Success Criteria

- [ ] **Data Quality:** Clean, well-annotated dataset with >95% annotation accuracy
- [ ] **Model Performance:** mAP@0.5 > 0.80 on validation set
- [ ] **Clinical Accuracy:** >90% agreement with expert diagnoses
- [ ] **Explainability:** Clear, interpretable visualizations for predictions
- [ ] **Speed:** Real-time inference (<100ms per image)
- [ ] **Robustness:** Consistent performance across different imaging conditions

---

## Risk Management

### Technical Risks
- **Limited data:** Mitigation through augmentation and transfer learning
- **Class imbalance:** Address via weighted losses and oversampling
- **Overfitting:** Use regularization, dropout, and early stopping
- **Computational resources:** Leverage cloud computing if needed

### Clinical Risks
- **False negatives:** Implement high-recall configurations for critical findings
- **Regulatory compliance:** Follow medical device regulations (FDA/CE)
- **Data privacy:** Ensure HIPAA/GDPR compliance

---

## Timeline Estimate

- **Phase 1-2:** 2-3 weeks (Environment & Data Preparation)
- **Phase 3-4:** 3-4 weeks (Labeling & Quality Control)
- **Phase 5-7:** 4-6 weeks (Model Development & Training)
- **Phase 8-9:** 2-3 weeks (Fusion & Explainability)
- **Phase 10-11:** 2-3 weeks (Evaluation & Optimization)
- **Phase 12:** 1-2 weeks (Documentation)

**Total Estimated Duration:** 14-21 weeks (~3.5-5 months)

---

## Next Steps

1. Review and approve this implementation plan
2. Set up development environment (Step 1.2)
3. Install required packages (Step 1.3)
4. Create project folder structure (Step 1.4)
5. Begin data acquisition (Step 2.1)

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-01-10 | 1.0 | Initial plan created | AI Assistant |

---

## Notes & Follow-ups

- [ ] Schedule kick-off meeting with team
- [ ] Obtain access to medical imaging data
- [ ] Consult with clinical experts for label hierarchy
- [ ] Review regulatory requirements for medical AI
- [ ] Plan compute resources and budget

---

**Document Status:** ✅ Ready for Review  
**Last Updated:** January 10, 2026
