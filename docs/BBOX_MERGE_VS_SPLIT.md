# Bounding Box Merge vs. Split: Multi-Lesion Disambiguation

> **Status**: Research / Expansion Planning  
> **Created**: 2026-02-24  
> **Priority**: Pivotal for dataset scaling and multi-focal disease

---

## 1. Problem Statement

When PatchCore produces multiple hotspot clusters on an image, the v6 pipeline
merges bounding boxes whose edges are within `BBOX_MERGE_GAP_FRAC` (8%) of
each other.  This was introduced to solve **fragmentation** — a single lesion
producing 2–3 disjoint hotspot clusters due to internal texture variation.

However, this creates a **merge-vs-split dilemma**:

| Scenario | Correct action | Current v6 behaviour |
|---|---|---|
| Single lesion, fragmented into 2 bboxes | **Merge** ✓ | Merges (correct) |
| Two separate lesions, close together | **Keep separate** ✗ | Merges (incorrect) |
| Two lesions, far apart | **Keep separate** ✓ | Keeps separate (correct) |

When two genuinely separate lesions are merged, the resulting bbox:
- Covers both lesions **plus** the healthy tissue between them
- Gives YOLO a confusing training signal (learns "lesion = anything in this large area")
- Loses per-lesion localisation — clinically important for treatment planning
- Artificially inflates bbox area, potentially crossing `MAX_BBOX_AREA_FRAC`

### Why This Becomes Pivotal at Scale

With only ~96 images per class, multi-focal presentation is rare in our current
dataset.  But as data grows:
- **Nasal polyps (NP)** are often bilateral / multifocal
- **Papillomas** can be multifocal (field cancerisation)
- **Metastatic deposits** may appear as multiple discrete lesions
- Post-treatment surveillance images often show residual + recurrence

Incorrect merging would systematically corrupt annotations for these cases.

---

## 2. Current Implementation (v6)

```python
# generate_bboxes.py — _merge_bboxes()
def _should_merge(a, b, gap):
    """True if boxes overlap or edges are within gap (normalised coords)."""
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    x_gap = max(ax1, bx1) - min(ax2, bx2)  # positive = separated
    y_gap = max(ay1, by1) - min(ay2, by2)
    return x_gap < gap and y_gap < gap      # merge if close on BOTH axes
```

**Limitations:**
- Purely geometric — ignores anomaly intensity between boxes
- Fixed threshold (8%) regardless of box sizes
- No validation that merged region is coherent

---

## 3. Proposed Solutions

### 3.1 Valley Analysis (Anomaly Profile Based)  ★ Recommended for MVP

**Idea:** Before merging, sample the anomaly intensity along the line connecting
the two boxes' centres.  If intensity dips below a threshold ("valley"), the
boxes likely correspond to separate lesions.

```
Box A            Valley             Box B
████████▓▓▓▓░░░░░░░░░░▓▓▓▓████████
         ↑  valley < 50% of min(peak_A, peak_B)  → DON'T merge
```

**Implementation sketch:**
```python
def _has_valley(amap, box_a, box_b, valley_threshold=0.5):
    """Check for an intensity valley between two bboxes."""
    ca = center_of(box_a)
    cb = center_of(box_b)
    # Sample N points along line ca→cb
    points = np.linspace(ca, cb, num=20)
    intensities = [amap[int(y), int(x)] for x, y in points]
    min_between = min(intensities[3:-3])  # exclude near-center of each box
    weaker_peak = min(max(intensities[:5]), max(intensities[-5:]))
    return min_between < valley_threshold * weaker_peak
```

**Pros:** Simple, interpretable, directly addresses the root cause  
**Cons:** Assumes anomaly map is smooth; noisy maps may create false valleys  
**Complexity:** ~20 lines of code

---

### 3.2 Area Ratio Constraint  ★ Recommended for MVP

**Idea:** After a candidate merge, compute the ratio of the merged bbox area to
the sum of the original areas.  A high ratio means lots of healthy tissue was
enclosed between the boxes.

```python
def _area_ratio_ok(box_a, box_b, max_ratio=1.5):
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    merged = merge(box_a, box_b)
    area_merged = merged[2] * merged[3]
    return area_merged / (area_a + area_b) <= max_ratio
```

| Situation | area_merged / (area_a + area_b) | Decision |
|---|---|---|
| Overlapping boxes (same lesion) | ~1.0–1.2 | Merge ✓ |
| Adjacent boxes, small gap | ~1.3–1.5 | Merge ✓ |
| Two lesions with tissue gap | ~2.0–3.0+ | Keep separate ✓ |

**Pros:** Extremely simple, no access to anomaly map needed  
**Cons:** Doesn't consider anomaly structure, just geometry  
**Complexity:** ~5 lines of code

---

### 3.3 Aspect Ratio Constraint

**Idea:** After merging, reject if the merged bbox has extreme aspect ratio
(e.g., > 3:1).  Two lesions side-by-side produce a wide merged box.

```python
def _aspect_ok(merged_box, max_aspect=3.0):
    w, h = merged_box[2], merged_box[3]
    aspect = max(w, h) / (min(w, h) + 1e-8)
    return aspect <= max_aspect
```

**Pros:** Trivial to implement  
**Cons:** Fails for vertically/horizontally aligned lesions of similar size;
doesn't distinguish single elongated lesion from two round ones  
**Complexity:** ~3 lines

---

### 3.4 Watershed Segmentation

**Idea:** Treat the anomaly map as a topographic surface.  Apply the watershed
algorithm to identify distinct peaks/basins.  Each basin = one potential
annotation target.

```python
from scipy.ndimage import label as ndlabel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def watershed_split(amap, min_distance=3):
    coords = peak_local_max(amap, min_distance=min_distance)
    markers = np.zeros_like(amap, dtype=int)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1
    labels = watershed(-amap, markers, mask=amap > threshold)
    return labels  # each unique label = one lesion
```

**Pros:** Handles arbitrary shapes, naturally separates multi-focal lesions,
well-studied algorithm  
**Cons:** Requires `skimage` (already available via scikit-image); sensitive to
`min_distance` parameter; may over-segment noisy anomaly maps  
**Complexity:** ~30 lines, replaces current connected-component approach

---

### 3.5 Peak Count / Local Maxima Analysis

**Idea:** Before merging, check if the combined region contains multiple
spatially separated local maxima.  Multiple peaks = multiple lesions.

**Implementation:**
1. Compute local maxima in the anomaly map within the merged bbox
2. Filter peaks by minimum prominence (e.g., > 30% of map max)
3. If ≥ 2 prominent peaks, reject the merge

**Pros:** Directly models the notion "one lesion = one peak"  
**Cons:** Doesn't handle flat-topped lesions well; needs peak prominence tuning  
**Complexity:** ~15 lines using `scipy.signal.find_peaks` on 2D slices

---

### 3.6 Feature Embedding Similarity

**Idea:** Compare the mean ResNet50 feature vectors of patches within each
candidate-merge box.  Similar embeddings → same pathology → likely one
fragmented lesion.  Dissimilar → different pathologies → keep separate.

```python
def _embeddings_similar(feat_map, box_a, box_b, threshold=0.85):
    patches_a = feat_map[box_a_rows, box_a_cols, :]  # (N, 1536)
    patches_b = feat_map[box_b_rows, box_b_cols, :]
    mean_a = patches_a.mean(axis=0)
    mean_b = patches_b.mean(axis=0)
    cos_sim = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
    return cos_sim > threshold
```

**Pros:** Uses rich feature information already computed; can distinguish
different pathology types  
**Cons:** Same pathology type in two locations would still appear similar
(would incorrectly merge two separate SCCs); requires passing feature map to
merge function  
**Complexity:** ~15 lines, but architectural change to pass features through

---

### 3.7 Size-Adaptive Merge Threshold

**Idea:** Instead of fixed 8% gap threshold, scale the merge distance
proportional to the smaller box's size.  Two small boxes close together are
more likely one fragmented lesion; two large boxes close together are more
likely separate.

```python
def _adaptive_gap(box_a, box_b, scale=0.5):
    smaller = min(box_a[2] * box_a[3], box_b[2] * box_b[3])
    return scale * smaller ** 0.5  # merge gap proportional to sqrt(smaller area)
```

**Pros:** Simple, addresses the intuition that "closeness" is relative to size  
**Cons:** Doesn't consider anomaly structure  
**Complexity:** ~5 lines

---

### 3.8 Contour Continuity Check

**Idea:** After thresholding the anomaly map, check if the anomaly contour is
continuous between the two boxes.  If there's a gap in the thresholded contour,
the boxes represent separate regions even if geometrically close.

This is essentially what connected components already check, but at a *lower*
threshold — asking "is there a weak but continuous anomaly bridge?"

**Pros:** Uses existing anomaly map; directly tests physical connectivity  
**Cons:** Sensitive to threshold choice; a thin bridge of specular noise could
falsely connect two separate lesions  
**Complexity:** ~10 lines

---

### 3.9 Clinical Size Priors

**Idea:** Incorporate domain knowledge about typical lesion sizes for each
pathology class.  If merging would create a bbox exceeding typical dimensions,
reject the merge.

| Class | Typical diameter (% of FOV) | Max merge size |
|---|---|---|
| NP polyp | 10–30% | 35% |
| SCC mass | 15–40% | 50% |
| IP papilloma | 5–15% | 20% |

**Pros:** Prevents anatomically implausible merges  
**Cons:** Requires class label (not available during anomaly detection — this
is a class-agnostic step); limits vary by patient/scope/zoom  
**Complexity:** ~10 lines + lookup table, but requires class awareness

---

### 3.10 Graph-Based Clustering

**Idea:** Build a graph where each connected component is a node.  Weight edges
by the anomaly signal in the gap between them.  Apply graph cut (min-cut /
spectral clustering) to decide which merges to keep.

**Pros:** Globally optimal merge decisions; handles chains (A–B–C where A–B
should merge but not B–C)  
**Cons:** Overkill for typical 2–3 bbox scenarios; complex implementation  
**Complexity:** ~50 lines + graph library dependency

---

## 4. Recommendation Matrix

| Solution | Difficulty | Effectiveness | When to implement |
|---|---|---|---|
| **3.1 Valley Analysis** | Low | High | **MVP (v7)** |
| **3.2 Area Ratio Constraint** | Trivial | Medium | **MVP (v7)** |
| 3.3 Aspect Ratio Constraint | Trivial | Low | MVP (quick win) |
| **3.4 Watershed Segmentation** | Medium | High | **Near-term (v8)** |
| 3.5 Peak Count Analysis | Low | Medium | Near-term |
| 3.6 Feature Embedding Similarity | Medium | Medium | Expansion |
| 3.7 Size-Adaptive Threshold | Trivial | Low-Med | MVP as enhancement |
| 3.8 Contour Continuity | Low | Medium | Near-term |
| 3.9 Clinical Size Priors | Low | Medium | Expansion (needs class info) |
| 3.10 Graph-Based Clustering | High | High | Expansion (large datasets) |

### Recommended MVP Implementation (v7)

Combine **Valley Analysis (3.1)** + **Area Ratio Constraint (3.2)** as
additional guards in `_should_merge()`:

```python
def _should_merge(a, b, gap, amap=None, max_area_ratio=1.5, valley_thresh=0.5):
    # Step 1: Original gap check (necessary condition)
    if not _edges_within_gap(a, b, gap):
        return False
    # Step 2: Area ratio guard
    if not _area_ratio_ok(a, b, max_area_ratio):
        return False
    # Step 3: Valley check (if anomaly map available)
    if amap is not None and _has_valley(amap, a, b, valley_thresh):
        return False
    return True
```

### Near-term: Watershed Replacement (v8)

Replace the entire connected-component → merge pipeline with watershed-based
segmentation.  This fundamentally solves the problem by starting from peaks
rather than merging fragments.

---

## 5. Evaluation Strategy

To validate any merge/split approach, we need ground truth for multi-lesion
images.  Approach:

1. **Manual audit**: Flag images in the current dataset where merging visually
   appears correct vs. incorrect (estimate: 5–10 images need checking)
2. **Synthetic test**: Create test cases by compositing two known single-lesion
   anomaly maps at varying distances
3. **Metrics**: For each approach, measure:
   - **Merge precision**: % of merges that were correct (same lesion)
   - **Split recall**: % of separate lesions correctly kept separate
   - **Bbox IoU with ground truth** (when available)

---

## 6. Impact on YOLO Training

Incorrect merging affects YOLO in two ways:

1. **Training labels**: Merged bboxes teach YOLO that the inter-lesion gap is
   part of the target, reducing localisation precision
2. **Evaluation metrics**: mAP calculation uses IoU thresholds — overly large
   merged boxes reduce IoU with ground truth, deflating reported performance

As the dataset scales beyond the current ~96 images/class, the fraction of
multi-focal images will increase, making this progressively more impactful.

---

*This document should be revisited when:*
- *Dataset size exceeds 200 images/class*
- *Multi-focal pathology types are added*
- *YOLO mAP plateaus and large-bbox analysis suggests merge errors*
