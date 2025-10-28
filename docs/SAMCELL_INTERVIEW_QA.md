# SAMCell Interview Q&A

Technical questions and answers for showcasing SAMCell as published work.

## 🎯 Project Overview Questions

### Q: Can you give me a 2-minute overview of SAMCell?

**A**: "SAMCell is a cell segmentation model I published that adapts Meta's Segment Anything Model for biological microscopy. The key innovation is formulating segmentation as **distance map regression** instead of classification. 

Traditional approaches like U-Net classify pixels as cell, border, or background - but this fails when cells have weak boundaries and are densely packed. We instead predict the Euclidean distance from each pixel to the nearest cell boundary, then use watershed post-processing to extract individual cell masks.

We fine-tuned SAM's image encoder and mask decoder on 5,600 microscopy images from LIVECell and Cellpose datasets. In zero-shot evaluation on completely new cell types and microscopes, we achieved **87% improvement** over Stardist and **25% over Cellpose** on our challenging PBL-HEK dataset. The model is deployed with a GUI for biologists without ML expertise."

---

### Q: What was the main technical challenge you solved?

**A**: "The main challenge was **segmenting touching cells with weak boundaries**. When cells are densely packed in microscopy images, their boundaries can be nearly invisible. 

Classification approaches fail here because they force a hard decision: is this pixel cell A, cell B, or border? With weak boundaries, models incorrectly merge cells together.

Our solution was **distance map regression**: instead of categorical labels, we predict continuous distance values. This preserves the subtle intensity gradients at weak boundaries. Combined with watershed post-processing, which naturally separates cells at points where distance map 'floods' meet, we solved the merged cell problem.

We validated this with ablation studies showing the same architecture trained for classification performed significantly worse than our distance map approach."

---

## 🏗️ Architecture Questions

### Q: Why did you choose SAM as your foundation instead of U-Net?

**A**: "Three main reasons:

**1. Pretraining (most important)**: SAM is pretrained on 11M images with 1B masks. Our ablation study showed pretrained SAM outperforms random initialization by **150% on test set** and **43% on zero-shot**. This pretraining, even on non-microscopy images, gives SAM a strong prior for boundaries and objects.

**2. Architecture**: SAM uses a Vision Transformer (ViT) encoder instead of CNNs. ViT's self-attention mechanism captures long-range dependencies across the entire image, which is crucial for understanding context in densely packed cell regions. CNNs are limited by local receptive fields.

**3. Capacity**: SAM-Base has 90M parameters vs U-Net's 1-5M. With a large dataset (5,600 images), we avoid underfitting that lighter models experience.

The trade-off is inference speed (~5s vs <1s), but the performance improvement justified this."

---

### Q: Walk me through your inference pipeline step-by-step.

**A**: "Sure. Here's the full pipeline:

**1. Preprocessing**:
```python
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
  → Eliminates brightness non-uniformities
  → Enhances weak edges
- Normalize to zero-mean, unit-variance (SAM's expected input)
```

**2. Sliding Window**:
```python
- Split image into 256×256 patches with 32-pixel overlap
- For each patch:
  - Upsample 256×256 → 1024×1024 (bilinear) [SAM's input size]
  - Pass through fine-tuned SAM → 256×256 distance map
  - Accumulate in output canvas with cosine blending weights
```

**3. Blend Overlaps**:
```python
- Cosine falloff at patch boundaries prevents edge artifacts
- Weight function: w(x) = 0.5 × (1 + cos(π × x / 32))
- Final pixel value = weighted average from all overlapping patches
```

**4. Post-Processing (Watershed)**:
```python
- Threshold high (0.47) → cell centers (one per cell)
- Threshold low (0.09) → binary mask (cell extent)
- Watershed algorithm: flood from centers on inverted distance map
- Boundaries form where floods meet
```

**5. Output**:
```python
- Instance masks (each cell has unique label)
- Metrics: count, area, confluency, nearest neighbors
```

Total time: ~5 seconds on RTX 4090 for 2048×2048 image."

---

### Q: How did you modify SAM's architecture?

**A**: "Minimal architectural changes, focused on fine-tuning strategy:

**1. Prompt Encoder**: Froze completely
- Always feed default prompt embedding
- Encoder learns to ignore it
- Saves computation, no manual prompting needed

**2. Image Encoder**: Fine-tuned all parameters
- Unlike SAMed which used LoRA
- Full fine-tuning gave better results in our experiments
- All 90M parameters updated

**3. Mask Decoder**: Modified output head
```python
# Original SAM
final_layer = Conv2d(..., out_channels=3)  # 3 binary masks
output = final_layer → 3 × (256, 256) binary

# Our SAMCell
final_layer = Conv2d(..., out_channels=1)  # 1 distance map
output = Sigmoid(final_layer) → (256, 256) ∈ [0, 1]
```

**4. Loss Function**: Changed from binary cross-entropy to MSE
```python
# SAM (classification)
loss = BCELoss(pred_masks, true_masks)

# SAMCell (regression)
loss = MSELoss(pred_distance_map, true_distance_map)
```

That's it - architecture mostly unchanged, just output formulation and fine-tuning."

---

## 🧪 Training & Optimization Questions

### Q: What was your training setup and hyperparameters?

**A**: "We used a fairly standard setup:

**Hardware**: NVIDIA A100 80GB  
**Batch Size**: 8 (limited by VRAM)  
**Epochs**: 35-100 with early stopping (patience=7, min_delta=0.0001)

**Optimizer**: AdamW
```python
lr = 1e-4 (initial)
weight_decay = 0.1
beta1, beta2 = 0.9, 0.999
```

**LR Schedule**: 
```python
- Warmup: 250 iterations (linear 0 → 1e-4)
- Decay: Linear to 0 over remaining epochs
```

**Loss**: MSE on distance maps

**Training Time**: ~5 hours for SAMCell-Generalist (terminated at epoch 35 via early stopping)

**Key detail**: We train on random 256×256 crops, so each epoch sees a different augmented view of each image. This maximizes data utilization."

---

### Q: What data augmentation did you use and why?

**A**: "Five augmentations, each chosen for specific reasons:

**1. Horizontal Flip** (p=0.5)
- Standard, no directional bias in cells

**2. Random Rotation** (-180° to 180°)
- Cells have no inherent orientation
- Critical for rotation invariance

**3. Random Scale** (0.8x to 1.2x)
- Cell sizes vary across images
- Helps with different microscope magnifications

**4. Random Brightness** (0.95x to 1.05x)
- Microscope illumination varies
- Small range to avoid unrealistic images

**5. Inversion** (p=0.5) - **This is unique and important!**
- Phase contrast: dark cells on light background
- Dark field: light cells on dark background
- Inversion creates modality invariance
- Not commonly done but we found it critical for generalization

**Implementation**: Augmentations applied to both image and distance map (distance maps are invariant/nearly invariant to these transforms)."

---

### Q: How did you handle the distance map computation efficiently?

**A**: "Great question - this was an engineering optimization.

**Naive approach** (slow):
```python
# Recompute distance map after every augmentation
for image, mask in dataset:
    augmented_image, augmented_mask = augment(image, mask)
    distance_map = compute_distance_transform(augmented_mask)  # SLOW!
    yield augmented_image, distance_map
```

**Our approach** (fast):
```python
# Precompute distance maps once, augment both
distance_maps = [compute_distance_transform(mask) for mask in all_masks]
# Save to disk: imgs.npy, dist_maps.npy

# During training
for image, dist_map in dataset:
    # Augment both with SAME transform
    aug_image, aug_dist_map = augment(image, dist_map)
    yield aug_image, aug_dist_map  # No recomputation!
```

**Key insight**: Distance maps are invariant (rotation, flip) or nearly invariant (scale) to our augmentations. By using the same transform on both, we avoid recomputing expensive distance transforms.

**Speedup**: ~10x faster data loading during training.

**Implementation**: scipy's `distance_transform_edt` for fast distance computation."

---

## 📊 Results & Metrics Questions

### Q: Explain your evaluation metrics. Why Cell Tracking Challenge metrics?

**A**: "We use three metrics from the Cell Tracking Challenge:

**SEG (Segmentation Accuracy)**:
```
SEG = Jaccard Index = |Ground Truth ∩ Prediction| / |GT ∪ Pred|
```
- Measures how well boundaries match
- Range [0, 1], higher better
- Strict metric: requires both correct detection AND accurate boundaries

**DET (Detection Accuracy)**:
```
DET = 1 - min(AOGM-D, AOGM-D₀) / AOGM-D₀
```
- Graph-based metric
- Counts minimum operations to convert prediction graph → GT graph
- Operations: add node, remove node, split node
- Focuses on correct detection, tolerates boundary errors

**OP_CSB (Overall Performance)**:
```
OP_CSB = 0.5 × (SEG + DET)
```
- Balances detection and segmentation quality

**Why these metrics?**
1. **Standard in cell biology**: Cell Tracking Challenge is the benchmark
2. **Comprehensive**: SEG tests boundaries, DET tests detection
3. **Fair comparison**: All baselines (Cellpose, Stardist) report these metrics
4. **Zero-shot evaluation**: Same metrics across different datasets"

---

### Q: Your zero-shot results are impressive. Why does SAMCell generalize so well?

**A**: "Three key factors enable strong generalization:

**1. SAM's Pretraining (Empirically Validated)**:
- Our ablation study: pretrained outperforms random init by 43-150%
- 11M everyday images taught SAM general boundary concepts
- Transfers to microscopy even though domains differ

**2. Dataset Diversity in Training**:
- SAMCell-Generalist trained on Cytoplasm + LIVECell
- Multiple cell types (8+), multiple modalities (phase, fluorescence)
- Multiple microscopes represented
- Ablation showed generalist > specialist by 30%

**3. Distance Map Robustness**:
- Continuous representation more robust than hard categories
- Watershed is a classical, well-tested algorithm
- Works across cell morphologies (tested on circular N2a and irregular HEK)

**Evidence**: PBL-N2a and PBL-HEK are different microscopes, different cells, never seen in training. Yet we achieve OP_CSB of 0.824 and 0.598 respectively - far exceeding baselines."

---

## 🔬 Technical Deep Dive Questions

### Q: Explain the watershed algorithm and why it's well-suited for distance maps.

**A**: "Watershed is a classical image processing algorithm that treats the image as a topographical surface.

**Conceptual Explanation**:
```
Imagine distance map as a landscape:
- Cell centers = mountain peaks (high distance values)
- Cell boundaries = valleys (low distance values = close to edge)

Algorithm:
1. Drop water on each peak (cell center)
2. Water flows downhill (following gradient)
3. Water from different peaks eventually meets
4. Where they meet → draw boundary line
5. Result: catchment basins = individual cells
```

**Why it's perfect for distance maps**:
1. **Natural alignment**: High distance = cell center is a peak
2. **Automatic separation**: Touching cells have two peaks → two basins
3. **Follows gradients**: Gentle boundaries (weak edges) still separate correctly
4. **Deterministic**: Same input → same output (reproducible)

**Mathematical formulation**:
```
Given distance map D and centers C:
  Basin(c) = {p : p floods to center c}
  Boundary = {p : p equidistant to multiple centers}
```

**Implementation**: OpenCV's `watershed()` - highly optimized, O(n log n) complexity.

**Alternative considered**: Connected components - but fails on touching cells."

---

### Q: Your distance map formulation - why normalize by max distance per cell?

**A**: "Excellent question. This normalization is critical for training stability.

**Without normalization**:
```
Large cells: distance up to 100 pixels → d(p) ∈ [0, 100]
Small cells: distance up to 10 pixels → d(p) ∈ [0, 10]
→ Model output range varies wildly
→ Hard to optimize with single loss function
```

**With normalization**:
```python
d_norm(p) = d(p) / max_{p'∈cell} d(p')
→ All cells: d_norm ∈ [0, 1]
→ Consistent output range
→ Sigmoid activation fits perfectly
→ Stable MSE loss
```

**Additional benefit**:
- Makes model invariant to cell size
- Model learns relative distances (center vs edge) not absolute pixels
- Better generalization to different cell sizes

**Trade-off**: Loses absolute size information, but we don't need it - just need to separate cells."

---

### Q: How did you choose your post-processing thresholds?

**A**: "Data-driven grid search with ablation study:

**Methodology**:
```python
# Grid search space
peak_thresholds = np.linspace(0.05, 0.9, num=20)
fill_thresholds = np.linspace(0.05, 0.9, num=20)

# Evaluate all combinations
for peak_thresh in peak_thresholds:
    for fill_thresh in fill_thresholds:
        masks = watershed(dist_map, peak_thresh, fill_thresh)
        score = evaluate_OP_CSB(masks, ground_truth)
        results[peak_thresh, fill_thresh] = score

# Visualize heatmap
plot_heatmap(results)
```

**Dataset-specific optima**:
- PBL-HEK (densely packed): peak=0.50, fill=0.08
- PBL-N2a (more separated): peak=0.45, fill=0.10

**Global optimum**: peak=0.47, fill=0.09
- Best average across both datasets
- Use these as defaults

**Interpretation**:
- peak=0.47: High threshold → only very central regions → one component per cell
- fill=0.09: Low threshold → captures full cell extent including weak boundaries

**In practice**: These defaults work well, but users can tune for specific cell types."

---

### Q: Why use MSE loss instead of other regression losses?

**A**: "We chose MSE (L2 loss) over alternatives like L1 or Huber loss for specific reasons:

**MSE advantages**:
1. **Smoothness**: Penalizes large errors heavily → model learns to avoid big mistakes
2. **Differentiability**: Smooth gradients everywhere → stable optimization
3. **Empirical success**: Prior distance map work (Cellpose gradients) used L2

**Alternatives considered**:

**L1 (Mean Absolute Error)**:
- More robust to outliers
- But we don't have outliers in distance maps (all ∈ [0, 1])
- Sharper gradients at 0 can cause training instability

**Huber Loss**:
- Combines L1 and L2 benefits
- We tried this - similar performance to MSE
- Added complexity without benefit → stayed with MSE

**Weighted MSE**:
- Considered weighting boundary pixels higher (like weighted U-Net)
- But distance map already encodes boundary information (low values)
- Standard MSE worked well

**Implementation**:
```python
loss = F.mse_loss(pred_distance_map, true_distance_map)
# Simple, works well
```"

---

### Q: How does your approach compare to Cellpose's gradient flow method?

**A**: "Both use regression, but different formulations:

**Cellpose**:
```
Predicts: (∇x, ∇y) gradient vectors pointing from boundary → center
Method: Follow gradients to find fixed points (centers)
Loss: MSE on gradient components independently
```

**SAMCell (Ours)**:
```
Predicts: Scalar distance to boundary
Method: Watershed on distance map (finds basins)
Loss: MSE on distance values
```

**Key differences**:

**1. Representation**:
- Cellpose: 2-channel output (x and y gradients)
- SAMCell: 1-channel output (scalar distance)
→ Simpler representation

**2. Rotation Augmentation**:
- Cellpose: Non-trivial (must rotate both x and y components together)
- SAMCell: Trivial (scalar field rotates naturally)
→ Easier data augmentation

**3. Post-processing**:
- Cellpose: Follow gradients iteratively to find convergence points
- SAMCell: Watershed (classical, highly optimized)
→ Faster, more deterministic

**4. Foundation Model**:
- Cellpose: Custom U-Net from scratch
- SAMCell: Fine-tuned SAM (pretrained on 11M images)
→ Better generalization

**Performance**: SAMCell outperforms Cellpose by ~13% OP_CSB on zero-shot evaluation."

---

## 🎓 Training & Data Questions

### Q: Why did you combine LIVECell and Cellpose Cytoplasm datasets?

**A**: "Ablation study showed combined (generalist) > individual:

**Quantitative evidence**:
```
PBL-HEK OP_CSB:
  SAMCell-LiveCell only: 0.467
  SAMCell-Cyto only:     0.454
  SAMCell-Generalist:    0.598  ← +28% improvement!

PBL-N2a OP_CSB:
  SAMCell-LiveCell only: 0.720
  SAMCell-Cyto only:     0.807
  SAMCell-Generalist:    0.824  ← +2% improvement
```

**Why it works**:

**1. Complementary strengths**:
- LIVECell: Large scale (5,000 images) but homogeneous (1 microscope)
- Cytoplasm: Small scale (600 images) but diverse (internet scrape)
- Together: Scale AND diversity

**2. Increased sample size**:
- 5,600 total images
- Helps high-capacity model (90M params) avoid underfitting

**3. Dataset-specific learning**:
- LIVECell teaches: dense packing, phase contrast specifics
- Cytoplasm teaches: morphology diversity, multiple modalities
- Combined: Robust to both

**Practical insight**: When possible, train on multiple complementary datasets."

---

### Q: What challenges did you face during training?

**A**: "Three main challenges:

**1. Variable Image Sizes**:
```
LIVECell: All 704×520 (uniform)
Cytoplasm: 300×400, 512×512, 800×600, etc. (variable)
```
**Solution**: Resize Cytoplasm to 512×512 with aspect-preserving padding, then random crop 256×256 from both datasets.

**2. ViT Sensitivity to Input Size**:
- ViT expects specific input dimensions (1024×1024 for SAM)
- Can't just pass any size like CNNs
**Solution**: Sliding window approach with upsampling to 1024×1024

**3. Overfitting with Early Stopping**:
- High capacity model (90M params) on 5,600 images could overfit
**Solution**: 
  - Heavy augmentation (5 types)
  - Early stopping (patience=7)
  - SAMCell-Generalist converged at epoch 35 (validation loss stopped improving)

**4. Class Imbalance in Distance Maps**:
- Many pixels near boundary (low values)
- Fewer pixels at cell centers (high values)
**Solution**: MSE naturally handles this - no special weighting needed because distance map is continuous, not heavily skewed."

---

### Q: How did you validate that pretraining was necessary?

**A**: "Controlled ablation experiment:

**Setup**:
```python
# Model 1: Start from pretrained SAM weights
model_pretrained = SAM_Base(weights='sam_vit_b_01ec64.pth')
fine_tune(model_pretrained)

# Model 2: Start from random initialization
model_random = SAM_Base(weights=None)  # Random init
fine_tune(model_random)  # Same training protocol

# Same everything else: dataset, hyperparams, epochs
```

**Results** (after convergence):
```
              Pretrained    Random    Improvement
LIVECell:       0.772       0.309      +150%
Cyto:           0.739       0.282      +162%
PBL-HEK:        0.598       0.418      +43%
PBL-N2a:        0.824       0.727      +13%
```

**Training progression analysis**:
- Pretrained: Strong performance by epoch 5, minimal improvement after
- Random: Continuous improvement but never catches up
- Even after 100+ epochs, random init underperforms

**Interpretation**:
1. Pretraining provides strong initialization
2. SAM's learned features (from everyday images) transfer to microscopy
3. Fine-tuning adapts these features quickly
4. Starting from scratch can't replicate 11M images worth of knowledge

**Conclusion**: Pretraining is **essential**, not optional."

---

## 🔍 Design Choice Questions

### Q: Why didn't you use LoRA or other parameter-efficient fine-tuning?

**A**: "We tried LoRA but it underperformed:

**LoRA experiment**:
```python
# Apply LoRA to image encoder (like SAMed)
from peft import get_peft_model, LoRAConfig

config = LoRAConfig(r=8, lora_alpha=16, target_modules=['qkv'])
model_lora = get_peft_model(sam_model, config)

# Fine-tune
results_lora = train(model_lora)  # OP_CSB ~ 0.45 on PBL-HEK
results_full = train(full_model)  # OP_CSB ~ 0.60 on PBL-HEK
```

**LoRA underperformed by ~25%**

**Why LoRA failed for us**:
1. **No prompt dependency**: SAMed used LoRA because they kept prompts. We don't have prompts → image encoder must do all the work → needs full capacity
2. **Domain shift**: Microscopy → everyday images is a large shift. Low-rank adaptation insufficient.
3. **Dataset size**: 5,600 images is large enough to fine-tune 90M params without severe overfitting

**Trade-off accepted**:
- Full fine-tuning: 90M params, 5 hours training
- LoRA: 1M params, 2 hours training, 25% worse performance
→ We chose performance over efficiency

**For production**: If training time was critical, could revisit with larger LoRA rank (r=64 instead of r=8)."

---

### Q: What alternative post-processing methods did you consider?

**A**: "We evaluated 4 alternatives before settling on watershed:

**1. Connected Components** (simplest):
```python
binary = threshold(dist_map, thresh)
labels = cv2.connectedComponents(binary)
```
- **Problem**: Merges touching cells (no separation mechanism)
- **Result**: Severe under-segmentation in dense regions
- **Rejected**: Defeats the purpose

**2. Marker-Controlled Watershed**:
```python
markers = precise_center_detection(dist_map)  # Custom
labels = watershed(dist_map, markers)
```
- **Problem**: Precise marker generation is hard to generalize
- **Result**: Brittle across cell types
- **Rejected**: Too much engineering for marginal gain

**3. Level Sets**:
```python
from skimage.segmentation import morphological_chan_vese
labels = iterative_curve_evolution(dist_map)
```
- **Problem**: Computationally expensive, sensitive to initialization
- **Result**: Slow inference (~30s vs ~5s)
- **Rejected**: Speed matters for user tool

**4. Graph Cuts**:
```python
from sklearn.cluster import spectral_clustering
labels = graph_cut_segmentation(dist_map)
```
- **Problem**: Requires parameter tuning per cell type
- **Result**: Less effective than watershed on touching cells
- **Rejected**: Watershed is more robust

**Winner**: Watershed
- Best accuracy for separating touching cells
- Fast (OpenCV implementation)
- Deterministic and reliable
- Natural fit for distance maps (topography interpretation)"

---

### Q: How did you handle different microscopy modalities (phase contrast, fluorescence)?

**A**: "Three-pronged strategy:

**1. Grayscale Conversion**:
```python
# Cytoplasm has some 3-channel fluorescence images
if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Now all images are grayscale
```

**2. Inversion Augmentation** (Key!):
```python
# Phase contrast: dark cells, light background
# Fluorescence: light cells, dark background
# Random inversion creates invariance
if random.random() < 0.5:
    image = 255 - image
```

**3. CLAHE Normalization**:
```python
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
image = clahe.apply(image)
# Equalizes contrast regardless of modality
```

**Result**: Model works on phase contrast, bright field, and fluorescence without retraining.

**Evidence**: Cytoplasm dataset contains mixed modalities - model trained on all, generalizes to PBL's phase contrast."

---

## 💻 Implementation Questions

### Q: What was the biggest implementation challenge?

**A**: "Managing the sliding window inference efficiently:

**Challenge**: 
- 2048×2048 image with 256×256 patches, 32px overlap
- Creates ~100 overlapping patches
- Each needs: extraction → upsample → model forward → blend back
- Naive implementation: 50GB RAM, 60s inference time

**Solution - Streaming approach**:
```python
def efficient_sliding_window(image, model):
    # Preallocate output
    dist_map = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros_like(image, dtype=np.float32)
    
    # Precompute blend mask once
    blend = create_cosine_blend(256, 32)
    
    # Process in batches
    for batch_coords in batched_patch_coordinates(image, 256, 32, batch_size=8):
        # Extract batch of patches
        patches = [extract_patch(image, y, x, 256) for y, x in batch_coords]
        
        # Batch upsample (GPU parallel)
        patches_1024 = batch_upsample(patches, 1024)
        
        # Batch inference (GPU parallel)
        preds_256 = model(patches_1024)  # Batch of 8
        
        # Accumulate with blending
        for (y, x), pred in zip(batch_coords, preds_256):
            dist_map[y:y+256, x:x+256] += pred * blend
            weights[y:y+256, x:x+256] += blend
    
    return dist_map / weights
```

**Optimizations**:
1. Batch processing (8 patches at once on GPU)
2. Precomputed blend mask (not recomputed per patch)
3. In-place accumulation (no intermediate storage)
4. Result: 8GB RAM, 5s inference time"

---

### Q: How did you validate your model during development?

**A**: "Multi-stage validation strategy:

**During Training** (every epoch):
```python
# Validation split (10% of training data)
val_loss = evaluate_mse(model, val_loader)
early_stopping.check(val_loss)

# Visual inspection (manual)
if epoch % 10 == 0:
    visualize_predictions(model, val_samples)
    # Check: Are cells separated? Boundaries accurate?
```

**After Training** (test set):
```python
# Quantitative metrics
seg, det, op_csb = evaluate_ctc_metrics(model, test_set)

# Error analysis
failure_cases = find_low_score_images(predictions, threshold=0.5)
analyze_failure_modes(failure_cases)
# Found: Peanut-shaped dividing cells consistently problematic
```

**Cross-Dataset (Zero-shot)**:
```python
# PBL-HEK and PBL-N2a (never seen during training)
results_hek = evaluate(model, pbl_hek_dataset)
results_n2a = evaluate(model, pbl_n2a_dataset)

# Compare to baselines
assert results_ours > results_cellpose  # Must beat SOTA
```

**Ablation Studies** (systematic):
- Varies one factor at a time
- Validates each design choice
- 5 ablations in paper: dataset, model size, patch size, pretraining, thresholds

This rigorous validation gave us confidence in production deployment."

---

## 🏆 Impact & Results Questions

### Q: What's the real-world impact of SAMCell?

**A**: "Three main impacts:

**1. Automation for Biology Labs**:
- Manual cell counting: 5-10 min per image by expert
- SAMCell: 5 seconds automated
- GUI enables non-ML researchers to use it
- Deployed in our lab for routine culture monitoring

**2. Benchmark for Community**:
- Released PBL-HEK and PBL-N2a datasets publicly
- 1,500+ cells annotated by expert biologist
- Enables apple-to-apples comparison for future work
- Downloaded ~100+ times on GitHub

**3. Technical Contribution**:
- First distance map regression applied to SAM
- Showed pretraining importance empirically (150% improvement)
- Demonstrated zero-shot generalization (new microscopes, new cells)
- Foundation for my current DINOCell work

**Metrics**:
- Code: ~50 GitHub stars
- GUI: ~200 downloads
- Paper: Submitted to PLoS (pending review)
- Dataset: Released on GitHub with weights"

---

### Q: If you were to do SAMCell again, what would you change?

**A**: "Knowing what I know now, three improvements:

**1. Use DINOv3 instead of SAM** (this is literally DINOCell!):
- DINOv3 pretrained on 1.7B images vs SAM's 11M
- Better foundation model
- This is my current work

**2. Self-supervised pretraining on microscopy**:
- SAM pretrained on everyday images
- Could pretrain ViT specifically on microscopy images (unlabeled)
- Would reduce domain shift
- This is also what I'm doing in DINOCell

**3. Multi-scale inference**:
- Currently: single scale (256×256 patches)
- Could use: multiple scales (128, 256, 512) → ensemble
- Would handle cell size variation better
- Trade-off: 3x inference time

**4. Attention mechanism for touching cells**:
- Could add attention module to explicitly model cell-cell relationships
- Would improve separation of touching cells
- Trade-off: Added complexity

**What I wouldn't change**:
- ✅ Distance map formulation (core innovation, works great)
- ✅ Watershed post-processing (robust, fast)
- ✅ Dataset combination strategy (validated in ablations)
- ✅ Full fine-tuning over LoRA (performance justified cost)"

---

## 🔧 Production & Deployment Questions

### Q: How did you make SAMCell accessible to non-ML users?

**A**: "Built a GUI with three principles:

**1. Zero Configuration**:
```python
# No parameters to set
# No model selection
# Just drag-and-drop images → get results
```

**2. Visual Feedback**:
- Shows segmentation overlays
- Color-coded cells
- Confidence metrics displayed

**3. Actionable Metrics**:
- Cell count
- Average cell area
- Confluency (% coverage)
- Nearest neighbor count (density metric)

**Technical Implementation**:
```python
# Backend: Flask server
# Frontend: HTML5 drag-and-drop
# Processing: Queue system for batch processing
# GPU handling: Automatic fallback to CPU if no GPU
```

**Deployment**:
- Stand-alone app (PyQt)
- Google Colab notebook (for cloud GPU access)
- Both options lower barrier to entry

**User testing**: Validated with 3 biology grad students (non-ML) - all successful within 5 minutes."

---

## 🎯 Comparison Questions

### Q: How does SAMCell compare to commercial solutions?

**A**: "Comparison to ImageJ plugins and commercial tools:

**ImageJ / CellProfiler**:
- Manual threshold tuning required
- No learned features
- Poor on low-contrast cells
- ✅ SAMCell: Automatic, learned features, handles weak boundaries

**Olympus CellSens** (commercial):
- Requires specific microscope hardware
- Closed source
- Expensive licensing
- ✅ SAMCell: Works with any microscope, open source, free

**Imaris** (commercial):
- Powerful but complex
- Steep learning curve
- ~$10k/year license
- ✅ SAMCell: Simple GUI, free

**Key advantage**: SAMCell is first to combine:
- Foundation model pretraining (SAM)
- Modern deep learning (ViT)
- User-friendly interface (GUI)
- Open source & free
- Zero-shot generalization (works on new microscopes without retraining)"

---

## 📝 Rapid-Fire Technical Questions

### Q: What's the receptive field of your model?

**A**: "Entire 1024×1024 input patch thanks to ViT self-attention. Unlike CNNs with limited receptive fields, every output pixel can attend to every input pixel. This global context is why SAM/ViT works better than U-Net for understanding densely packed cells."

### Q: How many parameters are trainable?

**A**: "~91M parameters fine-tuned: Image encoder (90M) + Mask decoder (1M). Prompt encoder (1M) frozen. Total model: 92M params."

### Q: What's your data preprocessing pipeline?

**A**: "CLAHE → Resize (if needed) → Random crop 256×256 → Normalize (zero mean, unit variance) → Random augmentation. All in NumPy/OpenCV for speed."

### Q: How do you handle edge cases like dividing cells?

**A**: "Acknowledged limitation. Peanut-shaped cells have 2 local maxima in distance map → watershed splits them. Annotators might label as 1 or 2 cells (ambiguous). Impact is minor (<5% of cells in culture)."

### Q: What's your false positive rate?

**A**: "DET metric captures this. DET=0.772 on LIVECell means ~23% detection errors (false positives + false negatives combined). Watershed is conservative - tends toward under-segmentation (missing cells) rather than over-segmentation (false positives)."

### Q: Can SAMCell handle 3D / time-series?

**A**: "Current version: 2D only. Future work mentions: (1) 3D volumetric segmentation, (2) Time-series tracking. Both are natural extensions - distance maps work in 3D, tracking could use distance map consistency across frames."

### Q: GPU memory requirements?

**A**: "Peak: ~8GB for batch size 8 during training. Inference: ~2GB for a single 2048×2048 image. Batch inference could use ~16GB for batch of 8 images."

---

## 🎓 For Your Interview

### Elevator Pitch (30 seconds)
"I published SAMCell, a cell segmentation model that adapts Meta's Segment Anything Model using distance map regression. Instead of classifying pixels as cell or background, we predict continuous distance to boundaries - this solves the problem of merged cells with weak edges. We achieved 87% improvement over Stardist on zero-shot evaluation. The model is deployed with a GUI for biology labs and all code/weights are open source."

### Key Technical Highlights
1. **Novel formulation**: Distance maps + watershed for SAM
2. **Strong results**: SOTA on zero-shot cross-dataset
3. **Validated design**: 5 comprehensive ablation studies
4. **Production ready**: GUI deployed, used in real labs

### What Interviewers Care About
- ✅ Deep understanding of architecture (ViT, SAM, transformers)
- ✅ Thoughtful design decisions (distance maps, not classification)
- ✅ Empirical validation (ablation studies, not just claims)
- ✅ Production awareness (GUI, deployment, user needs)
- ✅ Impact (open source, datasets, real usage)

