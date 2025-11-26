# Few-Shot Counting Model Finetuning Guide

## Overview

This guide covers finetuning the FAMNET few-shot object counting model on domain-specific data. The model uses a ResNet50-FPN backbone for feature extraction and a learned regressor head to predict object density maps.

## Architecture

The model consists of two components:
- **ResNet50-FPN**: Pretrained feature extractor (frozen by default during finetuning)
- **CountRegressor**: Lightweight 6-channel dense regressor predicting density maps

Feature extraction pipeline:
1. Extract multi-scale features (map3, map4) from ResNet50-FPN
2. Crop and normalize example boxes from exemplar regions
3. Convolve exemplars over the full image at multiple scales (0.9x, 1.1x)
4. Concatenate all features → 6 input channels to regressor

## Prerequisites

### Data Structure

```
data-final/
├── annotations.json           # Image metadata + exemplar boxes + dot annotations
├── Train_Test_Val.json        # Data splits
├── indt-objects-V4/           # Image directory
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── density_map_adaptive_V1/   # Precomputed ground truth density maps
    ├── image1.npy
    ├── image2.npy
    └── ...
```

### Annotation Format

**annotations.json:**
```json
{
  "image_filename.jpg": {
    "box_examples_coordinates": [
      [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],  // Exemplar box 1
      [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]   // Exemplar box 2
    ],
    "points": [[x1, y1], [x2, y2], ...]  // Ground truth dot locations
  }
}
```

**Train_Test_Val.json:**
```json
{
  "train": ["img1.jpg", "img2.jpg", ...],
  "val": ["img3.jpg", ...],
  "test": ["img4.jpg", ...]
}
```

### Density Map Generation

Ground truth density maps must be precomputed as `.npy` files. Generate using:
```python
from skimage.filters import gaussian
# Create density map from point annotations
density = create_density_map(points, sigma=15)  # Gaussian smoothing
np.save(f"density_map_adaptive_V1/{image_id.split('.')[0]}.npy", density)
```

## Usage

### Basic Finetuning

```bash
python finetune.py \
  -dp ./data-final/ \
  -anno ./data-final/annotations.json \
  -split ./data-final/Train_Test_Val.json \
  -img_dir ./data-final/indt-objects-V4 \
  -gt_dir ./data-final/density_map_adaptive_V1 \
  -o ./logsSave_finetuned \
  -ep 50
```

### With Pretrained Checkpoint

```bash
python finetune.py \
  -ckpt ./logsSave/best_model.pth \
  -dp ./data-final/ \
  -anno ./data-final/annotations.json \
  -split ./data-final/Train_Test_Val.json \
  -img_dir ./data-final/indt-objects-V4 \
  -gt_dir ./data-final/density_map_adaptive_V1 \
  -o ./logsSave_finetuned \
  -ep 100 \
  -lr 1e-5 \
  -bs 1
```

### Aggressive Finetuning (Lower Learning Rate)

For small domain datasets, use conservative learning rates:

```bash
python finetune.py \
  -ckpt ./logsSave/best_model.pth \
  -dp ./data-final/ \
  -anno ./data-final/annotations.json \
  -split ./data-final/Train_Test_Val.json \
  -img_dir ./data-final/indt-objects-V4 \
  -gt_dir ./data-final/density_map_adaptive_V1 \
  -o ./logsSave_finetuned \
  -ep 200 \
  -lr 5e-6 \
  -bs 1 \
  -wd 1e-4
```

## Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-dp, --data_path` | str | `./data-final/` | Dataset root directory |
| `-anno, --annotation_file` | str | `./data-final/annotations.json` | Annotations JSON path |
| `-split, --split_file` | str | `./data-final/Train_Test_Val.json` | Train/val/test splits JSON |
| `-img_dir, --image_dir` | str | `./data-final/indt-objects-V4` | Images directory |
| `-gt_dir, --gt_dir` | str | `./data-final/density_map_adaptive_V1` | Ground truth density maps |
| `-ckpt, --checkpoint` | str | `None` | Pretrained regressor checkpoint |
| `-o, --output_dir` | str | `./logsSave` | Output directory for models/logs |
| `-ep, --epochs` | int | `100` | Number of finetuning epochs |
| `-bs, --batch_size` | int | `1` | Batch size (currently enforced as 1) |
| `-lr, --learning_rate` | float | `1e-5` | Learning rate |
| `-wd, --weight_decay` | float | `0.0` | L2 regularization |
| `-freeze_backbone` | bool | `False` | Freeze ResNet50 backbone |
| `-freeze_conv_layers` | int | `0` | Number of FPN layers to freeze |
| `-ts, --test_split` | str | `val` | Evaluation split: `train/val/test` |
| `-eval_freq, --eval_frequency` | int | `5` | Evaluate every N epochs |
| `-g, --gpu` | int | `0` | GPU device ID |
| `-seed, --seed` | int | `42` | Random seed |
| `-save_freq, --save_frequency` | int | `10` | Save checkpoint every N epochs |

## Key Implementation Details

### Feature Extraction Pipeline

The model extracts features from exemplar boxes at multiple scales:

```python
# Extract features for each image and exemplar box
features = extract_features(
    resnet50_conv,
    image.unsqueeze(0),      # (1, 3, H, W)
    boxes.unsqueeze(0),      # (1, 1, M, 5) - M exemplar boxes
    MAPS=['map3', 'map4'],   # Multi-scale features
    Scales=[0.9, 1.1]        # Exemplar scale variations
)
# Output shape: (1, 6, H', W') - concatenated multi-scale features
```

Features are computed as:
1. **map3** (8x stride): 3 feature maps (1 + 2 scales)
2. **map4** (16x stride): 3 feature maps (1 + 2 scales)
- Total: 6 channels input to regressor

### Loss Function

MSE loss on density map prediction:
$$\mathcal{L} = \text{MSE}(\text{output}, \text{gt\_density})$$

Handles spatial misalignment via bilinear interpolation when output size doesn't match ground truth (common with 8-pixel stride).

### Learning Rate Scheduling

Uses `ReduceLROnPlateau`:
- Monitors validation MAE
- Reduces LR by 0.5x when no improvement for 5 epochs
- Minimum LR: 1e-7

## Output Structure

```
logsSave_finetuned/
├── checkpoints/
│   ├── best_model.pth              # Best validation performance
│   ├── checkpoint_epoch_10.pth     # Periodic checkpoints
│   └── checkpoint_epoch_20.pth
└── stats/
    └── finetune_stats.txt          # Epoch-wise metrics
```

### Stats Format

```
epoch,train_loss,train_mae,train_rmse,val_mae,val_rmse
1,0.0245,15.32,18.45,14.82,17.23
2,0.0198,12.15,14.67,11.95,13.98
...
```

## Performance Metrics

- **MAE** (Mean Absolute Error): Average count error in objects
- **RMSE** (Root Mean Squared Error): Sensitivity to large errors
- **Train Loss**: MSE on training density maps

## Finetuning Strategies

### Strategy 1: Conservative (Recommended for Small Datasets)

```bash
python finetune.py \
  -ckpt pretrained.pth \
  -dp ./data-final/ \
  -anno ./data-final/annotations.json \
  -split ./data-final/Train_Test_Val.json \
  -img_dir ./data-final/indt-objects-V4 \
  -gt_dir ./data-final/density_map_adaptive_V1 \
  -ep 150 \
  -lr 5e-6 \
  -wd 1e-4 \
  -eval_freq 5 \
  -save_freq 20
```

### Strategy 2: Moderate (Balanced Performance)

```bash
python finetune.py \
  -ckpt pretrained.pth \
  -dp ./data-final/ \
  -anno ./data-final/annotations.json \
  -split ./data-final/Train_Test_Val.json \
  -img_dir ./data-final/indt-objects-V4 \
  -gt_dir ./data-final/density_map_adaptive_V1 \
  -ep 100 \
  -lr 1e-5 \
  -wd 5e-5 \
  -eval_freq 3
```

### Strategy 3: Aggressive (Large Domain Dataset)

```bash
python finetune.py \
  -ckpt pretrained.pth \
  -dp ./data-final/ \
  -anno ./data-final/annotations.json \
  -split ./data-final/Train_Test_Val.json \
  -img_dir ./data-final/indt-objects-V4 \
  -gt_dir ./data-final/density_map_adaptive_V1 \
  -ep 80 \
  -lr 1e-4 \
  -wd 1e-4 \
  -eval_freq 2
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size (currently fixed at 1, consider code modification for larger batches)
- Reduce maximum image size in `utils.py` `MAX_HW` parameter
- Use gradient accumulation for effective larger batches

### Poor Convergence
- Check density map generation (verify sums of training counts)
- Reduce learning rate by 2-5x
- Verify data split contains sufficient training samples
- Increase weight decay for regularization

### Overfitting
- Increase weight decay (`-wd`)
- Reduce learning rate
- Stop training earlier (monitor validation MAE)
- Augment training data

### NaN Loss
- Check for invalid density maps (NaN/Inf values)
- Verify exemplar box coordinates are valid
- Reduce learning rate significantly

## Advanced: Backbone Finetuning

To finetune the ResNet50 backbone (requires careful tuning):

```python
# Modify finetune.py: remove freeze
# self.resnet50_conv.train()  # Uncomment
# Set much lower learning rate
python finetune.py \
  -ckpt pretrained.pth \
  -lr 1e-6 \  # 10x lower
  -wd 1e-3 \  # Higher regularization
  -ep 50
```

## Integration with Inference

After finetuning, use checkpoint for inference:

```python
from model import Resnet50FPN, CountRegressor
import torch

# Load finetuned model
regressor = CountRegressor(6, pool='mean')
checkpoint = torch.load('./logsSave_finetuned/checkpoints/best_model.pth')
regressor.load_state_dict(checkpoint)
regressor.eval()

# Inference (see inference.py or demo.py for full pipeline)
with torch.no_grad():
    output = regressor(features)
    count = output.sum().item()
```

## References

- Original paper: "Learning To Count Everything" (CVPR 2021)
- Feature extraction: Multi-scale convolution with exemplar normalization
- Density map supervision: Gaussian-smoothed point annotations
