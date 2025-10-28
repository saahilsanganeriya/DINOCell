# DINOCell Quick Start

Get started with DINOCell in 5 minutes!

## Install (1 minute)

```bash
# Clone and setup
git clone <repository-url>
cd DINOCell
pip install -r requirements.txt
pip install -e .

# Verify DINOv3 is available
cd ..
git clone https://github.com/facebookresearch/dinov3.git
cd DINOCell
```

## Run Example (2 minutes)

```python
import cv2
from dinocell import create_dinocell_model, DINOCellPipeline

# Load image
image = cv2.imread('your_cells.png', cv2.IMREAD_GRAYSCALE)

# Create model
model = create_dinocell_model('small', freeze_backbone=True)
pipeline = DINOCellPipeline(model, device='cuda')

# Segment!
labels = pipeline.run(image)
print(f"Found {len(np.unique(labels))-1} cells!")
```

## Train Your Model (2 hours)

```bash
# 1. Download LIVECell dataset (~5GB)
wget http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021.zip
unzip LIVECell_dataset_2021.zip

# 2. Process dataset
cd dataset_processing
python process_dataset.py livecell \
    --input ../LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train \
    --split train

# 3. Train model
cd ../training
python train.py \
    --dataset ../datasets/LIVECell-train \
    --model-size small \
    --freeze-backbone \
    --epochs 100

# Done! Model saved to: checkpoints/dinocell/best_model.pt
```

## Next Steps

- 📖 Read `GETTING_STARTED.md` for detailed setup
- 🎓 Check `TRAINING_GUIDE.md` for training tips
- 📊 See `DATASETS.md` for dataset information
- 📓 Try `Tutorial.ipynb` for interactive examples

## Quick Commands

```bash
# Segment image
python -m dinocell.cli segment image.png --model model.pt --output results/

# Train model
python training/train.py --dataset datasets/LIVECell-train --model-size small

# Evaluate model
python evaluation/evaluate.py --model model.pt --dataset datasets/PBL_HEK
```

## Help

```bash
# Get help for any script
python training/train.py --help
python evaluation/evaluate.py --help
python dataset_processing/process_dataset.py --help
```

That's it! You're ready to use DINOCell! 🎉



