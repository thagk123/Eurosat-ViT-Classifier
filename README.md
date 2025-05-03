# EuroSAT Classification with Vision Transformer
This project fine-tunes a Vision Transformer (ViT) model to classify satellite images from the EuroSAT dataset.

## Dataset
- EuroSAT contains satellite images from the Sentinel-2 mission.
- It includes 10 land use and land cover classes (e.g. Forest, Residential, River).
- Loaded using `torchvision.datasets.EuroSAT`.

## Model
- Pretrained Vision Transformer: `google/vit-base-patch16-224-in21k`
- Fine-tuned for 3 epochs using Hugging Face Transformers

## Setup
- Python
- PyTorch
- Hugging Face Transformers
- Torchvision
- scikit-learn

## Results
- Train/test split: 85% / 15%
- Test Accuracy: 98.79%

## Files
- `eurosat_vit.py`: Full training and testing file
