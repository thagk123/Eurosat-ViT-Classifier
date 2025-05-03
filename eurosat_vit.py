"""Fine-tuning a Vision Transformer on the EuroSAT satellite image dataset."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np

def get_image_processor():
    """Load the pretrained ViT image processor."""
    return ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def get_transform(processor):
    """Return image transform pipeline compatible with ViT."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

def load_and_split_dataset(transform):
    """Load EuroSAT dataset and split it into train and test sets (85/15)."""
    dataset = datasets.EuroSAT(root="./data", download=True, transform=transform)
    indices = list(range(len(dataset)))
    labels = [dataset[i][1] for i in indices]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    return Subset(dataset, train_idx), Subset(dataset, test_idx), dataset.classes

class EuroSATDataset(torch.utils.data.Dataset):
    """Wrapper to convert torchvision dataset to Hugging Face format."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {"pixel_values": img, "label": label}

def get_model(num_labels, label_names):
    """Load the pretrained ViT model and adapt it for classification."""
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

def get_training_args():
    """Define training arguments for the Trainer."""
    return TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="no",
        report_to="none"
    )

def get_compute_metrics():
    """Return evaluation metric function (accuracy)."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
    return compute_metrics

def main():
    """Main execution function: training and evaluation."""
    processor = get_image_processor()
    transform = get_transform(processor)
    train_ds, test_ds, label_names = load_and_split_dataset(transform)

    train_hf = EuroSATDataset(train_ds)
    test_hf = EuroSATDataset(test_ds)

    model = get_model(num_labels=len(label_names), label_names=label_names)
    training_args = get_training_args()
    compute_metrics = get_compute_metrics()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=None,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate(eval_dataset=test_hf)
    print("Test Results:")
    print(f"Accuracy: {results['eval_accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()
