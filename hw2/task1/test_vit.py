import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CONFIG = {
    "model_path": "./models/vit/best.pth",
    "data_root": "./data",
    "batch_size": 32,
    "num_classes": 37,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "plot_confusion": True,
}

def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

def get_test_loader(batch_size, data_root):
    transform = get_test_transform()
    
    test_dataset = datasets.OxfordIIITPet(
        root=data_root,
        split='test',
        transform=transform,
        download=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    return test_loader, test_dataset

def load_model(model_path, num_classes, device):
    model = timm.create_model(
        "vit_tiny_patch16_224", 
        pretrained=False,
        num_classes=num_classes
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate(model, test_loader, device, num_classes):
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        preds = outputs.argmax(dim=1)
        probs = torch.softmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
    
    all_probs = np.vstack(all_probs)
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    results = {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "total_samples": total_samples
    }
    return results

def plot_confusion_matrix(labels, preds, num_classes):
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./result_images/vit.jpg", dpi=100, bbox_inches='tight')
    plt.close()


def print_classification_report(labels, preds, num_classes):
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(labels, preds, 
                               target_names=[f"class_{i}" for i in range(num_classes)],
                               digits=4))

def main():
    c = CONFIG
    test_loader, test_dataset = get_test_loader(c['batch_size'], c['data_root'])
    print(f"Test set size: {len(test_dataset)} samples")
    model = load_model(c['model_path'], c['num_classes'], c['device'])
    results = evaluate(model, test_loader, c['device'], c['num_classes'])
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    print_classification_report(results['labels'], results['predictions'], c['num_classes'])
    if c['plot_confusion']:
        plot_confusion_matrix(
            results['labels'], 
            results['predictions'], 
            c['num_classes']
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results['accuracy']

if __name__ == "__main__":
    main()