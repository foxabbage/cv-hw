import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import argparse
from unet import Unet
from dice import DiceLoss, MixLoss

class SegTransform:
    def __init__(self, img_size=224, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

    def __call__(self, image, mask):
        image = TF.resize(image, 256)
        mask = TF.resize(mask, 256, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            image = TF.center_crop(image, self.img_size)
            mask = TF.center_crop(mask, self.img_size)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = torch.from_numpy(np.array(mask)).long().squeeze()
        mask = (mask - 1).clamp(0, 2)
        return image, mask

class PetSegDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): 
        return len(self.subset)
    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        return self.transform(img, mask)

def run_test(model_path, batch_size=16, img_size=224, num_classes=3, loss="dice"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(in_ch=3, out_ch=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    test_base = datasets.OxfordIIITPet(
        root='./data', split='test', 
        target_types=['segmentation'], download=True
    )
    test_ds = PetSegDataset(test_base, SegTransform(img_size=img_size, is_train=False))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True)

    if loss == "dice":
        criterion = DiceLoss(num_classes)
    elif loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = MixLoss(num_classes, 0.5)

    test_loss = 0.0
    correct = 0
    total_pixels = 0
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            for cls in range(num_classes):
                pred_cls = (preds == cls)
                target_cls = (masks == cls)
                inter = (pred_cls & target_cls).sum()
                intersection[cls] += inter
                union[cls] += pred_cls.sum() + target_cls.sum() - inter

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / total_pixels
    valid_mask = union > 0
    miou = intersection[valid_mask].div(union[valid_mask] + 1e-8).mean().item() if valid_mask.any() else 0.0
    class_ious = [intersection[i].item() / (union[i].item() + 1e-8) for i in range(num_classes)]

    print(f"Loss             : {avg_loss:.4f}")
    print(f"Pixel Accuracy   : {accuracy:.4f}")
    print(f"Mean IoU (mIoU)  : {miou:.4f}")
    print("-" * 40)
    print("Per-Class IoU:")
    for i, iou in enumerate(class_ious):
        class_name = ["Pixel belonging to the pet.", "Pixel bordering the pet.", "Background"][i]
        print(f"  Class {i} ({class_name:10s}) : {iou:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Pet Segmentation Test Script")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained .pth model file")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for inference")
    parser.add_argument("--img_size", type=int, default=224, 
                        help="Input image size")
    parser.add_argument("--loss", type=str, default="dice", 
                        help="Loss function")
    
    args = parser.parse_args()
    run_test(args.model_path, args.batch_size, args.img_size, loss=args.loss)