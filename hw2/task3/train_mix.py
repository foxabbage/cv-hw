import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import swanlab
import itertools
import copy
import os
import random
import numpy as np
from unet import Unet
from dice import MixLoss

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

def get_dataloaders(batch_size):
    base_ds = datasets.OxfordIIITPet(root='./data', split='trainval', 
                                     target_types=['segmentation'], download=True)
    val_size = int(0.2 * len(base_ds))
    train_size = len(base_ds) - val_size
    train_sub, val_sub = random_split(base_ds, [train_size, val_size])

    train_ds = PetSegDataset(train_sub, SegTransform(img_size=224, is_train=True))
    val_ds = PetSegDataset(val_sub, SegTransform(img_size=224, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(config):
    set_seed(config['seed'])
    swanlab.init(project="pet_unet_mix", config=config,
                experiment_name=f"lr_{config['lr']}")
    c = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Unet(in_ch=3, out_ch=3).to(device)
    train_loader, val_loader = get_dataloaders(c.get('batch_size', 16))

    optimizer = optim.SGD(model.parameters(), lr=c['lr'], momentum=0.9, weight_decay=1e-3)
    criterion = MixLoss(3, 0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'], eta_min=1e-6)

    num_classes = 3
    best_val_miou = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 8
    trigger = 0

    for epoch in range(c['epochs']):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_intersection = torch.zeros(num_classes, device=device)
        train_union = torch.zeros(num_classes, device=device)

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += preds.eq(masks).sum().item()
            total += masks.numel()

            for cls in range(num_classes):
                pred_cls = preds == cls
                target_cls = masks == cls
                intersection = (pred_cls & target_cls).sum()
                train_intersection[cls] += intersection
                train_union[cls] += pred_cls.sum() + target_cls.sum() - intersection

        train_acc = correct / total
        train_loss /= total
        valid_mask = train_union > 0
        train_miou = train_intersection[valid_mask].div(train_union[valid_mask] + 1e-8).mean().item() if valid_mask.any() else 0.0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_intersection = torch.zeros(num_classes, device=device)
        val_union = torch.zeros(num_classes, device=device)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)

                preds = outputs.argmax(1)
                val_correct += preds.eq(masks).sum().item()
                val_total += masks.numel()

                for cls in range(num_classes):
                    pred_cls = preds == cls
                    target_cls = masks == cls
                    intersection = (pred_cls & target_cls).sum()
                    val_intersection[cls] += intersection
                    val_union[cls] += pred_cls.sum() + target_cls.sum() - intersection

        val_acc = val_correct / val_total
        val_loss /= val_total
        valid_mask = val_union > 0
        val_miou = val_intersection[valid_mask].div(val_union[valid_mask] + 1e-8).mean().item() if valid_mask.any() else 0.0

        scheduler.step()

        swanlab.log({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_miou": train_miou,
            "val_loss": val_loss, "val_acc": val_acc, "val_miou": val_miou,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Early Stopping
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_model_state = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    save_name = f"./models/mix/unet_lr_{c['lr']}.pth"
    torch.save(best_model_state, save_name)
    print(f"Local model saved: {save_name}")

    swanlab.log({"final_best_val_miou": best_val_miou})
    swanlab.finish()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    param_grid = {
        "lr": [1e-3, 1e-2],
        "epochs": [100],
        "seed": [42],
        "batch_size": [16]
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp}")
        try:
            run_experiment(exp)
        except Exception as e:
            print(f"Run failed with error: {e}")
            if swanlab.run is not None:
                swanlab.finish()