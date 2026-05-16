import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import swanlab
import itertools
import copy
import os

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.OxfordIIITPet(root='./data', split='trainval',
                                          transform=train_transform, download=False)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def run_experiment(config):
    set_seed(config['seed'])
    swanlab.init(project="pet_dataset_hypertune", config=config, 
                 experiment_name=f"lr_base_{config['lr_base']}_lr_head_{config['lr_head']}_epochs_{config['epochs']}")
    c = config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model_fn = models.resnet18
    model = model_fn(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 37)
    model = model.to(device)

    # load data
    train_loader, val_loader = get_dataloaders(32)

    # optimize
    param_groups = []
    for name, param in model.named_parameters():
        lr = c['lr_head'] if 'fc' in name else c['lr_base']
        param_groups.append({'params': param, 'lr': lr})
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)

    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'], eta_min=1e-6)
    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience = 5
    trigger = 0

    # train
    for epoch in range(c['epochs']):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += imgs.size(0)
        train_acc = correct / total
        train_loss /= total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += outputs.argmax(1).eq(labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        val_loss /= val_total

        scheduler.step()

        swanlab.log({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr_base": optimizer.param_groups[0]['lr'],
            "lr_head": optimizer.param_groups[-1]['lr']
        })

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    save_name = f"./models/tuned/best_model_lr_base_{c['lr_base']}_lr_head_{c['lr_head']}_epochs_{c['epochs']}.pth"
    torch.save(best_model_state, save_name)
    print(f"Local model saved: {save_name}")

    swanlab.log({"final_best_val_acc": best_val_acc})
    swanlab.finish()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    param_grid = {
        "lr_base": [5e-5, 1e-4, 5e-4],
        "lr_head": [1e-3, 3e-3, 5e-3],
        "epochs": [30],
        "seed": [42]
    }

    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp}")
        try:
            run_experiment(exp)
        except Exception as e:
            print(f"{e}")
            if swanlab.run is not None:
                swanlab.finish()