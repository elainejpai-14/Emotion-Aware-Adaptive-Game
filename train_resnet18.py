# train_resnet18.py
import os, json, time, yaml, math, random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from utils import seed_everything, get_device, ensure_dir
from dataset_fer2013 import FER2013CSV  # still available for CSV mode


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def adjust_head_for_classes(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    return model


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / max(total, 1)
    return acc, all_preds, all_labels


def stratified_split_indices(targets, val_fraction=0.1, seed=42):
    targets = torch.as_tensor(targets).numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    idx = list(range(len(targets)))
    train_idx, val_idx = next(sss.split(idx, targets))
    return train_idx.tolist(), val_idx.tolist()


def main():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['seed'])
    device = get_device()
    print('Using device:', device)

    train_tf, eval_tf = build_transforms(cfg['image_size'])

    # ---------- Load data (folder or csv) ----------
    data_mode = cfg.get('data_mode', 'csv').lower()
    if data_mode == 'folder':
        train_dir = cfg['train_dir']
        test_dir  = cfg['test_dir']

        # ImageFolder expects subfolders with class names
        full_train = datasets.ImageFolder(train_dir, transform=train_tf)
        test_ds    = datasets.ImageFolder(test_dir,  transform=eval_tf)

        # Build class names from folder order (class_to_idx is alphabetic)
        idx2name = {v: k for k, v in full_train.class_to_idx.items()}
        class_names = [idx2name[i] for i in range(len(idx2name))]
        cfg['class_names'] = class_names
        cfg['num_classes'] = len(class_names)
        print('Classes (folder mode):', class_names)

        # Stratified split train -> train/val
        train_idx, val_idx = stratified_split_indices(full_train.targets, val_fraction=0.1, seed=cfg['seed'])
        train_ds = Subset(full_train, train_idx)
        val_ds   = Subset(datasets.ImageFolder(train_dir, transform=eval_tf), val_idx)

    else:
        # CSV mode (original plan)
        from dataset_fer2013 import FER2013CSV
        train_ds = FER2013CSV(cfg['csv_path'], usage=cfg['train_split'], transform=train_tf)
        val_ds   = FER2013CSV(cfg['csv_path'], usage=cfg['val_split'],   transform=eval_tf)
        test_ds  = FER2013CSV(cfg['csv_path'], usage=cfg['test_split'],  transform=eval_tf)
        class_names = cfg.get('class_names', ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
        cfg['class_names'] = class_names
        cfg['num_classes'] = len(class_names)

    # ---------- DataLoaders ----------
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=cfg['num_workers'], pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)

    # ---------- Model / Optim ----------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = adjust_head_for_classes(model, cfg['num_classes'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['step_lr_milestones'], gamma=cfg['step_lr_gamma'])

    ensure_dir(cfg['model_dir'])
    ensure_dir(cfg['run_dir'])
    best_val_acc = 0.0
    best_path = os.path.join(cfg['model_dir'], 'resnet18_fer2013_best.pth')

    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        t0 = time.time()
        running_loss, running_correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device); y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            running_loss   += loss.item() * y.size(0)
            running_correct += (preds == y).sum().item()
            total          += y.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_acc  = running_correct / max(total, 1)

        val_acc, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg['max_epochs']} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state': model.state_dict(), 'config': cfg}, best_path)
            print(f"→ Saved new best model to {best_path}")

    # ---------- Test ----------
    print("\nEvaluating best checkpoint on test set…")
    ckpt = torch.load(best_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

    # Use class names saved in cfg
    class_names = ckpt['config']['class_names']
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(labels, preds).tolist()
    out = {
        'test_accuracy': test_acc,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    with open(os.path.join(cfg['run_dir'], 'test_report.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(report)


if __name__ == '__main__':
    main()

