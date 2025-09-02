
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from utils import seed_everything, discover_dataset, save_class_index, AverageMeter, accuracy, EarlyStopping, LabelSmoothingCrossEntropy, mixup_data, mixup_criterion

class AlbumentationsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.aug = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        # Convert PIL to numpy for Albumentations
        image_np = np.array(image)
        if self.aug is not None:
            image_np = self.aug(image=image_np)["image"]
        return image_np, target

def build_transforms(img_size):
    train_tfms = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0,0,0)),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
        ], p=0.7),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=25, p=0.7),
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_tfms = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0,0,0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_tfms, val_tfms

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def get_model(num_classes, model_name='resnet50', pretrained=True):
    if model_name.lower() == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name.lower() == 'mobilenet_v3':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported model. Use resnet50 or mobilenet_v3.")
    return net

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, mixup_alpha=0.4):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_alpha > 0.0:
            images, targets_a, lam, targets_b = mixup_data(images, targets, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            if mixup_alpha > 0.0:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # compute accuracy on non-mixed targets best-effort
        acc1 = accuracy(outputs, targets, topk=(1,))[0] if mixup_alpha == 0.0 else 0.0

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(acc1, bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{top1_meter.avg:.2f}")
    return loss_meter.avg, top1_meter.avg

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()

    all_preds = []
    all_tgts = []
    for images, targets in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        top1_meter.update(acc1, bs)
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_tgts.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_tgts)
    return loss_meter.avg, top1_meter.avg, preds, tgts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root. Subfolders should be classes or contain train/val.")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "mobilenet_v3"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)


    # Use explicit train/val folders if they exist
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        from torchvision.datasets import ImageFolder
        class_names = ImageFolder(train_dir).classes
    else:
        train_dir, val_dir, class_names = discover_dataset(args.data_dir)

    train_tfms, val_tfms = build_transforms(args.img_size)
    train_ds = AlbumentationsImageFolder(train_dir, transform=train_tfms)
    val_ds   = AlbumentationsImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(class_names), model_name=args.model_name, pretrained=True).to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Prepare output directory
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"exp-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_class_index(out_dir, class_names)

    early = EarlyStopping(patience=args.patience, min_delta=0.0)
    best_acc = 0.0
    best_wts = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, mixup_alpha=args.mixup_alpha)
        val_loss, val_acc, preds, tgts = evaluate(model, val_loader, device, criterion)
        scheduler.step(epoch + val_loss)

        print(f"Train Loss {train_loss:.4f} | Acc@1 {train_acc:.2f}")
        print(f"Valid Loss {val_loss:.4f} | Acc@1 {val_acc:.2f}")

        # Save last
        torch.save({"model": model.state_dict(),
                    "model_name": args.model_name,
                    "img_size": args.img_size,
                    "class_index_path": str(out_dir / "class_index.json")},
                   out_dir / "last.pth")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict()
            torch.save({"model": best_wts,
                        "model_name": args.model_name,
                        "img_size": args.img_size,
                        "class_index_path": str(out_dir / "class_index.json")},
                       out_dir / "best.pth")
            print(f"Saved new best with Acc@1 {best_acc:.2f}")

        early.step(val_acc)
        if early.should_stop:
            print("Early stopping triggered.")
            break

    # Final evaluation & reports
    model.load_state_dict(best_wts if best_wts is not None else torch.load(out_dir / "last.pth")["model"])
    val_loss, val_acc, preds, tgts = evaluate(model, val_loader, device, criterion)

    report = classification_report(tgts, preds, target_names=class_names, digits=4)
    print("\nValidation report:\n", report)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(tgts, preds)
    plot_confusion_matrix(cm, class_names, save_path=out_dir / "confusion_matrix.png")
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
