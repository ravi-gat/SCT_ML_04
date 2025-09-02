
import os
import json
import random
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def discover_dataset(data_dir: str):
    """
    Expects a folder with subfolders per class (ImageFolder-style).
    If train/val folders are absent, will create a deterministic split (85/15) virtually.
    Returns:
        train_dir, val_dir, class_names
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # Use existing split
        ds_tmp = datasets.ImageFolder(train_dir)
        class_names = ds_tmp.classes
        return train_dir, val_dir, class_names

    # Otherwise assume data_dir contains class subfolders and perform a virtual split
    ds = datasets.ImageFolder(data_dir)
    class_names = ds.classes

    # Create tmp split folders under .splits (symlinks if possible, else copy paths list)
    split_root = os.path.join(data_dir, ".splits_autogen")
    os.makedirs(split_root, exist_ok=True)
    s_train = os.path.join(split_root, "train")
    s_val = os.path.join(split_root, "val")
    for p in (s_train, s_val):
        os.makedirs(p, exist_ok=True)
        for c in class_names:
            os.makedirs(os.path.join(p, c), exist_ok=True)

    # Deterministic split by hashing path order
    indices = list(range(len(ds.samples)))
    rng = np.random.RandomState(123)
    rng.shuffle(indices)
    n_train = int(0.85 * len(indices))
    train_idx = set(indices[:n_train])

    # Write text manifests for speed (paths only)
    train_txt = os.path.join(split_root, "train_list.txt")
    val_txt = os.path.join(split_root, "val_list.txt")
    with open(train_txt, "w") as ft, open(val_txt, "w") as fv:
        for i, (path, label) in enumerate(ds.samples):
            if i in train_idx:
                ft.write(f"{path}\t{label}\n")
            else:
                fv.write(f"{path}\t{label}\n")

    return s_train, s_val, class_names

def save_class_index(out_dir: str, class_names: List[str]):
    d = {"class_names": class_names}
    with open(os.path.join(out_dir, "class_index.json"), "w") as f:
        json.dump(d, f, indent=2)

def load_class_index(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)["class_names"]

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, pred, target):
        n = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0.0:
        return x, y, 1.0, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, lam, y_b

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
