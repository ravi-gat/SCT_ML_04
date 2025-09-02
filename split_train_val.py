import os
import shutil
import random
from glob import glob

src_root = 'leapGestRecog_flat'
train_root = os.path.join(src_root, 'train')
val_root = os.path.join(src_root, 'val')
train_ratio = 0.85

# Remove old splits if they exist
for split_root in [train_root, val_root]:
    if os.path.exists(split_root):
        shutil.rmtree(split_root)
    os.makedirs(split_root)

# For each class folder
for class_name in os.listdir(src_root):
    class_src = os.path.join(src_root, class_name)
    if not os.path.isdir(class_src) or class_name in ['train', 'val']:
        continue
    images = glob(os.path.join(class_src, '*.png'))
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]
    # Create class folders in train/val
    train_class = os.path.join(train_root, class_name)
    val_class = os.path.join(val_root, class_name)
    os.makedirs(train_class, exist_ok=True)
    os.makedirs(val_class, exist_ok=True)
    # Copy images
    for img in train_imgs:
        shutil.copy2(img, os.path.join(train_class, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy2(img, os.path.join(val_class, os.path.basename(img)))
print('Train/val split complete. Use leapGestRecog_flat as your data_dir.')
