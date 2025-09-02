import os
import shutil
from glob import glob

src_root = 'leapGestRecog'
dst_root = 'leapGestRecog_flat'

if not os.path.exists(dst_root):
    os.makedirs(dst_root)

# Find all numbered folders (00, 01, ...)
for numbered in os.listdir(src_root):
    if numbered.startswith('.'):
        continue  # skip hidden/system folders
    numbered_path = os.path.join(src_root, numbered)
    if not os.path.isdir(numbered_path):
        continue
    # For each class folder inside numbered folder
    for class_name in os.listdir(numbered_path):
        if class_name.startswith('.'):
            continue  # skip hidden/system folders
        class_src = os.path.join(numbered_path, class_name)
        if not os.path.isdir(class_src):
            continue
        class_dst = os.path.join(dst_root, class_name)
        os.makedirs(class_dst, exist_ok=True)
        # Copy all images
        for img_path in glob(os.path.join(class_src, '*')):
            if not os.path.isfile(img_path):
                continue  # skip directories
            base = os.path.basename(img_path)
            new_name = f'{numbered}_{base}'
            dst_path = os.path.join(class_dst, new_name)
            shutil.copy2(img_path, dst_path)
print('Flattening complete. Use leapGestRecog_flat as your data directory for training.')
