import os
import shutil

root = 'leapGestRecog_flat'
for dirpath, dirnames, filenames in os.walk(root):
    for dirname in dirnames:
        if dirname == '.splits_autogen':
            full_path = os.path.join(dirpath, dirname)
            print(f"Removing {full_path}")
            shutil.rmtree(full_path)
print('All .splits_autogen folders removed recursively.')
