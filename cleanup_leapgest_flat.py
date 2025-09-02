import os
import shutil

flat_dir = 'leapGestRecog_flat'
# List of valid class folder names (those containing an underscore)
for folder in os.listdir(flat_dir):
    folder_path = os.path.join(flat_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    # Remove folders that do not contain an underscore (not class folders)
    if '_' not in folder:
        print(f"Removing non-class folder: {folder_path}")
        shutil.rmtree(folder_path)
print('Cleanup complete. Only class folders remain in leapGestRecog_flat.')
