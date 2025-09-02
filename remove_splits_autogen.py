import shutil
import os

folder = 'leapGestRecog_flat/.splits_autogen'
if os.path.exists(folder):
    shutil.rmtree(folder)
    print(f"Removed {folder}")
else:
    print(f"{folder} does not exist.")
