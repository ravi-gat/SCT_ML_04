# Hand Gesture Recognition — Advanced PyTorch Starter

This is a production-ready training and inference pipeline that uses transfer learning
(ResNet50 by default), mixed precision, label smoothing, MixUp, cosine LR schedule,
and early stopping. It auto-discovers your dataset in the **same folder** and
works with either a single-folder ImageNet-style dataset or an explicit `train/` and `val/` split.

## 1) Dataset layout

Place your dataset in a folder; the script will try the following:

- If `data/train` and `data/val` exist, it uses them (each with subfolders per class).
- Otherwise it assumes `data/` contains class subfolders and creates a deterministic 85/15 split.

You can adjust with `--data_dir`.

```
your_project/
  data/
    classA/
      img1.jpg
      ...
    classB/
      ...
```

## 2) Quick start

Create a virtual env, install requirements, and train:

```bash
pip install -r requirements.txt
python train.py --data_dir ./data --epochs 20 --batch_size 32 --img_size 256 --model_name resnet50
```

Artifacts are written to `./outputs/exp-YYYYmmdd-HHMMSS/`:
- `best.pth` and `last.pth`
- `class_index.json`
- `report.txt` (precision/recall/F1)
- `confusion_matrix.png`

## 3) Inference on an image

```bash
python infer_image.py --ckpt outputs/exp-*/best.pth --image path/to/sample.jpg
```

## 4) Real-time webcam demo

```bash
python infer_webcam.py --ckpt outputs/exp-*/best.pth
# Press 'q' to quit
```

## 5) Tips
- Use `--model_name mobilenet_v3` for a lighter, faster model.
- Increase `--img_size` to 320+ for higher accuracy (at some cost).
- Tweak `--mixup_alpha` (0.0 disables it) and `--label_smoothing` for better generalization.
- If you already created your own `train/` and `val/` folders, the script will use them directly.

Good luck, and happy training! ✌️
