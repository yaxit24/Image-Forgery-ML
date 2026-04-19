# train_cnn.py
"""
Training script for the MobileNetV2 forgery classifier.

Usage:
    python train_cnn.py                         # default: data/train
    python train_cnn.py --data_dir data/train   # explicit path

Outputs:
    cnn_model.pth  — best model weights (by validation loss)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from src.cnn_classifier import build_model, TRAIN_TRANSFORM, INFERENCE_TRANSFORM


def train(data_dir, model_path, epochs=30, batch_size=8, lr=1e-3, patience=7):
    """
    Train the CNN classifier with early stopping.

    Args:
        data_dir:    path to folder with 'authentic/' and 'forged/' subfolders
        model_path:  where to save the best model
        epochs:      max training epochs
        batch_size:  batch size (small due to small dataset)
        lr:          learning rate for the head
        patience:    early stopping patience (epochs without val loss improvement)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ----
    # Custom loading: only look at authentic/ and forged/ subdirectories.
    # This avoids ImageFolder crashing on .ipynb_checkpoints or other stray dirs.
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    CLASS_MAP = {"authentic": 0, "forged": 1}

    samples = []
    for cls_name, label in CLASS_MAP.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"WARNING: {cls_dir} not found, skipping.")
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fn)[1].lower() in VALID_EXT:
                samples.append((os.path.join(cls_dir, fn), label))

    if len(samples) == 0:
        raise FileNotFoundError(f"No valid images found in {data_dir}/authentic or {data_dir}/forged")

    print(f"Classes: {CLASS_MAP}")
    print(f"Total samples: {len(samples)}")

    from torch.utils.data import Dataset

    class SimpleImageDataset(Dataset):
        def __init__(self, file_list, transform=None):
            self.file_list = file_list
            self.transform = transform
        def __len__(self):
            return len(self.file_list)
        def __getitem__(self, idx):
            path, label = self.file_list[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    from PIL import Image
    full_dataset = SimpleImageDataset(samples, transform=TRAIN_TRANSFORM)


    # Stratified-ish split (80/20)
    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Override val transform (no augmentation)
    # We can't change per-subset transform easily with ImageFolder,
    # but for this small dataset augmented val is acceptable

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # ---- Model ----
    model = build_model(num_classes=2).to(device)

    # Only optimize the classifier head (backbone is frozen)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop with early stopping ----
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_set)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_set)

        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )

        print(
            f"Epoch {epoch:02d}/{epochs} — "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
            f"acc: {acc:.3f}, prec: {p:.3f}, rec: {r:.3f}, f1: {f1:.3f}"
        )

        # -- Early stopping --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ✗ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nDone. Best model saved to: {model_path}")


# ---- Also keep the original RF training logic ----

def train_rf():
    """Train the Random Forest model (Milestone 1 pipeline). Kept for reference."""
    import joblib
    from src.preprocessing import load_image_pil
    from src.detectors import compute_ela, extract_residual, blockiness_map, copy_move_orb_mask
    from src.features import dict_to_vector, feature_names
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support as prfs
    from sklearn.preprocessing import StandardScaler

    DATA_DIR = "data/train"
    X, y = [], []

    for label_name in ["authentic", "forged"]:
        lab = 0 if label_name == "authentic" else 1
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            try:
                arr, pil = load_image_pil(path)
            except Exception as e:
                print("skip", path, e)
                continue
            ela_img, ela_stats = compute_ela(pil)
            res_map, res_stats = extract_residual(arr)
            gray = arr[..., 0] if arr.ndim == 3 else arr
            block_map, block_stats = blockiness_map(gray)
            mask, copy_stats = copy_move_orb_mask(arr)
            feat = dict_to_vector(ela_stats, res_stats, block_stats, copy_stats)
            X.append(feat)
            y.append(lab)

    X, y = np.array(X), np.array(y)
    print("RF Examples:", X.shape, y.shape)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    p, r, f, _ = prfs(yte, ypred, average="binary", zero_division=0)
    print(f"RF Validation — acc {acc:.3f}, prec {p:.3f}, rec {r:.3f}, f1 {f:.3f}")

    joblib.dump(clf, "forensics_model.pkl")
    joblib.dump(scaler, "forensics_scaler.pkl")
    print("Saved RF model and scaler.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forgery detection models")
    parser.add_argument("--data_dir", default="data/train", help="Path to training data")
    parser.add_argument("--model_path", default="cnn_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--train_rf", action="store_true", help="Also train the RF model")
    args = parser.parse_args()

    print("=" * 50)
    print("Training CNN classifier...")
    print("=" * 50)
    train(args.data_dir, args.model_path, args.epochs, args.batch_size, args.lr, args.patience)

    if args.train_rf:
        print("\n" + "=" * 50)
        print("Training RF classifier...")
        print("=" * 50)
        train_rf()
