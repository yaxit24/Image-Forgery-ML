# train_model.py
import os, joblib
import numpy as np
from src.preprocessing import load_image_pil
from src.detectors import compute_ela, extract_residual, blockiness_map, copy_move_orb_mask
from src.features import dict_to_vector, feature_names
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data/train"   # data has subfolders 'authentic' and 'forged'
MODEL_PATH = "forensics_model.pkl"
SCALER_PATH = "forensics_scaler.pkl"

def gather_examples():
    X = [] #feature vectors
    y = [] #lables 0-> forged 1-> authentic here.
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
            # run detectors 
            ela_img, ela_stats = compute_ela(pil)
            res_map, res_stats = extract_residual(arr)
            # ensure grayscale for blockiness
            if arr.ndim == 3:
                gray = arr[...,0]  # quick grayscale using R channel (or use cv2)
            else:
                gray = arr
            block_map, block_stats = blockiness_map(gray)
            mask, copy_stats = copy_move_orb_mask(arr)
            feat = dict_to_vector(ela_stats, res_stats, block_stats, copy_stats)
            X.append(feat)
            y.append(lab)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Gathering examples...")
    X, y = gather_examples()
    print("Examples:", X.shape, y.shape)
    # simple sanity check
    if len(X) < 10:
        print("WARNING: too few examples to train reliably.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    p, r, f, _ = precision_recall_fscore_support(yte, ypred, average='binary', zero_division=0)
    print("Validation — acc {:.3f}, prec {:.3f}, rec {:.3f}, f1 {:.3f}".format(acc, p, r, f))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Saved model:", MODEL_PATH, "and scaler:", SCALER_PATH)
    print("Feature names order:", feature_names())
