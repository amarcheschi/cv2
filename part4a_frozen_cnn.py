"""Part 4.A - linear SVM on frozen ResNet-18 features.

Pipeline
--------
1. Run every image of a split through a frozen, pretrained ResNet-18.
2. Global-average-pool the output of the last block of layer2/layer3/layer4
   and concatenate -> 128 + 256 + 512 = 896-D descriptor per image.
3. Cache the descriptors to data/deep_features/<tag>/<subset>.npz.
4. L2-normalise, then train the same LinearSVC used in Part 1
   (src.utils.train_svm) for each one-vs-rest category.

Protocol (PDF compliant): the supplied *_val split is used ONCE as test.
SVM C is selected on an internal validation split carved out of *_train.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC


def _find_code_dir():
    """Locate the directory containing src/ (works from code/ or the repo root)."""
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        for candidate in (base, base / "code"):
            if (candidate / "src" / "deep_models.py").exists():
                return candidate
    raise FileNotFoundError(
        "Could not find src/deep_models.py. Run this script from the code/ "
        "directory or from the repository root."
    )


CODE_DIR = _find_code_dir()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)  # data/ and src/ paths are relative to code/

from src.deep_models import IntermediateFeatureExtractor, getModel
from src.utils import loadSamplesTable, train_svm

LAYERS = ("layer2.1", "layer3.1", "layer4.1")
CATEGORIES = ("aeroplane", "car", "horse", "motorbike", "person")


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def extract_split(df, extractor, preprocess, device, batch_size, layers):
    """Return an (N, D) float32 matrix of pooled intermediate features."""
    feats = []
    with torch.inference_mode():
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            images = [preprocess(Image.open(p).convert("RGB"))
                      for p in batch.filepath]
            x = torch.stack(images).to(device)
            _, maps = extractor(x)
            desc = torch.cat(
                [F.adaptive_avg_pool2d(maps[name], 1).flatten(1) for name in layers],
                dim=1,
            )
            feats.append(desc.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def get_or_extract(subset, args, extractor, preprocess, device):
    cache = Path("data/deep_features") / args.tag / f"{subset}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not args.force:
        data = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache} {data['features'].shape}")
        return data["features"], data["file_names"], data["categories"]

    df = loadSamplesTable("data/splits", "data/images", subset=subset, per_class=-1)
    if args.limit:
        df = df.groupby("category", group_keys=False).head(args.limit).reset_index(drop=True)

    print(f"[extract] {subset}: {len(df)} images")
    t0 = time.perf_counter()
    features = extract_split(df, extractor, preprocess, device, args.batch_size, LAYERS)
    dt = time.perf_counter() - t0
    print(f"[extract] {features.shape[1]}-D, {dt:.1f}s "
          f"({1000 * dt / max(len(df), 1):.1f} ms/image, "
          f"device={device}, preprocessing included)")

    np.savez_compressed(
        cache,
        features=features,
        file_names=df.filename.to_numpy().astype(str),
        categories=df.category.to_numpy().astype(str),
    )
    return features, df.filename.to_numpy(), df.category.to_numpy()


def select_c(X_train, y_train, X_val, y_val, grid, max_iter):
    best_c, best_ap = None, -np.inf
    for c in grid:
        w, b = train_svm(X_train, y_train, C=c, max_iter=max_iter)
        ap = average_precision_score(y_val, X_val @ w + b)
        print(f"  C={c:<7g} val AP={ap:.4f}")
        if ap > best_ap:
            best_c, best_ap = c, ap
    return best_c, best_ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--layers", default="all", choices=("all", "layer4"),
                    help="'all' = GAP of layer2/3/4 (896-D), 'layer4' = 512-D ablation")
    ap.add_argument("--tag", default="resnet18_gap896")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--c-grid", type=float, nargs="+",
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: max images per category (0 = all)")
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.layers == "layer4":
        global LAYERS
        LAYERS = ("layer4.1",)
        if args.tag == "resnet18_gap896":
            args.tag = "resnet18_layer4_512"
    model, preprocess = getModel(args.model, device, nofc=False)
    extractor = IntermediateFeatureExtractor(model)

    Ftr, names_tr, cats_tr = get_or_extract("train", args, extractor, preprocess, device)
    Fte, names_te, cats_te = get_or_extract("val", args, extractor, preprocess, device)
    Ftr, Fte = l2norm(Ftr), l2norm(Fte)

    # ---- internal validation split (grouped per category, seeded) -----------
    rng = np.random.default_rng(args.seed)
    is_val = np.zeros(len(cats_tr), dtype=bool)
    for cat in np.unique(cats_tr):
        idx = np.flatnonzero(cats_tr == cat)
        is_val[rng.choice(idx, max(1, int(round(args.val_fraction * len(idx)))),
                          replace=False)] = True
    print(f"\ntrain={np.sum(~is_val)}  internal val={np.sum(is_val)}  test={len(cats_te)}")

    # ---- one-vs-rest per category -------------------------------------------
    results = {}
    for cat in CATEGORIES:
        if cat not in set(cats_tr):
            continue
        y_tr = np.where(cats_tr == cat, 1, -1)
        y_te = np.where(cats_te == cat, 1, -1)
        print(f"\n=== {cat}: {(y_tr > 0).sum()} pos / {(y_tr < 0).sum()} neg ===")
        c, _ = select_c(Ftr[~is_val], y_tr[~is_val], Ftr[is_val], y_tr[is_val],
                        args.c_grid, args.max_iter)
        # refit on the full training split with the selected C
        w, b = train_svm(Ftr, y_tr, C=c, max_iter=args.max_iter)
        test_scores = Fte @ w + b
        test_ap = average_precision_score(y_te, test_scores)
        results[cat] = (test_ap, c)

        # optional multiclass confusion matrix uses the same per-class scores
        print(f"  selected C={c:g} | TEST AP={test_ap:.4f}")

    m_ap = np.mean([v[0] for v in results.values()])
    print("\n--- Part 4.A frozen CNN (one-vs-rest, test split) ---")
    for cat, (score, c) in results.items():
        print(f"{cat:<10s} AP={score:.4f}  (C={c:g})")
    print(f"mAP={m_ap:.4f}  dim={Ftr.shape[1]}")

    # ---- 6-way confusion matrix (5 objects + background) ---------------------
    if not args.limit:
        labels = np.unique(cats_tr)
        clf = LinearSVC(C=10, max_iter=args.max_iter, dual=True, random_state=42)
        clf.fit(Ftr, cats_tr)
        pred = clf.predict(Fte)
        print("\nMulticlass accuracy:", (pred == cats_te).mean())
        print("Macro F1:", f1_score(cats_te, pred, average="macro"))
        print("Confusion matrix (rows=true, cols=pred):")
        print("labels:", list(labels))
        print(confusion_matrix(cats_te, pred, labels=labels))


if __name__ == "__main__":
    main()
