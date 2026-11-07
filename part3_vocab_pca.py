"""Part 3 - large BoVW vocabularies with image-level PCA vs small vocabularies.

What this script does
---------------------
1. Freezes the evaluation protocol:
   - supplied ``*_val`` split is the final TEST set (evaluated once);
   - the shared internal-validation split from Part 4 is used for model
     selection (``results/part4_internal_val_names.csv`` when available);
   - codebooks and PCA are fitted on TRAIN data only.
2. Trains the BoVW codebooks needed by the experiment (K = 128, 256, 512, 1024
   by default) on the same sampled descriptor matrix, and extracts global
   (default) or 2x2-tiled BoVW encodings.
3. Fits PCA to the *image-level* BoVW histograms - NOT to local SIFT
   descriptors.  ``src.train -c`` is a different operation (local descriptor
   PCA before k-means) and must not be used for Part 3.
4. Evaluates conditions A-D with the Part 1/2/4 protocol:
   - one-vs-rest AP per category (positives = category, negatives = background)
     and mAP over the five categories;
   - test accuracy / macro-F1 from the same binary tasks, plus an optional
     6-way confusion matrix on de-duplicated single-label images;
   - SVM C selected on internal validation only.
5. Measures codebook fitting, encoding, PCA fitting, SVM training, inference
   time and feature storage.
6. Runs a PCA-dimension sweep on validation data and writes plots/tables.

Typical usage
-------------
From the ``code/`` directory (or the repository root)::

    # Full Part 3 pipeline with the provided K=256/K=512 codebooks.
    python part3_vocab_pca.py --mode all

    # Strict protocol: retrain every codebook on train minus internal-val.
    python part3_vocab_pca.py --mode all --codebook-source dev

    # Fast check using only the existing K=512 encodings.
    python part3_vocab_pca.py --mode evaluate --k-grid 512 \
        --pca-dims 128 --no-sweep --no-multiclass

    # Add the 2x2 spatial-tiling control.
    python part3_vocab_pca.py --mode all --tiled

Notes
-----
* The default representation is the *global* K-dimensional BoVW histogram,
  which matches the assignment's statement that the BoVW histogram dimension
  is K.  Use ``--tiled`` for the 4K-dimensional spatial-pyramid control.
* ``removeSpatialInformation`` in ``src/encoders.py`` assumes an interleaved
  tile layout; the encoders actually concatenate tile histograms.  This script
  therefore implements the correct tile-major reduction locally.
* ``results/part4_internal_val_names.csv`` stores file names with leading
  zeros; always read it with ``dtype=str``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Locate the code directory and import the project modules
# ---------------------------------------------------------------------------
def _find_code_dir() -> Path:
    """Locate the directory containing ``src/`` (works from code/ or root)."""
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        for candidate in (base, base / "code"):
            if (candidate / "src" / "deep_eval.py").exists():
                return candidate
    raise FileNotFoundError(
        "Could not find src/deep_eval.py. Run this script from the code/ "
        "directory or from the repository root."
    )


CODE_DIR = _find_code_dir()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)

from src import utils  # noqa: E402
from src.deep_eval import (  # noqa: E402
    BACKGROUND,
    CATEGORIES,
    DEFAULT_C_GRID,
    binary_subset,
    evaluate_binary,
    evaluate_multiclass,
    internal_val_names,
    l2norm,
    save_results,
    storage_mb,
    sweep_c,
)
from src.encoders import ImageEncoder, load_npz  # noqa: E402
from src.extract import encodeFeatures  # noqa: E402
from src.train import sampleLocalFeatures  # noqa: E402

ALL_CATEGORIES = tuple(sorted(CATEGORIES + (BACKGROUND,)))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[part3] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tag_suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def global_from_tiled(histograms: np.ndarray) -> np.ndarray:
    """Average 2x2 tiled histograms into one global histogram per image.

    ``ImageEncoder.encode_tiling`` concatenates the four tile histograms
    (tile-major layout), so the correct reshape is ``(N, 4, K)``.  The
    provided ``removeSpatialInformation`` reshapes to ``(N, K, 4)`` and is
    therefore incorrect for this layout.
    """
    histograms = np.asarray(histograms)
    if histograms.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {histograms.shape}")
    n, d = histograms.shape
    if d % 4 != 0:
        raise ValueError(
            f"Tiled encoding dimension {d} is not divisible by 4; cannot "
            "reduce to a global histogram."
        )
    return histograms.reshape(n, 4, d // 4).mean(axis=1)


def read_csv_strings(path: Path) -> pd.DataFrame:
    """Read a CSV with file names preserved as strings (leading zeros)."""
    return pd.read_csv(path, dtype=str)


def split_hash(names: Iterable[str]) -> str:
    """Short stable hash used to key caches to the internal-validation split."""
    payload = "|".join(sorted(str(n) for n in names)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    name: str
    K: int
    pca_dim: Optional[int]
    representation: str

    @property
    def pca_label(self) -> str:
        return "none" if self.pca_dim is None else str(self.pca_dim)


def build_conditions(args: argparse.Namespace, n_train_dev: int) -> List[Condition]:
    """Build the A-D condition matrix and validate PCA dimensions."""
    tiled = bool(args.tiled)
    rep = "bovw_tiled" if tiled else "bovw_global"
    conditions: List[Condition] = []
    seen: Dict[Tuple[int, Optional[int]], str] = {}
    aliases: Dict[str, str] = {}
    required_k = set(int(k) for k in args.k_grid)

    def add(name: str, K: int, pca_dim: Optional[int]) -> None:
        key = (int(K), pca_dim)
        if key in seen:
            return
        seen[key] = name
        conditions.append(Condition(name=name, K=int(K), pca_dim=pca_dim,
                                    representation=rep))
        required_k.add(int(K))

    max_pca_dim = max(1, n_train_dev - 1)
    for K in sorted(set(int(k) for k in args.k_grid)):
        raw_dim = 4 * K if tiled else K
        add(f"K{K}_none", K, None)
        for d in args.pca_dims:
            d = int(d)
            if d >= raw_dim:
                log(f"skip K={K} PCA d={d}: d must be < original dimension {raw_dim}")
                continue
            if d > max_pca_dim:
                log(f"skip K={K} PCA d={d}: d > n_train_dev - 1 = {max_pca_dim}")
                continue
            add(f"K{K}_pca{d}", K, d)

    # Condition D: directly small codebook at a matched final dimension.
    for d in args.matched_dims:
        d = int(d)
        k_match = d // 4 if tiled else d
        if k_match < 2:
            log(f"skip matched dimension {d}: implied K={k_match} is too small")
            continue
        name = f"D{d}_K{k_match}"
        if (k_match, None) in seen:
            aliases[name] = seen[(k_match, None)]
            log(
                f"matched condition D{d} coincides with {seen[(k_match, None)]} "
                f"(K={k_match}, no PCA)"
            )
        else:
            add(name, k_match, None)

    args.required_k = sorted(required_k)
    args.condition_alias = aliases
    return conditions


# ---------------------------------------------------------------------------
# Internal validation split
# ---------------------------------------------------------------------------
def get_internal_val_names(
    args: argparse.Namespace, train_df: pd.DataFrame
) -> np.ndarray:
    """Load, or create, the shared internal-validation file names."""
    path = Path(args.val_names)
    if path.exists():
        df = read_csv_strings(path)
        col = "file_name" if "file_name" in df.columns else df.columns[0]
        names = df[col].astype(str).str.strip().to_numpy()
        log(f"loaded {len(np.unique(names))} internal-val names from {path}")
    else:
        names = internal_val_names(
            train_df["filename"].to_numpy(),
            train_df["category"].to_numpy(),
            fraction=args.val_fraction,
            seed=args.val_seed,
        )
        ensure_dir(path.parent)
        pd.DataFrame({"file_name": names}).to_csv(path, index=False)
        log(f"wrote {len(np.unique(names))} internal-val names to {path}")
    return np.unique(names)


# ---------------------------------------------------------------------------
# Codebook preparation
# ---------------------------------------------------------------------------
def _codebook_meta(enc: Dict[str, np.ndarray]) -> Dict[str, object]:
    def scalar(key: str, default=None):
        if key not in enc:
            return default
        value = enc[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value

    return {
        "num_words": int(scalar("num_words", 0)),
        "num_samples": int(scalar("num_samples", 0) or 0),
        "num_samples_per_image": int(scalar("num_samples_per_image", 0) or 0),
        "num_images": int(scalar("num_images", 0) or 0),
        "seed": int(scalar("seed", -1) or -1),
        "fit_seconds": float(scalar("fit_seconds", float("nan")) or float("nan")),
        "codebook_source": str(scalar("codebook_source", "unknown")),
    }


def fit_codebook(
    descs: np.ndarray,
    K: int,
    args: argparse.Namespace,
    seed: int,
    source: str,
    n_images: int,
) -> Tuple[Path, Dict[str, object]]:
    """Fit one k-means codebook and save it in the project's npz format."""
    log(f"fitting K={K} k-means on {descs.shape[0]} descriptors "
        f"(n_init={args.kmeans_n_init}, max_iter={args.kmeans_max_iter})")
    t0 = time.perf_counter()
    km = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=seed,
    )
    km.fit(descs)
    fit_s = time.perf_counter() - t0
    log(f"  K={K}: fitted in {fit_s:.1f}s")

    out_dir = ensure_dir(Path(args.encoders_dir))
    out_path = out_dir / f"part3_bovw_K{K}_C0.npz"
    np.savez_compressed(
        out_path,
        type="bovw",
        num_words=K,
        pca_dim=0,
        projection=np.array(None, dtype=object),
        words=km.cluster_centers_.astype(np.float32),
        num_samples=int(descs.shape[0]),
        num_samples_per_image=int(args.per_image),
        num_images=int(n_images),
        seed=int(seed),
        cache_dir=str(args.cache_dir),
        fit_seconds=float(fit_s),
        codebook_source=source,
    )
    meta = {
        "num_words": K,
        "num_samples": int(descs.shape[0]),
        "num_samples_per_image": int(args.per_image),
        "num_images": int(n_images),
        "seed": int(seed),
        "fit_seconds": float(fit_s),
        "codebook_source": source,
    }
    return out_path, meta


def resolve_codebook_path(K: int, args: argparse.Namespace) -> Path:
    """Return the path of the codebook that prepare/evaluate should use."""
    provided = Path(args.encoders_dir) / f"bovw_K{K}_C0.npz"
    dev_path = Path(args.encoders_dir) / f"part3_bovw_K{K}_C0.npz"
    if args.codebook_source == "provided" and provided.exists() and not args.force_codebooks:
        return provided
    return dev_path


def resolve_encoding_dir(K: int, args: argparse.Namespace) -> Path:
    return Path(args.encodings_dir) / resolve_codebook_path(K, args).stem


def limit_per_class(
    df: pd.DataFrame, per_class: int, seed: int
) -> pd.DataFrame:
    """Sample at most ``per_class`` images per category for codebook fitting."""
    if per_class <= 0:
        return df.reset_index(drop=True)
    sampled = []
    for _, group in df.groupby("category", sort=True):
        n = min(per_class, len(group))
        sampled.append(group.sample(n=n, random_state=seed))
    return pd.concat(sampled).sort_values(["category", "filename"]).reset_index(drop=True)


def ensure_codebook(
    K: int,
    args: argparse.Namespace,
    df_fit: pd.DataFrame,
    descs_cache: Dict[str, object],
) -> Tuple[Path, Dict[str, object]]:
    """Return a codebook path, training it if necessary.

    ``--codebook-source provided`` reuses ``data/encoders/bovw_K<K>_C0.npz``
    when available.  Any missing codebook is trained on dev-train (the supplied
    train split minus the internal validation names).  ``dev`` forces every
    codebook to be trained on dev-train; this is the strict protocol.
    """
    provided = Path(args.encoders_dir) / f"bovw_K{K}_C0.npz"
    dev_path = Path(args.encoders_dir) / f"part3_bovw_K{K}_C0.npz"

    if args.codebook_source == "provided" and provided.exists() and not args.force_codebooks:
        meta = _codebook_meta(load_npz(provided))
        meta["codebook_source"] = "provided"
        log(f"using provided codebook {provided}")
        return provided, meta

    if dev_path.exists() and not args.force_codebooks:
        meta = _codebook_meta(load_npz(dev_path))
        log(f"using cached dev codebook {dev_path}")
        return dev_path, meta

    if "descs" not in descs_cache:
        cache_path = ensure_dir(Path(args.part3_dir)) / (
            f"descs_n{len(df_fit)}_pc{args.per_class}_per{args.per_image}_"
            f"seed{args.seed}_{args.val_hash}.npy"
        )
        if cache_path.exists() and not args.force_codebooks:
            log(f"loading descriptor sample {cache_path}")
            descs_cache["descs"] = np.load(cache_path)
        else:
            log(f"sampling local descriptors from {len(df_fit)} dev-train images")
            np.random.seed(args.seed)
            t0 = time.perf_counter()
            descs = sampleLocalFeatures(
                df_fit,
                num_samples=args.per_image,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )
            log(f"  sampled {descs.shape[0]} descriptors in "
                f"{time.perf_counter() - t0:.1f}s")
            np.save(cache_path, descs.astype(np.float32))
            descs_cache["descs"] = descs.astype(np.float32)

    descs = np.asarray(descs_cache["descs"])
    return fit_codebook(descs, K, args, args.seed, "dev", len(df_fit))


# ---------------------------------------------------------------------------
# Encoding extraction
# ---------------------------------------------------------------------------
def expected_encoding_files(enc_dir: Path) -> List[Path]:
    return [
        enc_dir / f"{cat}_{subset}.npz"
        for cat in ALL_CATEGORIES
        for subset in ("train", "val")
    ]


def ensure_encodings(
    encoder_path: Path, args: argparse.Namespace, df_all: pd.DataFrame
) -> Tuple[Path, Dict[str, object]]:
    """Extract per-image encodings with ``src.extract.encodeFeatures``."""
    enc_dir = Path(args.encodings_dir) / encoder_path.stem
    # A retrained dev codebook invalidates its cached encodings automatically.
    overwrite = bool(
        args.force_encodings
        or (args.force_codebooks and encoder_path.stem.startswith("part3_"))
    )
    files = expected_encoding_files(enc_dir)
    if all(p.exists() for p in files) and not overwrite:
        log(f"using cached encodings {enc_dir}")
        return enc_dir, {"encoding_seconds": float("nan"), "encoding_images": int(len(df_all))}

    ensure_dir(enc_dir)
    log(f"extracting encodings for {encoder_path.stem} ({len(df_all)} rows)")
    t0 = time.perf_counter()
    encodeFeatures(
        df_all,
        encoder_path,
        output_dir=enc_dir,
        cache_dir=args.cache_dir,
        overwrite=overwrite,
        jobs=args.jobs,
    )
    seconds = time.perf_counter() - t0
    return enc_dir, {"encoding_seconds": float(seconds), "encoding_images": int(len(df_all))}


def benchmark_encoding(
    encoder_path: Path, df_all: pd.DataFrame, args: argparse.Namespace,
    n_images: int = 40,
) -> Dict[str, float]:
    """Time pure encoding on cached descriptors (SIFT extraction excluded)."""
    encoder = ImageEncoder(str(encoder_path))
    subset = (
        df_all[df_all["subset"] == "train"]
        .drop_duplicates("filename")
        .head(n_images)
    )
    items = []
    for path in subset["filepath"]:
        try:
            kpts, desc, isize = encoder.computeFeatures(
                path, cache_dir=args.cache_dir
            )
        except Exception:
            continue
        items.append((kpts, desc, isize))
    if not items:
        return {"pure_encoding_ms_per_image": float("nan"), "benchmark_images": 0}

    t0 = time.perf_counter()
    for kpts, desc, isize in items:
        if args.tiled:
            encoder.encode_tiling(kpts, desc, isize)
        else:
            encoder.encoder.encode(desc)
    seconds = time.perf_counter() - t0
    return {
        "pure_encoding_ms_per_image": 1000.0 * seconds / len(items),
        "benchmark_images": int(len(items)),
    }


def prepare(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_names: np.ndarray,
    conditions: Sequence[Condition],
) -> pd.DataFrame:
    """Train required codebooks and extract their encodings."""
    df_dev = train_df[~train_df["filename"].isin(val_names)].reset_index(drop=True)
    log(f"dev-train: {len(df_dev)} images; internal val: {len(val_names)} names")
    if len(df_dev) == 0:
        raise ValueError("Internal-validation filtering removed every training image")

    df_fit = limit_per_class(df_dev, args.per_class, args.seed)
    log(f"codebook fitting sample: {len(df_fit)} images "
        f"(per_class={args.per_class})")
    df_all = utils.loadDataTables(args.data_dir, args.img_dir)
    descs_cache: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []

    for K in args.required_k:
        codebook_path, meta = ensure_codebook(K, args, df_fit, descs_cache)
        enc_dir, enc_meta = ensure_encodings(codebook_path, args, df_all)
        bench = benchmark_encoding(codebook_path, df_all, args)
        rows.append(
            {
                "K": int(K),
                "codebook_path": str(codebook_path),
                "encoding_dir": str(enc_dir),
                **meta,
                **enc_meta,
                **bench,
                "encoding_ms_per_image": (
                    1000.0 * enc_meta["encoding_seconds"] / max(enc_meta["encoding_images"], 1)
                    if np.isfinite(enc_meta["encoding_seconds"])
                    else float("nan")
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    save_results(df, Path(args.results_dir) / f"part3_codebooks{tag_suffix(args.tag)}.csv")
    return df


# ---------------------------------------------------------------------------
# Loading encodings and image-level PCA
# ---------------------------------------------------------------------------
def load_encoding_matrix(
    enc_dir: Path,
    subset: str,
    tiled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a subset and return ``(X, names, categories)``.

    Rows are grouped by category in ``ALL_CATEGORIES`` order.  With
    ``tiled=False`` the four tile histograms are averaged into a global
    K-dimensional histogram.
    """
    xs, names, cats = [], [], []
    for cat in ALL_CATEGORIES:
        path = enc_dir / f"{cat}_{subset}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing encoding file: {path}")
        with np.load(path, allow_pickle=True) as data:
            hist = np.asarray(data["histograms"], dtype=np.float32)
            fnames = np.asarray(data["file_names"], dtype=str)
        if not tiled:
            hist = global_from_tiled(hist)
        if hist.shape[0] != len(fnames):
            raise ValueError(f"{path}: {hist.shape[0]} features but {len(fnames)} names")
        xs.append(hist)
        names.append(fnames)
        cats.append(np.full(len(fnames), cat, dtype=object))

    X = np.concatenate(xs, axis=0).astype(np.float32)
    n = np.concatenate(names, axis=0).astype(str)
    c = np.concatenate(cats, axis=0)
    return X, n, c


def fit_pca_on_dev(
    X_train: np.ndarray,
    names_train: np.ndarray,
    val_names: np.ndarray,
    d: int,
    seed: int,
    cache_dir: Path,
    cache_key: str,
    force: bool = False,
) -> Tuple[PCA, np.ndarray, np.ndarray, Dict[str, float]]:
    """Fit PCA on dev-train image histograms and transform train/test.

    The pipeline is: L2-normalise raw histograms -> PCA -> L2-normalise.
    Keeping the same normalisation for every condition makes the comparison
    meaningful.  PCA is fitted on dev-train only.
    """
    val_mask = np.isin(names_train, val_names)
    if not val_mask.any() or val_mask.all():
        raise ValueError("Internal validation split is empty or covers all training rows")

    X_train_l2 = l2norm(X_train)
    X_dev_l2 = X_train_l2[~val_mask]

    if d >= X_dev_l2.shape[1]:
        raise ValueError(f"PCA d={d} must be smaller than feature dimension {X_dev_l2.shape[1]}")
    if d > X_dev_l2.shape[0] - 1:
        raise ValueError(f"PCA d={d} exceeds n_dev - 1 = {X_dev_l2.shape[0] - 1}")

    cache_path = cache_dir / f"{cache_key}_d{d}.npz"
    t0 = time.perf_counter()
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=True) as data:
            pca = PCA(n_components=int(data["n_components"]), svd_solver="randomized",
                      random_state=seed)
            pca.components_ = data["components"]
            pca.mean_ = data["mean"]
            pca.explained_variance_ = data["explained_variance"]
            pca.explained_variance_ratio_ = data["explained_variance_ratio"]
            pca.singular_values_ = data["singular_values"]
            pca.n_features_in_ = data["mean"].shape[0]
            pca.n_samples_ = int(data["n_samples"])
            pca.n_components_ = int(data["n_components"])
            fit_s = float(data["fit_seconds"]) if "fit_seconds" in data else float("nan")
    else:
        pca = PCA(
            n_components=d,
            svd_solver="randomized",
            random_state=seed,
            whiten=False,
        )
        pca.fit(X_dev_l2)
        fit_s = time.perf_counter() - t0
        ensure_dir(cache_dir)
        np.savez_compressed(
            cache_path,
            components=pca.components_,
            mean=pca.mean_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            singular_values=pca.singular_values_,
            n_components=pca.n_components_,
            n_samples=pca.n_samples_,
            fit_seconds=fit_s,
        )

    X_train_pca = l2norm(pca.transform(X_train_l2))
    info = {
        "pca_fit_s": float(fit_s),
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "n_components": int(pca.n_components_),
        "n_dev": int(X_dev_l2.shape[0]),
    }
    return pca, X_train_pca, info


def transform_with_pca(X: np.ndarray, pca: PCA) -> np.ndarray:
    return l2norm(pca.transform(l2norm(X)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def deduplicate_multiclass(
    X: np.ndarray, names: np.ndarray, cats: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop images that occur in more than one category file.

    The dataset is multi-label: e.g. a person can also be listed as horse or
    motorbike.  A 6-way single-label classifier would otherwise see the same
    image twice with different labels.  Dropping multi-label images gives a
    clean, smaller single-label evaluation set; the one-vs-rest AP protocol is
    unaffected because its negatives are background only.
    """
    uniq, counts = np.unique(names, return_counts=True)
    keep = set(uniq[counts == 1].tolist())
    mask = np.array([n in keep for n in names], dtype=bool)
    return X[mask], names[mask], cats[mask]


def evaluate_condition(
    cond: Condition,
    X_train: np.ndarray,
    X_test: np.ndarray,
    names_train: np.ndarray,
    names_test: np.ndarray,
    cats_train: np.ndarray,
    cats_test: np.ndarray,
    val_names: np.ndarray,
    args: argparse.Namespace,
    collect_scores: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, Tuple[np.ndarray, np.ndarray]], Optional[pd.DataFrame]]:
    """Run one-vs-rest and (optionally) 6-way evaluation for one condition."""
    per_class_rows: List[Dict[str, object]] = []
    score_store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    condition_fields = {
        "condition": cond.name,
        "representation": cond.representation,
        "K": int(cond.K),
        "pca_dim": -1 if cond.pca_dim is None else int(cond.pca_dim),
        "dim": int(X_train.shape[1]),
    }

    for cat in CATEGORIES:
        Xtr_b, ytr_b, ntr_b = binary_subset(cat, X_train, names_train, cats_train)
        Xte_b, yte_b, nte_b = binary_subset(cat, X_test, names_test, cats_test)
        summary, _sweep, scores = evaluate_binary(
            cat,
            Xtr_b,
            ytr_b,
            ntr_b,
            Xte_b,
            yte_b,
            val_names,
            C_grid=args.c_grid,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        pred = (scores > 0).astype(int)
        y_true = (yte_b > 0).astype(int)
        acc = float(np.mean(pred == y_true))
        f1_pos = float(f1_score(y_true, pred, pos_label=1, zero_division=0))
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, pred, labels=[1], zero_division=0
        )
        row = {
            **condition_fields,
            **summary,
            "test_accuracy": acc,
            "test_f1": f1_pos,
            "test_precision": float(precision[0]),
            "test_recall": float(recall[0]),
        }
        per_class_rows.append(row)
        if collect_scores:
            score_store[cat] = (yte_b.copy(), scores.copy())

    per_class = pd.DataFrame(per_class_rows)
    overall: Dict[str, object] = {
        **condition_fields,
        "n_train": int(len(names_train)),
        "n_test": int(len(names_test)),
        "mAP": float(per_class["test_ap"].mean()),
        "mean_binary_accuracy": float(per_class["test_accuracy"].mean()),
        "mean_binary_f1": float(per_class["test_f1"].mean()),
        "mean_C": float(per_class["C"].mean()),
        "svm_fit_s": float(per_class["fit_s"].mean()),
        "infer_ms_per_img": float(per_class["infer_ms_per_img"].mean()),
        "all_converged": bool(per_class["converged"].all()),
        "num_converged": int(per_class["converged"].sum()),
        "storage_MB": float(storage_mb(X_train) + storage_mb(X_test)),
    }

    multiclass = None
    if not args.no_multiclass:
        Xtr_d, ntr_d, ctr_d = deduplicate_multiclass(X_train, names_train, cats_train)
        Xte_d, nte_d, cte_d = deduplicate_multiclass(X_test, names_test, cats_test)
        if len(ctr_d) and len(cte_d):
            try:
                summary, _sweep, per_class_multi = evaluate_multiclass(
                    Xtr_d,
                    ctr_d,
                    ntr_d,
                    Xte_d,
                    cte_d,
                    val_names,
                    C_grid=args.c_grid,
                    max_iter=args.max_iter,
                    seed=args.seed,
                )
                overall.update(
                    {
                        "multiclass_accuracy": float(summary["accuracy"]),
                        "multiclass_macro_f1": float(summary["macro_f1"]),
                        "multiclass_C": float(summary["C"]),
                        "multiclass_n_train": int(len(ctr_d)),
                        "multiclass_n_test": int(len(cte_d)),
                    }
                )
                labels = list(summary["labels"])
                conf = pd.DataFrame(summary["confusion"], index=labels, columns=labels)
                multiclass = conf
            except Exception as exc:  # keep the one-vs-rest results if this fails
                log(f"multiclass evaluation failed for {cond.name}: {exc}")
        else:
            log(f"multiclass evaluation skipped for {cond.name}: empty de-duplicated split")

    return per_class, overall, score_store, multiclass


def run_pca_sweep(
    args: argparse.Namespace,
    X_by_k: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    val_names: np.ndarray,
) -> pd.DataFrame:
    """Validation-only PCA-dimension sweep (no test-set selection)."""
    rows: List[Dict[str, object]] = []
    for K in args.required_k:
        X_train, _X_test, ntr, _nte, ctr, _cte = X_by_k[K]
        tiled = bool(args.tiled)
        raw_dim = X_train.shape[1]
        if not tiled and K not in [int(k) for k in args.k_grid]:
            continue
        for d in args.sweep_dims:
            d = int(d)
            if d >= raw_dim:
                continue
            try:
                pca, Xtr_pca, info = fit_pca_on_dev(
                    X_train, ntr, val_names, d, args.seed,
                    Path(args.pca_dir), f"{args.pca_cache_tag}_K{K}",
                    force=args.force_pca,
                )
            except ValueError as exc:
                log(f"skip sweep K={K} d={d}: {exc}")
                continue
            cat_rows = []
            for cat in CATEGORIES:
                Xtr_b, ytr_b, ntr_b = binary_subset(cat, Xtr_pca, ntr, ctr)
                vmask = np.isin(ntr_b, val_names)
                sweep = sweep_c(
                    Xtr_b, ytr_b, vmask,
                    C_grid=args.c_grid, max_iter=args.max_iter, seed=args.seed,
                )
                best = sweep.loc[sweep["val_ap"].idxmax()]
                cat_rows.append(
                    {
                        "K": K,
                        "pca_dim": d,
                        "dim": int(Xtr_pca.shape[1]),
                        "category": cat,
                        "val_ap": float(best["val_ap"]),
                        "train_ap": float(best["train_ap"]),
                        "C": float(best["C"]),
                        "explained_variance": float(info["explained_variance"]),
                        "pca_fit_s": float(info["pca_fit_s"]),
                        "converged": bool(best["converged"]),
                    }
                )
            cat_df = pd.DataFrame(cat_rows)
            rows.append(
                {
                    "K": K,
                    "pca_dim": d,
                    "dim": int(Xtr_pca.shape[1]),
                    "val_mAP": float(cat_df["val_ap"].mean()),
                    "mean_train_ap": float(cat_df["train_ap"].mean()),
                    "explained_variance": float(info["explained_variance"]),
                    "pca_fit_s": float(info["pca_fit_s"]),
                    "all_converged": bool(cat_df["converged"].all()),
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["K", "pca_dim"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    sweep: Optional[pd.DataFrame],
    variance: Optional[pd.DataFrame],
    figures_dir: Path,
    tag: str,
) -> None:
    ensure_dir(figures_dir)
    suffix = tag_suffix(tag)

    if len(summary):
        # mAP vs K, one line per PCA setting
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for (pca_dim,), group in summary.groupby(["pca_dim"]):
                group = group.sort_values("K")
                label = "no PCA" if int(pca_dim) < 0 else f"PCA {int(pca_dim)}"
                ax.plot(group["K"], group["mAP"], marker="o", label=label)
            ax.set_xlabel("Codebook size K")
            ax.set_ylabel("Test mAP")
            ax.set_title("Part 3: test mAP vs vocabulary size")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / f"part3_map_vs_k{suffix}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            log(f"could not plot mAP vs K: {exc}")

        # mAP vs storage
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(summary["storage_MB"], summary["mAP"])
            for _, row in summary.iterrows():
                ax.annotate(str(row["condition"]), (row["storage_MB"], row["mAP"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Feature storage (MB, float32)")
            ax.set_ylabel("Test mAP")
            ax.set_title("Part 3: accuracy vs storage")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / f"part3_map_vs_storage{suffix}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            log(f"could not plot mAP vs storage: {exc}")

    if len(per_class):
        try:
            pivot = per_class.pivot_table(
                index="condition", columns="category", values="test_ap", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(pivot))))
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                            color="white" if pivot.values[i, j] < 0.6 else "black", fontsize=7)
            fig.colorbar(im, ax=ax, label="Test AP")
            ax.set_title("Part 3: per-class AP")
            fig.tight_layout()
            fig.savefig(figures_dir / f"part3_per_class_ap{suffix}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            log(f"could not plot per-class heatmap: {exc}")

    if sweep is not None and len(sweep):
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for K, group in sweep.groupby("K"):
                group = group.sort_values("pca_dim")
                ax.plot(group["pca_dim"], group["val_mAP"], marker="o", label=f"K={K}")
            ax.set_xlabel("PCA output dimension d")
            ax.set_ylabel("Validation mAP")
            ax.set_title("Part 3: PCA dimension sweep (validation only)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / f"part3_map_vs_pca_dim{suffix}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            log(f"could not plot PCA sweep: {exc}")

    if variance is not None and len(variance):
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for K, group in variance.groupby("K"):
                group = group.sort_values("pca_dim")
                ax.plot(group["pca_dim"], group["explained_variance"], marker="o", label=f"K={K}")
            ax.set_xlabel("PCA output dimension d")
            ax.set_ylabel("Cumulative explained variance")
            ax.set_title("Part 3: PCA explained variance")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / f"part3_explained_variance{suffix}.png", dpi=160)
            plt.close(fig)
        except Exception as exc:
            log(f"could not plot explained variance: {exc}")


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------
def evaluate(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_names: np.ndarray,
    conditions: Sequence[Condition],
) -> None:
    # Sanity: all required encodings must exist.
    missing = []
    for K in args.required_k:
        enc_dir = resolve_encoding_dir(K, args)
        if not all(p.exists() for p in expected_encoding_files(enc_dir)):
            missing.append(str(enc_dir))
    if missing:
        raise FileNotFoundError(
            "Missing encodings:\n  " + "\n  ".join(missing) +
            "\nRun `python part3_vocab_pca.py --mode prepare` first (or --mode all)."
        )

    X_by_k: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for K in args.required_k:
        enc_dir = resolve_encoding_dir(K, args)
        Xtr, ntr, ctr = load_encoding_matrix(enc_dir, "train", tiled=args.tiled)
        Xte, nte, cte = load_encoding_matrix(enc_dir, "val", tiled=args.tiled)
        X_by_k[K] = (Xtr, Xte, ntr, nte, ctr, cte)
        log(
            f"K={K}: train={Xtr.shape}, test={Xte.shape}, "
            f"representation={'tiled' if args.tiled else 'global'}"
        )

    # Pre-compute PCA projections once per (K, d).  The PCA cache is shared
    # by the main grid and the sweep.
    pca_cache: Dict[Tuple[int, int], Tuple[PCA, np.ndarray, Dict[str, float]]] = {}
    variance_rows: List[Dict[str, object]] = []
    pca_dir = ensure_dir(Path(args.pca_dir))

    def get_pca(K: int, d: int):
        key = (K, d)
        if key not in pca_cache:
            Xtr, Xte, ntr, nte, ctr, cte = X_by_k[K]
            pca, Xtr_pca, info = fit_pca_on_dev(
                Xtr, ntr, val_names, d, args.seed, pca_dir,
                f"{args.pca_cache_tag}_K{K}", force=args.force_pca,
            )
            pca_cache[key] = (pca, Xtr_pca, info)
            variance_rows.append({"K": K, "pca_dim": d, **info})
        return pca_cache[key]

    summary_rows: List[Dict[str, object]] = []
    per_class_parts: List[pd.DataFrame] = []
    score_store: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    for cond in conditions:
        Xtr, Xte, ntr, nte, ctr, cte = X_by_k[cond.K]
        if cond.pca_dim is None:
            Xtr_c = l2norm(Xtr)
            Xte_c = l2norm(Xte)
            pca_info = {"pca_fit_s": 0.0, "explained_variance": float("nan"),
                        "n_components": int(Xtr.shape[1]), "n_dev": int(len(ntr))}
        else:
            pca, Xtr_pca, pca_info = get_pca(cond.K, cond.pca_dim)
            Xtr_c = Xtr_pca
            Xte_c = transform_with_pca(Xte, pca)

        log(f"evaluating {cond.name}: K={cond.K}, d={cond.pca_dim}, "
            f"dim={Xtr_c.shape[1]}")
        per_class, overall, scores, conf = evaluate_condition(
            cond, Xtr_c, Xte_c, ntr, nte, ctr, cte, val_names, args,
            collect_scores=True,
        )
        overall.update(
            {
                "pca_fit_s": float(pca_info["pca_fit_s"]),
                "explained_variance": float(pca_info["explained_variance"]),
            }
        )
        summary_rows.append(overall)
        per_class_parts.append(per_class)
        score_store[cond.name] = scores

        if conf is not None:
            conf_path = Path(args.results_dir) / f"part3_confusion_{cond.name}{tag_suffix(args.tag)}.csv"
            ensure_dir(conf_path.parent)
            conf.to_csv(conf_path)

    summary = pd.DataFrame(summary_rows).sort_values(["K", "pca_dim"]).reset_index(drop=True)
    per_class = pd.concat(per_class_parts, ignore_index=True)
    save_results(summary, Path(args.results_dir) / f"part3_condition_summary{tag_suffix(args.tag)}.csv")
    save_results(per_class, Path(args.results_dir) / f"part3_per_class{tag_suffix(args.tag)}.csv")

    variance = pd.DataFrame(variance_rows)
    if len(variance):
        variance = variance.drop_duplicates(["K", "pca_dim"]).sort_values(["K", "pca_dim"])
        save_results(variance, Path(args.results_dir) / f"part3_pca_variance{tag_suffix(args.tag)}.csv")

    sweep = None
    if not args.no_sweep and args.sweep_dims:
        log("running validation-only PCA dimension sweep")
        sweep = run_pca_sweep(args, X_by_k, val_names)
        save_results(sweep, Path(args.results_dir) / f"part3_pca_sweep{tag_suffix(args.tag)}.csv")

    # Cost table: merge condition summary with codebook/encoding/PCA metadata.
    cost = summary.copy()
    cost["codebook_storage_MB"] = cost["K"] * 128 * 4 / 1e6
    cost["pca_projection_storage_MB"] = np.where(
        cost["pca_dim"].astype(int) > 0,
        cost["pca_dim"].astype(float) * (
            (4 * cost["K"] if args.tiled else cost["K"]).astype(float)
        ) * 4 / 1e6,
        0.0,
    )
    codebook_csv = Path(args.results_dir) / f"part3_codebooks{tag_suffix(args.tag)}.csv"
    if codebook_csv.exists():
        cb = pd.read_csv(codebook_csv)
        keep = ["K", "fit_seconds", "encoding_ms_per_image", "num_samples",
                "num_images", "codebook_source"]
        cost = cost.merge(cb[keep], on="K", how="left")
    save_results(cost, Path(args.results_dir) / f"part3_cost{tag_suffix(args.tag)}.csv")

    if not args.no_plots:
        plot_results(
            summary, per_class,
            sweep if sweep is not None else None,
            variance if len(variance) else None,
            Path(args.figures_dir), args.tag,
        )

    # Console summary
    columns = ["condition", "K", "pca_dim", "dim", "mAP", "multiclass_accuracy",
               "multiclass_macro_f1", "storage_MB"]
    columns = [c for c in columns if c in summary.columns]
    print("\nPart 3 summary")
    print(summary[columns].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 3: large BoVW vocabularies + image-level PCA vs small vocabularies."
    )
    parser.add_argument("--mode", choices=("prepare", "evaluate", "all"),
                        default="all", help="Stage to run (default: all).")
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--img-dir", default="data/images")
    parser.add_argument("--encoders-dir", default="data/encoders")
    parser.add_argument("--encodings-dir", default="data/encodings")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--part3-dir", default="data/part3")
    parser.add_argument("--pca-dir", default="data/part3/pca")
    parser.add_argument("--cache-dir", default="cache/", help="Dense SIFT cache.")
    parser.add_argument("--tag", default="", help="Suffix for result file names.")

    parser.add_argument("--k-grid", type=int, nargs="+",
                        default=[128, 256, 512, 1024],
                        help="Vocabulary sizes to evaluate.")
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[128, 256],
                        help="PCA output dimensions for the main grid.")
    parser.add_argument("--sweep-dims", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256, 512],
                        help="PCA dimensions for the validation-only sweep.")
    parser.add_argument("--matched-dims", type=int, nargs="+", default=[128, 256],
                        help="Condition D final dimensions.")
    parser.add_argument("--tiled", action="store_true",
                        help="Use 2x2 spatial tiling (4K-D) instead of global K-D histograms.")
    parser.add_argument("--codebook-source", choices=("provided", "dev"),
                        default="provided",
                        help="Reuse existing codebooks when possible, or retrain all on dev-train.")
    parser.add_argument("--force-codebooks", action="store_true")
    parser.add_argument("--force-encodings", action="store_true")
    parser.add_argument("--force-pca", action="store_true")
    parser.add_argument("--per-class", type=int, default=150,
                        help="Images per class used to sample codebook descriptors.")
    parser.add_argument("--per-image", type=int, default=400,
                        help="Local descriptors sampled per image.")
    parser.add_argument("--kmeans-n-init", type=int, default=3,
                        help="k-means restarts; use the same value for every K.")
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)

    parser.add_argument("--val-names", default="results/part4_internal_val_names.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--no-multiclass", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.representation_tag = "tiled" if args.tiled else "global"
    args.val_hash = "unknown"
    args.pca_cache_tag = f"{args.representation_tag}_{args.codebook_source}_s{args.seed}"
    return args


def main() -> None:
    args = parse_args()
    for d in (args.results_dir, args.figures_dir, args.part3_dir, args.pca_dir,
              args.encoders_dir, args.encodings_dir):
        ensure_dir(Path(d))

    log(f"code directory: {CODE_DIR}")
    log(f"mode={args.mode}, representation={args.representation_tag}, "
        f"codebook_source={args.codebook_source}")

    train_df = utils.loadSamplesTable(
        args.data_dir, args.img_dir, subset="train", per_class=-1
    )
    val_names = get_internal_val_names(args, train_df)
    args.val_hash = split_hash(val_names)
    args.pca_cache_tag = (
        f"{args.representation_tag}_{args.codebook_source}_"
        f"{args.val_hash}_s{args.seed}"
    )
    df_dev = train_df[~train_df["filename"].isin(val_names)].reset_index(drop=True)
    conditions = build_conditions(args, n_train_dev=len(df_dev))
    log(f"required codebooks: {args.required_k}")
    log(f"conditions: {[c.name for c in conditions]}")
    condition_rows = [
        {
            "condition": c.name,
            "K": c.K,
            "pca_dim": -1 if c.pca_dim is None else c.pca_dim,
            "representation": c.representation,
            "alias_of": "",
        }
        for c in conditions
    ]
    for alias, target in args.condition_alias.items():
        condition_rows.append(
            {
                "condition": alias,
                "K": int(alias.split("_K")[-1]),
                "pca_dim": -1,
                "representation": f"bovw_{args.representation_tag}",
                "alias_of": target,
            }
        )
    pd.DataFrame(condition_rows).to_csv(
        Path(args.results_dir) / f"part3_conditions{tag_suffix(args.tag)}.csv", index=False
    )

    if args.mode in ("prepare", "all"):
        prepare(args, train_df, val_names, conditions)
    if args.mode in ("evaluate", "all"):
        evaluate(args, train_df, val_names, conditions)


if __name__ == "__main__":
    main()
