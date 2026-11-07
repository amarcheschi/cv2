"""Compare VLAD retrieval with 128-D SIFT descriptors and PCA to 80-D.

Run from the project root in the cvc-a2 environment:
    python -m src.compare_vlad_pca

The no-PCA codebook must first be trained with the same sampling settings as
the supplied PCA model:
    python -m src.train -e vlad -k 32 -c 0 -i 150 -n 400 --seed 42 \
        --output-dir outputs/part5_motorbike
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.encoders import ImageEncoder


ROOT = Path(__file__).resolve().parents[1]
ENCODINGS = ROOT / "data" / "encodings"
MODEL_NO_PCA = ROOT / "outputs" / "part5_motorbike" / "vlad_K32_C0.npz"
OUTPUT = ROOT / "outputs" / "part5_motorbike"
FEATURE_CACHE = Path("/tmp/part5_pca/vlad_K32_C0_features.npz")
CATEGORIES = ("aeroplane", "background", "car", "horse", "motorbike", "person")
TARGET = "motorbike"
QUERY_IDS = (
    "000427", "001201", "001617", "002567", "003760",
    "004275", "005996", "006532", "007647", "009781",
)


def load_c80() -> tuple[list[str], np.ndarray, np.ndarray, list[str], np.ndarray]:
    vectors_by_name: dict[str, np.ndarray] = {}
    relevant_names: set[str] = set()
    directory = ENCODINGS / "vlad_K32_C80"
    for category in CATEGORIES:
        with np.load(directory / f"{category}_train.npz") as archive:
            names = [str(x) for x in archive["file_names"]]
            vectors = np.asarray(archive["histograms"], dtype=np.float32)
        if category == TARGET:
            relevant_names.update(names)
        for name, vector in zip(names, vectors):
            if name in vectors_by_name and not np.array_equal(vectors_by_name[name], vector):
                raise ValueError(f"Inconsistent PCA features for repeated image {name}")
            vectors_by_name.setdefault(name, vector)

    corpus_names = list(vectors_by_name)
    corpus_vectors = np.stack([vectors_by_name[name] for name in corpus_names])
    labels = np.asarray([int(name in relevant_names) for name in corpus_names], dtype=np.int8)

    with np.load(directory / f"{TARGET}_val.npz") as archive:
        val_names = [str(x) for x in archive["file_names"]]
        val_vectors = np.asarray(archive["histograms"], dtype=np.float32)
    positions = {name: i for i, name in enumerate(val_names)}
    missing = [name for name in QUERY_IDS if name not in positions]
    if missing:
        raise ValueError(f"Selected queries missing from validation encodings: {missing}")
    query_vectors = val_vectors[[positions[name] for name in QUERY_IDS]]
    return corpus_names, corpus_vectors, labels, list(QUERY_IDS), query_vectors


def encode_no_pca(corpus_names: list[str], query_names: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    if not MODEL_NO_PCA.exists():
        raise FileNotFoundError(f"Train the no-PCA VLAD model first: {MODEL_NO_PCA}")
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as cache:
            if cache["corpus_ids"].tolist() == corpus_names and cache["query_ids"].tolist() == query_names:
                return (
                    np.asarray(cache["corpus_features"], dtype=np.float32),
                    np.asarray(cache["query_features"], dtype=np.float32),
                    float(cache["encoding_seconds"]),
                )
    encoder = ImageEncoder(MODEL_NO_PCA)
    all_names = corpus_names + query_names
    paths = [ROOT / "data" / "images" / f"{name}.jpg" for name in all_names]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = list(tqdm(
            pool.map(encoder.encode, paths),
            total=len(paths),
            desc="Encoding VLAD without PCA",
            ncols=100,
        ))
    elapsed = time.perf_counter() - start
    matrix = np.stack(features).astype(np.float32)
    corpus_features = matrix[:len(corpus_names)]
    query_features = matrix[len(corpus_names):]
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURE_CACHE,
        corpus_ids=np.asarray(corpus_names),
        query_ids=np.asarray(query_names),
        corpus_features=corpus_features,
        query_features=query_features,
        encoding_seconds=np.asarray(elapsed),
    )
    return corpus_features, query_features, elapsed


def score_method(
    method: str,
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    labels: np.ndarray,
    query_names: list[str],
) -> tuple[dict, list[dict]]:
    corpus_vectors = normalize(corpus_vectors, norm="l2")
    query_vectors = normalize(query_vectors, norm="l2")
    scores = query_vectors @ corpus_vectors.T
    per_query: list[dict] = []
    aps: list[float] = []
    for index, query_name in enumerate(query_names):
        ranked = np.argsort(-scores[index], kind="stable")
        ap = float(average_precision_score(labels, scores[index]))
        aps.append(ap)
        row: dict[str, object] = {"method": method, "query_id": query_name, "AP": ap}
        for k in (5, 10, 20):
            row[f"P@{k}"] = float(labels[ranked[:k]].mean())
        per_query.append(row)
    summary: dict[str, object] = {
        "method": method,
        "corpus_images": len(labels),
        "relevant_motorbikes": int(labels.sum()),
        "queries": len(query_names),
        "feature_dimension": int(corpus_vectors.shape[1]),
        "mean_AP_mAP": float(np.mean(aps)),
        "Precision@5": float(np.mean([row["P@5"] for row in per_query])),
        "Precision@10": float(np.mean([row["P@10"] for row in per_query])),
        "Precision@20": float(np.mean([row["P@20"] for row in per_query])),
    }
    return summary, per_query


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    corpus_names, vectors_pca, labels, query_names, queries_pca = load_c80()
    vectors_no_pca, queries_no_pca, extraction_seconds = encode_no_pca(corpus_names, query_names)
    summaries: list[dict] = []
    query_rows: list[dict] = []
    for method, corpus, queries in (
        ("VLAD + PCA (80-D)", vectors_pca, queries_pca),
        ("VLAD without PCA (128-D)", vectors_no_pca, queries_no_pca),
    ):
        summary, per_query = score_method(method, corpus, queries, labels, query_names)
        summaries.append(summary)
        query_rows.extend(per_query)
    summaries[1]["no_PCA_encoding_time_seconds"] = round(extraction_seconds, 2)
    write_csv(OUTPUT / "vlad_pca_comparison.csv", summaries)
    write_csv(OUTPUT / "vlad_pca_per_query.csv", query_rows)
    print("VLAD PCA comparison: same corpus, labels, and 10 queries")
    for row in summaries:
        print(
            f"{row['method']}: mAP={row['mean_AP_mAP']:.4f}, "
            f"P@5={row['Precision@5']:.4f}, P@10={row['Precision@10']:.4f}, "
            f"P@20={row['Precision@20']:.4f}, dim={row['feature_dimension']}"
        )
    print(f"No-PCA corpus/query encoding time: {extraction_seconds:.2f} s")
    print(f"Saved summary to {(OUTPUT / 'vlad_pca_comparison.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
