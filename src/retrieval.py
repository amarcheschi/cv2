"""Part 5: image retrieval with BoVW and VLAD.

The fixed corpus is built from all unique training image IDs. Some source class
lists overlap, so each image is stored once and is relevant when it has a
motorbike training annotation. Queries are a small, visually varied selection
of motorbikes from the validation split.

Run from the project root with:
    python -m src.retrieval
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
ENCODINGS = ROOT / "data" / "encodings"
OUTPUT = ROOT / "outputs" / "part5_motorbike"
CATEGORIES = ("aeroplane", "background", "car", "horse", "motorbike", "person")
TARGET = "motorbike"

# Selected after visually inspecting the validation motorbike contact sheets.
# The queries vary in viewpoint, scale, setting, and whether people are present.
QUERY_IDS = (
    "000427", "001201", "001617", "002567", "003760",
    "004275", "005996", "006532", "007647", "009781",
)
METHODS = {
    "BoVW": "bovw_K512_C0",
    "VLAD": "vlad_K32_C80",
}


def load_archive(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["file_names"]]
        vectors = np.asarray(archive["histograms"], dtype=np.float32)
    return names, vectors


def load_corpus(encoding_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    vectors_by_name: dict[str, np.ndarray] = {}
    relevant_names: set[str] = set()
    compressed_bytes = 0
    for category in CATEGORIES:
        path = encoding_dir / f"{category}_train.npz"
        category_names, vectors = load_archive(path)
        if category == TARGET:
            relevant_names.update(category_names)
        for name, vector in zip(category_names, vectors):
            if name in vectors_by_name and not np.array_equal(vectors_by_name[name], vector):
                raise ValueError(f"Inconsistent features for repeated corpus image {name}")
            vectors_by_name.setdefault(name, vector)
        compressed_bytes += path.stat().st_size
    names = list(vectors_by_name)
    vectors = np.stack([vectors_by_name[name] for name in names])
    labels = np.asarray([int(name in relevant_names) for name in names], dtype=np.int8)
    return names, vectors, labels, compressed_bytes


def load_queries(encoding_dir: Path) -> tuple[list[str], np.ndarray, int]:
    path = encoding_dir / f"{TARGET}_val.npz"
    names, vectors = load_archive(path)
    positions = {name: i for i, name in enumerate(names)}
    missing = [name for name in QUERY_IDS if name not in positions]
    if missing:
        raise ValueError(f"Selected query IDs missing from {path}: {missing}")
    indices = [positions[name] for name in QUERY_IDS]
    return [names[i] for i in indices], vectors[indices], path.stat().st_size


def check_no_exact_query_duplicates(corpus_names: list[str], query_names: list[str]) -> None:
    """Check selected queries against the corpus using exact file hashes."""
    corpus_hashes: set[str] = set()
    for name in corpus_names:
        image_path = ROOT / "data" / "images" / f"{name}.jpg"
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        corpus_hashes.add(digest)
    duplicates = []
    for name in query_names:
        image_path = ROOT / "data" / "images" / f"{name}.jpg"
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in corpus_hashes:
            duplicates.append(name)
    if duplicates:
        raise ValueError(f"Selected queries have exact duplicates in corpus: {duplicates}")


def evaluate(encoding_dir: Path, method: str) -> tuple[dict, list[dict], list[dict]]:
    corpus_names, corpus_vectors, corpus_labels, corpus_archive_bytes = load_corpus(encoding_dir)
    query_names, query_vectors, query_archive_bytes = load_queries(encoding_dir)
    check_no_exact_query_duplicates(corpus_names, query_names)

    # BoVW uses histogram intersection after L1 normalization; VLAD uses cosine.
    if method == "BoVW":
        corpus_features = normalize(corpus_vectors, norm="l1")
        query_features = normalize(query_vectors, norm="l1")
        similarity_metric = "histogram intersection (L1)"
    else:
        corpus_features = normalize(corpus_vectors, norm="l2")
        query_features = normalize(query_vectors, norm="l2")
        similarity_metric = "cosine"
    rank_start = time.perf_counter()
    if method == "BoVW":
        scores = np.stack([
            np.minimum(query_features[i][None, :], corpus_features).sum(axis=1)
            for i in range(len(query_names))
        ])
    else:
        scores = query_features @ corpus_features.T
    ranking_seconds = time.perf_counter() - rank_start

    per_query_ap: list[float] = []
    per_query_p: dict[int, list[float]] = {5: [], 10: [], 20: []}
    per_query_r: dict[int, list[float]] = {5: [], 10: [], 20: []}
    result_rows: list[dict] = []
    example_rows: list[dict] = []
    relevant_total = int(corpus_labels.sum())

    for qi, query_name in enumerate(query_names):
        order = np.argsort(-scores[qi], kind="stable")
        ranked_labels = corpus_labels[order]
        ap = float(average_precision_score(corpus_labels, scores[qi]))
        per_query_ap.append(ap)
        row: dict[str, object] = {"method": method, "query_id": query_name, "AP": ap}
        for k in per_query_p:
            hits = int(ranked_labels[:k].sum())
            precision = hits / min(k, len(corpus_names))
            recall = hits / relevant_total if relevant_total else float("nan")
            per_query_p[k].append(precision)
            per_query_r[k].append(recall)
            row[f"P@{k}"] = precision
            row[f"R@{k}"] = recall
        result_rows.append(row)
        for rank, corpus_index in enumerate(order[:20], start=1):
            example_rows.append({
                "method": method,
                "query_id": query_name,
                "rank": rank,
                "corpus_id": corpus_names[corpus_index],
                "relevant": int(corpus_labels[corpus_index]),
                "cosine_similarity": float(scores[qi, corpus_index]),
            })

    feature_dim = int(corpus_vectors.shape[1])
    compressed_index = io.BytesIO()
    np.savez_compressed(
        compressed_index,
        corpus_ids=np.asarray(corpus_names),
        corpus_features=corpus_vectors,
        corpus_relevant=corpus_labels,
        query_ids=np.asarray(query_names),
        query_features=query_vectors,
    )
    summary: dict[str, object] = {
        "method": method,
        "similarity_metric": similarity_metric,
        "corpus_images": len(corpus_names),
        "relevant_corpus_images": relevant_total,
        "query_images": len(query_names),
        "feature_dimension": feature_dim,
        "feature_extraction_time": "not measured (precomputed features; runtime lacks scikit-image)",
        "mean_AP_mAP": float(np.mean(per_query_ap)),
        "mean_query_ranking_ms": ranking_seconds * 1000 / len(query_names),
        "feature_matrix_memory_MiB": (corpus_vectors.nbytes + query_vectors.nbytes) / 2**20,
        "encoding_archives_disk_MiB": (corpus_archive_bytes + query_archive_bytes) / 2**20,
        "deduplicated_index_disk_MiB": len(compressed_index.getvalue()) / 2**20,
    }
    for k in per_query_p:
        summary[f"Precision@{k}"] = float(np.mean(per_query_p[k]))
        summary[f"Recall@{k}"] = float(np.mean(per_query_r[k]))
    return summary, result_rows, example_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(summaries: list[dict], examples: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    # Compact chart for the report: retrieval quality at the requested cutoffs.
    metric_names = ["mean_AP_mAP", "Precision@5", "Precision@10", "Precision@20"]
    labels = ["mAP", "P@5", "P@10", "P@20"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, summary in enumerate(summaries):
        values = [float(summary[key]) for key in metric_names]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=str(summary["method"]))
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.25)
    ax.set_ylabel("Score")
    ax.set_title("Motorbike retrieval: BoVW vs VLAD")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "metrics_comparison.png", dpi=180)
    plt.close(fig)

    # Show a failure case and a stronger case for each method, with labels.
    chosen_queries = ("000427", "009781")
    fig, axes = plt.subplots(4, 6, figsize=(14, 9))
    for row, (method, query_id) in enumerate(
        (method, query_id) for method in ("BoVW", "VLAD") for query_id in chosen_queries
    ):
        query_path = ROOT / "data" / "images" / f"{query_id}.jpg"
        with Image.open(query_path) as im:
            axes[row, 0].imshow(im.convert("RGB"))
        axes[row, 0].set_title(f"Consulta {query_id}\n{method}", fontsize=9)
        axes[row, 0].axis("off")
        selected = [r for r in examples if r["method"] == method and r["query_id"] == query_id]
        for column, result in enumerate(selected[:5], start=1):
            image_path = ROOT / "data" / "images" / f"{result['corpus_id']}.jpg"
            with Image.open(image_path) as im:
                axes[row, column].imshow(im.convert("RGB"))
            label = "relevante" if result["relevant"] else "irrelevante"
            axes[row, column].set_title(f"#{result['rank']} {label}\n{result['corpus_id']}", fontsize=8)
            axes[row, column].axis("off")
    fig.suptitle("Ejemplos de recuperación: consulta + top 5", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "retrieval_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    query_rows: list[dict] = []
    example_rows: list[dict] = []
    for method, folder in METHODS.items():
        summary, per_query, examples = evaluate(ENCODINGS / folder, method)
        summaries.append(summary)
        query_rows.extend(per_query)
        example_rows.extend(examples)

    write_csv(OUTPUT / "metrics_summary.csv", summaries)
    write_csv(OUTPUT / "metrics_per_query.csv", query_rows)
    write_csv(OUTPUT / "top20_examples.csv", example_rows)
    corpus_names, _, corpus_labels, _ = load_corpus(ENCODINGS / METHODS["BoVW"])
    write_csv(OUTPUT / "corpus_ground_truth.csv", [
        {"image_id": name, "relevant_motorbike": int(label)}
        for name, label in zip(corpus_names, corpus_labels)
    ])
    write_csv(OUTPUT / "query_set.csv", [
        {"query_id": name, "category": TARGET, "split": "validation", "selected_after_visual_review": True}
        for name in QUERY_IDS
    ])
    save_figures(summaries, example_rows)

    print("Part 5 retrieval comparison")
    for summary in summaries:
        print(
            f"{summary['method']}: mAP={summary['mean_AP_mAP']:.4f}, "
            f"P@5={summary['Precision@5']:.4f}, "
            f"P@10={summary['Precision@10']:.4f}, "
            f"P@20={summary['Precision@20']:.4f}, "
            f"ranking={summary['mean_query_ranking_ms']:.2f} ms/query, "
            f"dim={summary['feature_dimension']}"
        )
    print(f"Saved results to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
