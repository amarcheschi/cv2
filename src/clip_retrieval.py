"""CLIP text-to-image retrieval prompt comparison for assignment Part 5.

Run from the project root in the cvc-a2 environment:
    python -m src.clip_retrieval

The corpus is the set of unique training image IDs. A corpus image is relevant
when its training annotations include ``motorbike``. Prompts are compared on
the same corpus and labels; this is a prompt-sensitivity experiment, not a
trained classifier.
"""

from __future__ import annotations

import csv
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from deep_models import getModel


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT = ROOT / "outputs" / "part5_motorbike"
FEATURE_CACHE = OUTPUT / "clip_vit_b_32_train_embeddings.npz"
TARGET = "motorbike"
PROMPTS = (
    "a photo of a motorbike",
    "a photo of a motorcycle",
    "a photo of a motorcycle with a rider",
    "a photo of a motorcycle on a race track",
    "a close-up photo of a motorcycle",
)


def load_train_corpus() -> tuple[list[str], np.ndarray]:
    annotations: dict[str, set[str]] = {}
    for path in sorted(SPLITS.glob("*_train.txt")):
        category = path.name.removesuffix("_train.txt")
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                annotations.setdefault(name, set()).add(category)
    names = sorted(annotations)
    labels = np.asarray([int(TARGET in annotations[name]) for name in names], dtype=np.int8)
    return names, labels


def extract_image_embeddings(names: list[str], preprocess, model, device: torch.device) -> np.ndarray:
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE) as cache:
            if cache["image_ids"].tolist() == names:
                return np.asarray(cache["image_features"], dtype=np.float32)

    batch_size = 32
    outputs: list[np.ndarray] = []
    for start in tqdm(range(0, len(names), batch_size), desc="CLIP image embeddings", ncols=100):
        batch_names = names[start:start + batch_size]
        tensors = []
        for name in batch_names:
            path = IMAGE_DIR / f"{name}.jpg"
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        images = torch.stack(tensors).to(device)
        with torch.inference_mode():
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
        outputs.append(features.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(outputs, axis=0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, image_ids=np.asarray(names), image_features=embeddings)
    return embeddings


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    return float(labels[order].mean())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, labels = load_train_corpus()
    model, preprocess = getModel("clip_vit_b_32", device=device)
    image_features = extract_image_embeddings(names, preprocess, model, device)

    tokens = clip.tokenize(list(PROMPTS)).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = (text_features @ torch.from_numpy(image_features).to(device).T).cpu().numpy()

    results: list[dict] = []
    rankings: list[dict] = []
    for index, prompt in enumerate(PROMPTS):
        prompt_scores = scores[index]
        results.append({
            "prompt": prompt,
            "corpus_images": len(names),
            "relevant_motorbikes": int(labels.sum()),
            "mAP": float(average_precision_score(labels, prompt_scores)),
            "P@5": precision_at_k(labels, prompt_scores, 5),
            "P@10": precision_at_k(labels, prompt_scores, 10),
            "P@20": precision_at_k(labels, prompt_scores, 20),
        })
        order = np.argsort(-prompt_scores, kind="stable")
        for rank, image_index in enumerate(order[:20], start=1):
            rankings.append({
                "prompt": prompt,
                "rank": rank,
                "image_id": names[image_index],
                "relevant_motorbike": int(labels[image_index]),
                "cosine_similarity": float(prompt_scores[image_index]),
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "clip_prompt_results.csv", results)
    write_csv(OUTPUT / "clip_prompt_top20.csv", rankings)
    print(f"CLIP ViT-B/32 text-to-image retrieval on {len(names)} unique training images")
    print(f"Relevant motorbike images: {int(labels.sum())}; device: {device}")
    for row in results:
        print(
            f"{row['prompt']}: mAP={row['mAP']:.4f}, P@5={row['P@5']:.4f}, "
            f"P@10={row['P@10']:.4f}, P@20={row['P@20']:.4f}"
        )
    print(f"Saved results to {(OUTPUT / 'clip_prompt_results.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
