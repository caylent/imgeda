"""Image embedding computation using CLIP (optional dependency).

Requires: pip install imgeda[embeddings]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _check_deps() -> None:
    """Check that embedding dependencies are available."""
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        msg = (
            "Embedding support requires open_clip and torch.\n"
            "Install with: pip install imgeda[embeddings]"
        )
        raise ImportError(msg) from e


def compute_embeddings(
    image_paths: list[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 32,
    device: str | None = None,
    progress_callback: Any = None,
) -> NDArray[np.float32]:
    """Compute CLIP embeddings for a list of images.

    Args:
        image_paths: List of image file paths
        model_name: OpenCLIP model name
        pretrained: Pretrained weights name
        batch_size: Batch size for inference
        device: Torch device ('cuda', 'cpu', 'mps'). Auto-detected if None.
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        (N, D) numpy array of normalized embeddings
    """
    _check_deps()

    import torch
    import open_clip
    from PIL import Image

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    all_embeddings: list[NDArray[np.float32]] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception:
                # Use zero vector for unreadable images
                images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy().astype(np.float32))

        if progress_callback:
            progress_callback(min(start + batch_size, len(image_paths)), len(image_paths))

    return np.concatenate(all_embeddings, axis=0)


def compute_umap_projection(
    embeddings: NDArray[np.float32],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> NDArray[np.float32]:
    """Project embeddings to 2D using UMAP.

    Requires: pip install umap-learn
    """
    try:
        import umap
    except ImportError as e:
        msg = "UMAP requires umap-learn: pip install umap-learn"
        raise ImportError(msg) from e

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric="cosine",
    )
    result: NDArray[np.float32] = reducer.fit_transform(embeddings).astype(np.float32)
    return result


def find_outliers(
    embeddings: NDArray[np.float32],
    threshold_percentile: float = 5.0,
) -> NDArray[np.bool_]:
    """Find outlier images based on cosine distance from centroid.

    Returns:
        Boolean mask where True = outlier
    """
    centroid = embeddings.mean(axis=0, keepdims=True)
    centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
    similarities = (embeddings @ centroid.T).flatten()
    threshold = np.percentile(similarities, threshold_percentile)
    mask: NDArray[np.bool_] = similarities < threshold
    return mask


def find_semantic_duplicates(
    embeddings: NDArray[np.float32],
    threshold: float = 0.95,
) -> list[tuple[int, int, float]]:
    """Find semantically similar image pairs.

    Returns:
        List of (idx_a, idx_b, similarity) tuples above threshold.
    """
    n = len(embeddings)
    duplicates: list[tuple[int, int, float]] = []

    # Process in chunks to avoid memory issues
    chunk_size = 1000
    for i in range(0, n, chunk_size):
        chunk = embeddings[i : i + chunk_size]
        # Compute similarities with all following images
        for j in range(i, n, chunk_size):
            other = embeddings[j : j + chunk_size]
            sims = chunk @ other.T

            for ci in range(len(chunk)):
                start_cj = ci + 1 if i == j else 0
                for cj in range(start_cj, len(other)):
                    if sims[ci, cj] >= threshold:
                        duplicates.append((i + ci, j + cj, float(sims[ci, cj])))

    return duplicates


def save_embeddings(
    embeddings: NDArray[np.float32],
    paths: list[str],
    output_path: str,
) -> None:
    """Save embeddings alongside manifest as .npz file."""
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        paths=np.array(paths),
    )


def load_embeddings(path: str) -> tuple[NDArray[np.float32], list[str]]:
    """Load embeddings from .npz file."""
    data = np.load(path)
    return data["embeddings"], data["paths"].tolist()
