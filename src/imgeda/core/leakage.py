"""Cross-split data leakage detection via perceptual hashing."""

from __future__ import annotations

from typing import Any

from imgeda.models.manifest import ImageRecord


def detect_leakage(
    splits: dict[str, list[ImageRecord]],
    hamming_threshold: int = 8,
) -> list[dict[str, Any]]:
    """Find images that appear in multiple splits.

    Args:
        splits: Mapping of split name -> records
        hamming_threshold: Hamming distance threshold for near-match (0 = exact only)

    Returns:
        List of dicts with 'path', 'phash', 'found_in' keys.
    """
    # Build phash -> (split_name, path) index
    hash_index: dict[str, list[tuple[str, str]]] = {}
    for split_name, records in splits.items():
        for r in records:
            if r.phash:
                hash_index.setdefault(r.phash, []).append((split_name, r.path))

    leaked: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    # Exact matches: same phash in different splits
    for phash, entries in hash_index.items():
        split_names = {s for s, _ in entries}
        if len(split_names) > 1:
            for split_name, path in entries:
                if path not in seen_paths:
                    seen_paths.add(path)
                    leaked.append(
                        {
                            "path": path,
                            "phash": phash,
                            "found_in": sorted(split_names),
                            "match_type": "exact",
                        }
                    )

    # Near matches if threshold > 0
    if hamming_threshold > 0:
        _detect_near_leakage(splits, hamming_threshold, leaked, seen_paths)

    return sorted(leaked, key=lambda x: x["path"])


def _hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return bin(i1 ^ i2).count("1")
    except (ValueError, TypeError):
        return 999


def _detect_near_leakage(
    splits: dict[str, list[ImageRecord]],
    threshold: int,
    leaked: list[dict[str, Any]],
    seen_paths: set[str],
) -> None:
    """Find near-duplicate images across splits using sub-hash bucketing."""
    # Build per-split hash lists
    split_hashes: list[tuple[str, list[tuple[str, str]]]] = []
    for split_name, records in splits.items():
        hashes = [(r.phash, r.path) for r in records if r.phash]
        split_hashes.append((split_name, hashes))

    if len(split_hashes) < 2:
        return

    # Compare each pair of splits
    for i in range(len(split_hashes)):
        for j in range(i + 1, len(split_hashes)):
            name_a, hashes_a = split_hashes[i]
            name_b, hashes_b = split_hashes[j]

            # Sub-hash bucketing for efficiency
            buckets_b: dict[str, list[tuple[str, str]]] = {}
            quarter = max(1, len(hashes_b[0][0]) // 4) if hashes_b else 4
            for h, p in hashes_b:
                for k in range(4):
                    sub = h[k * quarter : (k + 1) * quarter]
                    buckets_b.setdefault(sub, []).append((h, p))

            for h_a, p_a in hashes_a:
                if p_a in seen_paths:
                    continue
                candidates: set[tuple[str, str]] = set()
                for k in range(4):
                    sub = h_a[k * quarter : (k + 1) * quarter]
                    if sub in buckets_b:
                        candidates.update(buckets_b[sub])

                for h_b, p_b in candidates:
                    if p_b in seen_paths:
                        continue
                    if h_a == h_b:
                        continue  # Already caught by exact match
                    if _hamming_distance(h_a, h_b) <= threshold:
                        seen_paths.add(p_a)
                        seen_paths.add(p_b)
                        leaked.append(
                            {
                                "path": p_a,
                                "phash": h_a,
                                "found_in": sorted([name_a, name_b]),
                                "match_type": "near",
                                "matched_path": p_b,
                            }
                        )
                        break
