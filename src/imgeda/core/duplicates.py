"""Hash-based duplicate clustering."""

from __future__ import annotations

from collections import defaultdict

from imgeda.core.hasher import hamming_distance
from imgeda.models.manifest import ImageRecord

# Skip buckets larger than this to avoid O(n^2) blowup on hot sub-hashes
_MAX_BUCKET_SIZE = 500


def find_exact_duplicates(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Group records by exact phash match. Returns groups with 2+ members."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        if rec.phash and not rec.is_corrupt:
            groups[rec.phash].append(rec)
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_near_duplicates(
    records: list[ImageRecord],
    hamming_threshold: int = 8,
) -> list[list[ImageRecord]]:
    """Find near-duplicate groups using sub-hash bucketing to avoid O(n^2).

    Strategy: split each phash hex string into 4 sub-hashes, bucket by each sub-hash,
    then compare within buckets only. Buckets exceeding _MAX_BUCKET_SIZE are skipped
    to prevent quadratic blowup.
    """
    # Filter to hashable records
    hashable = [(rec, rec.phash) for rec in records if rec.phash and not rec.is_corrupt]
    if not hashable:
        return []

    # Build sub-hash buckets (4 quarters of the hex string)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, (_, phash) in enumerate(hashable):
        chunk_size = max(1, len(phash) // 4)
        for i in range(4):
            sub = phash[i * chunk_size : (i + 1) * chunk_size]
            buckets[f"{i}:{sub}"].append(idx)

    # Find candidate pairs within buckets
    pairs: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) < 2 or len(indices) > _MAX_BUCKET_SIZE:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                pair = (min(a, b), max(a, b))
                if pair not in pairs:
                    dist = hamming_distance(hashable[a][1], hashable[b][1])
                    if dist <= hamming_threshold:
                        pairs.add(pair)

    # Union-find to cluster connected pairs
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    # Group by root
    clusters: dict[int, list[int]] = defaultdict(list)
    seen = set()
    for a, b in pairs:
        for x in (a, b):
            if x not in seen:
                clusters[find(x)].append(x)
                seen.add(x)

    return [[hashable[idx][0] for idx in group] for group in clusters.values() if len(group) > 1]
