"""Perceptual hashing wrappers."""

from __future__ import annotations

from PIL import Image
import imagehash


def compute_phash(img: Image.Image, hash_size: int = 16) -> str:
    return str(imagehash.phash(img, hash_size=hash_size))


def compute_dhash(img: Image.Image, hash_size: int = 16) -> str:
    return str(imagehash.dhash(img, hash_size=hash_size))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    h1 = imagehash.hex_to_hash(hash_a)
    h2 = imagehash.hex_to_hash(hash_b)
    return int(h1 - h2)
