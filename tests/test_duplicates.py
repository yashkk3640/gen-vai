"""Near-duplicate detection. Camera rolls are full of five takes of the same thing."""

import numpy as np

from genvai.analysis import (
    group_duplicates,
    hamming,
    perceptual_hash,
    to_greyscale,
)

RNG = np.random.default_rng(7)


def _scene(seed: int, size: int = 128) -> np.ndarray:
    """A reproducible, structured image - smooth gradients, not noise.

    Noise would hash randomly and make every test pass for the wrong reason.
    """
    generator = np.random.default_rng(seed)
    coarse = generator.uniform(0, 255, (8, 8))
    return np.kron(coarse, np.ones((size // 8, size // 8)))


def _jpeg_like(frame: np.ndarray, strength: float = 6.0) -> np.ndarray:
    """Mild noise, standing in for a re-encode or a resize."""
    return np.clip(frame + RNG.normal(0, strength, frame.shape), 0, 255)


# ------------------------------------------------------------------------ hashing


def test_identical_frames_hash_identically() -> None:
    frame = _scene(1)
    assert perceptual_hash(frame) == perceptual_hash(frame)


def test_re_encoding_does_not_change_the_hash_much() -> None:
    """Two exports of the same photo must land in one group."""
    frame = _scene(2)
    assert hamming(perceptual_hash(frame), perceptual_hash(_jpeg_like(frame))) <= 6


def test_brightness_shift_does_not_change_the_hash() -> None:
    """dHash records relative brightness, so exposure changes should not matter."""
    frame = _scene(3)
    brighter = np.clip(frame * 0.7 + 60, 0, 255)
    assert hamming(perceptual_hash(frame), perceptual_hash(brighter)) <= 2


def test_different_scenes_hash_differently() -> None:
    assert hamming(perceptual_hash(_scene(10)), perceptual_hash(_scene(99))) > 12


def test_hash_is_64_bits() -> None:
    assert 0 <= perceptual_hash(_scene(4)) < 2**64


def test_empty_frame_hashes_to_zero() -> None:
    assert perceptual_hash(np.zeros((0, 0))) == 0


# ------------------------------------------------------------------------ hamming


def test_hamming_of_a_value_with_itself_is_zero() -> None:
    assert hamming(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming_counts_differing_bits() -> None:
    assert hamming(0b1010, 0b0101) == 4


# ----------------------------------------------------------------------- grouping


def test_burst_frames_group_together() -> None:
    base = _scene(20)
    hashes = {f"take{i}": perceptual_hash(_jpeg_like(base)) for i in range(4)}
    groups = group_duplicates(hashes)

    assert len(groups) == 4
    assert len(set(groups.values())) == 1, "all four takes are one shot"


def test_distinct_shots_are_left_ungrouped() -> None:
    """A one-of-a-kind photo should not be put in a group of itself."""
    hashes = {f"shot{i}": perceptual_hash(_scene(100 + i * 37)) for i in range(4)}
    assert group_duplicates(hashes) == {}


def test_two_bursts_form_two_groups() -> None:
    first, second = _scene(30), _scene(500)
    hashes = {
        "a1": perceptual_hash(_jpeg_like(first)),
        "a2": perceptual_hash(_jpeg_like(first)),
        "b1": perceptual_hash(_jpeg_like(second)),
        "b2": perceptual_hash(_jpeg_like(second)),
    }
    groups = group_duplicates(hashes)
    assert len(set(groups.values())) == 2
    assert groups["a1"] == groups["a2"]
    assert groups["b1"] == groups["b2"]
    assert groups["a1"] != groups["b1"]


def test_grouping_is_order_independent() -> None:
    """Import order must not change which shots are considered the same."""
    base = _scene(40)
    hashes = {f"x{i}": perceptual_hash(_jpeg_like(base)) for i in range(3)}
    forward = group_duplicates(hashes)
    backward = group_duplicates(dict(reversed(list(hashes.items()))))
    assert forward == backward


def test_chaining_is_avoided() -> None:
    """A matches B and B matches C, but A and C are plainly different.

    Comparing against representatives rather than every member stops a whole afternoon
    quietly collapsing into one group.
    """
    groups = group_duplicates({"a": 0b0, "b": 0b11111, "c": 0b1111111111}, threshold=5)
    assert groups.get("a") != groups.get("c") or "c" not in groups


def test_empty_input_is_empty_output() -> None:
    assert group_duplicates({}) == {}


def test_greyscale_conversion_feeds_hashing() -> None:
    """The real path: colour frame in, stable fingerprint out."""
    colour = RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    frame = to_greyscale(colour)
    assert perceptual_hash(frame) == perceptual_hash(to_greyscale(colour))
