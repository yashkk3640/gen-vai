"""Beat detection, against synthetic tracks whose tempo is known exactly.

A click track is the one audio fixture where the right answer is not a matter of
opinion, which makes this the part of the audio work that can actually be pinned down.
"""

import numpy as np
import pytest

from genvai.beats import (
    MAX_BPM,
    MIN_BPM,
    beat_grid,
    detect,
    estimate_tempo,
    onset_envelope,
    snap,
)

RATE = 22050


def _click_track(bpm: float, seconds: float = 12.0, *, noise: float = 0.0) -> np.ndarray:
    """Short percussive bursts at a fixed tempo, with optional noise between them."""
    rng = np.random.default_rng(int(bpm))
    samples = rng.normal(0, noise, int(seconds * RATE)) if noise else np.zeros(int(seconds * RATE))
    period = 60.0 / bpm
    burst = int(0.02 * RATE)
    envelope = np.exp(-np.linspace(0, 6, burst))
    tone = np.sin(2 * np.pi * 900 * np.arange(burst) / RATE) * envelope
    for beat in np.arange(0.0, seconds, period):
        at = int(beat * RATE)
        if at + burst < samples.size:
            samples[at : at + burst] += tone
    return samples


def _music_like(bpm: float, seconds: float = 12.0) -> np.ndarray:
    """A click track over a sustained pad, closer to a real mix than clicks alone."""
    base = _click_track(bpm, seconds, noise=0.01)
    time = np.arange(base.size) / RATE
    pad = 0.15 * (np.sin(2 * np.pi * 110 * time) + np.sin(2 * np.pi * 165 * time))
    return base + pad


# -------------------------------------------------------------------- envelope


def test_envelope_spikes_on_onsets() -> None:
    envelope = onset_envelope(_click_track(120.0), RATE)
    assert envelope.size > 0
    assert envelope.max() == pytest.approx(1.0), "normalised to its own peak"
    assert envelope.mean() < 0.3, "onsets are sparse, not everywhere"


def test_silence_has_no_onsets() -> None:
    assert onset_envelope(np.zeros(RATE * 3), RATE).max() == 0.0


def test_a_too_short_clip_is_survivable() -> None:
    """One odd file must not take down an import."""
    assert onset_envelope(np.zeros(100), RATE).size == 0


# ----------------------------------------------------------------------- tempo


@pytest.mark.parametrize("bpm", [75.0, 90.0, 100.0, 110.0, 120.0, 128.0, 140.0, 150.0, 160.0])
def test_tempo_is_recovered(bpm: float) -> None:
    envelope = onset_envelope(_click_track(bpm), RATE)
    assert estimate_tempo(envelope, RATE) == pytest.approx(bpm, rel=0.04)


@pytest.mark.parametrize("bpm", [100.0, 120.0, 140.0, 160.0])
def test_tempo_survives_noise(bpm: float) -> None:
    envelope = onset_envelope(_click_track(bpm, noise=0.02), RATE)
    assert estimate_tempo(envelope, RATE) == pytest.approx(bpm, rel=0.04)


@pytest.mark.parametrize("bpm", [120.0, 140.0])
def test_tempo_is_not_halved(bpm: float) -> None:
    """The failure this was built to avoid.

    A beat period is rarely a whole number of frames, which makes the correlation at
    *two* periods stronger than at one - so the naive reading of a 140 BPM track is 70.
    """
    envelope = onset_envelope(_click_track(bpm), RATE)
    assert estimate_tempo(envelope, RATE) > bpm * 0.75


def test_tempo_survives_a_sustained_background() -> None:
    envelope = onset_envelope(_music_like(128.0), RATE)
    assert estimate_tempo(envelope, RATE) == pytest.approx(128.0, rel=0.05)


def test_tempo_stays_in_range() -> None:
    """The prior exists because autocorrelation cannot tell 90 from 180."""
    rng = np.random.default_rng(1)
    envelope = onset_envelope(rng.normal(0, 0.2, RATE * 8), RATE)
    assert MIN_BPM <= estimate_tempo(envelope, RATE) <= MAX_BPM


def test_an_empty_envelope_falls_back() -> None:
    assert MIN_BPM <= estimate_tempo(np.zeros(0), RATE) <= MAX_BPM


# ------------------------------------------------------------------------ grid


def test_beats_are_evenly_spaced() -> None:
    envelope = onset_envelope(_click_track(120.0), RATE)
    beats = beat_grid(envelope, RATE, 120.0, 12.0)
    gaps = np.diff(beats)
    assert gaps.std() < 1e-9, "a fixed grid does not drift"
    assert gaps.mean() == pytest.approx(0.5, rel=0.01)


def test_beats_land_on_the_clicks() -> None:
    """Tempo says how far apart; phase says where. Both have to be right."""
    envelope = onset_envelope(_click_track(120.0), RATE)
    beats = beat_grid(envelope, RATE, 120.0, 12.0)
    for beat in beats[:8]:
        assert min(abs(beat - k * 0.5) for k in range(24)) < 0.06


def test_grid_covers_the_track() -> None:
    beats = beat_grid(np.zeros(0), RATE, 120.0, 10.0)
    assert beats[0] == 0.0
    assert beats[-1] < 10.0
    assert len(beats) == 20


def test_degenerate_inputs_yield_no_grid() -> None:
    assert beat_grid(np.zeros(0), RATE, 0.0, 10.0) == ()
    assert beat_grid(np.zeros(0), RATE, 120.0, 0.0) == ()


# ------------------------------------------------------------------- beat map


def test_detect_produces_a_usable_map() -> None:
    result = detect(_music_like(128.0), RATE, 12.0)
    assert result.bpm == pytest.approx(128.0, rel=0.05)
    assert len(result.beats) > 20
    assert len(result.downbeats) == len(result.beats[::4])


def test_downbeats_are_every_fourth_beat() -> None:
    result = detect(_click_track(120.0), RATE, 8.0)
    assert result.downbeats[1] == pytest.approx(result.beats[4])


def test_beat_map_roundtrips() -> None:
    from genvai.timeline import BeatMap

    result = detect(_click_track(120.0), RATE, 6.0)
    assert BeatMap.model_validate_json(result.model_dump_json()) == result


# --------------------------------------------------------------------- snapping


def test_a_nearby_cut_moves_to_the_beat() -> None:
    assert snap(1.02, (0.0, 0.5, 1.0, 1.5), tolerance=0.12) == 1.0


def test_a_distant_cut_stays_put() -> None:
    """A span was chosen because it holds the good moment; dragging it would cut it out."""
    assert snap(1.24, (0.0, 0.5, 1.0, 1.5), tolerance=0.12) == 1.24


def test_snapping_picks_the_nearest_beat() -> None:
    assert snap(0.74, (0.0, 0.5, 1.0), tolerance=0.5) == 0.5


def test_snapping_without_beats_changes_nothing() -> None:
    assert snap(1.23, (), tolerance=1.0) == 1.23
