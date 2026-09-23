"""Finding the beat in a music track. Pure numpy over a waveform.

Cutting on the beat is most of what separates a reel that feels edited from one that
feels assembled, and it needs signal processing rather than a model - a few hundred
milliseconds of arithmetic, no GPU, no download.

The approach is deliberately the simple one: a *fixed* tempo grid rather than a beat
tracker that follows tempo drift. Short-form music is almost always machine-timed, and a
rigid grid is both more robust and more useful here, because scene boundaries snap to it
without accumulating error across a thirty-second cut.
"""

import numpy as np
from numpy.typing import NDArray

from genvai.timeline import BeatMap

Waveform = NDArray[np.float64]

HOP = 512
WINDOW = 1024
MIN_BPM = 60.0
MAX_BPM = 180.0
ANALYSIS_SECONDS = 120.0
"""How much of a track to examine. Tempo does not change, and a five-minute file
would cost memory for nothing."""

TEMPO_STEPS = 600
"""How finely the tempo range is searched - about 0.2 BPM apart."""

HARMONICS = 3
"""How many multiples of a candidate period contribute to its score."""

SMOOTHING = 7
"""Width, in frames, of the window applied to the onset envelope."""

PREFERRED_BPM = 120.0
TEMPO_SPREAD = 0.9
"""Width of the tempo prior, in log2 units.

Autocorrelation peaks just as hard at half and double the true tempo, so something has
to break the tie. A preference for ordinary dance tempos is the standard answer and is
right far more often than picking the largest peak.
"""


def onset_envelope(samples: Waveform, sample_rate: int) -> Waveform:
    """Where energy suddenly arrives, frame by frame.

    Spectral flux: the summed *positive* change in each frequency bin between
    consecutive frames. Only rises count, because a note starting is an onset and a note
    ending is not.
    """
    if samples.size < WINDOW * 2:
        return np.zeros(0)

    limit = int(ANALYSIS_SECONDS * sample_rate)
    signal = samples[:limit]

    frames = 1 + (signal.size - WINDOW) // HOP
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(frames)[:, None]
    windowed = signal[index] * np.hanning(WINDOW)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1))

    rising = np.diff(magnitude, axis=0)
    flux = np.maximum(rising, 0.0).sum(axis=1)
    return _normalise(flux)


def estimate_tempo(envelope: Waveform, sample_rate: int) -> float:
    """Beats per minute, from the periodicity of the onset envelope.

    Three details, each fixing a way the textbook version halves the tempo on real
    material. All three were arrived at by measurement, not taste - the plain version
    read a 140 BPM click track as 70.

    - **The envelope is smoothed first.** A beat period is rarely a whole number of
      analysis frames, so onsets land alternately either side of a frame boundary. The
      correlation at one period then misses half of them while the correlation at two
      periods, being nearer a whole number, does not.
    - **Lags are interpolated, not rounded.** Rounding reintroduces exactly the bias the
      smoothing removes, and worst at fast tempos where a period is few frames.
    - **Harmonics are summed.** A candidate tempo is scored on its period *and* its
      multiples, which is what separates a tempo from half of it: 140 gets credit at the
      lag belonging to 70, while 70 gets nothing from 140's.
    """
    if envelope.size < 16:
        return PREFERRED_BPM

    smoothed = _smooth(envelope)
    centred = smoothed - smoothed.mean()
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1 :]
    if correlation.size < 4:
        return PREFERRED_BPM

    frames_per_second = sample_rate / HOP
    lags = np.arange(correlation.size, dtype=np.float64)
    candidates = np.linspace(MIN_BPM, MAX_BPM, TEMPO_STEPS)

    score = np.zeros_like(candidates)
    for harmonic in range(1, HARMONICS + 1):
        at = 60.0 * frames_per_second / candidates * harmonic
        score += np.interp(at, lags, correlation, left=0.0, right=0.0) / harmonic

    return float(candidates[int(np.argmax(score * _tempo_prior(candidates)))])


def beat_grid(
    envelope: Waveform, sample_rate: int, bpm: float, duration: float
) -> tuple[float, ...]:
    """Beat times in seconds, from a tempo and the phase that best fits the onsets.

    The tempo says how far apart the beats are; the phase says where the first one
    lands. Choosing it by testing every offset within one beat and keeping whichever
    lines up with the most onset energy is cheap and avoids a tracker.
    """
    if bpm <= 0 or duration <= 0:
        return ()

    period = 60.0 / bpm
    if envelope.size == 0:
        return tuple(np.arange(0.0, duration, period))

    frames_per_second = sample_rate / HOP
    period_frames = period * frames_per_second
    offsets = np.linspace(0.0, period_frames, 24, endpoint=False)

    best_offset, best_score = 0.0, -np.inf
    for offset in offsets:
        positions = np.round(np.arange(offset, envelope.size, period_frames)).astype(int)
        positions = positions[positions < envelope.size]
        score = float(envelope[positions].sum()) if positions.size else 0.0
        if score > best_score:
            best_offset, best_score = float(offset), score

    start = best_offset / frames_per_second
    return tuple(float(t) for t in np.arange(start, duration, period))


def detect(samples: Waveform, sample_rate: int, duration: float) -> BeatMap:
    """A full beat map: tempo, beats, and the bar starts among them."""
    envelope = onset_envelope(samples, sample_rate)
    bpm = estimate_tempo(envelope, sample_rate)
    beats = beat_grid(envelope, sample_rate, bpm, duration)
    return BeatMap(bpm=bpm, beats=beats, downbeats=beats[::4])


def snap(time: float, beats: tuple[float, ...], *, tolerance: float) -> float:
    """Move a cut to the nearest beat, unless that would drag it too far.

    The tolerance is what keeps beat alignment from wrecking a trim: a span was chosen
    because it holds the good moment, and yanking its boundary half a second to land on
    a beat would cut the moment out. Beyond the tolerance the cut stays where it was.
    """
    if not beats:
        return time
    nearest = min(beats, key=lambda b: abs(b - time))
    return nearest if abs(nearest - time) <= tolerance else time


def _tempo_prior(bpm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Log-normal weighting around a typical tempo."""
    return np.exp(-0.5 * (np.log2(bpm / PREFERRED_BPM) / TEMPO_SPREAD) ** 2)


def _smooth(values: Waveform) -> Waveform:
    """Widen each onset so a beat falling between two frames still correlates."""
    if values.size <= SMOOTHING:
        return values
    kernel = np.hanning(SMOOTHING + 2)[1:-1]
    return np.convolve(values, kernel / kernel.sum(), mode="same")


def _normalise(values: Waveform) -> Waveform:
    peak = values.max() if values.size else 0.0
    return values / peak if peak > 0 else values
