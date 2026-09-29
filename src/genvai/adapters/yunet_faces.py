"""FaceDetector backed by OpenCV's YuNet.

Optional: needs `opencv-python` (already pulled in by `--extra ocr`) and a 230 KB ONNX
model, shipped in `assets/models/`. CPU only, about 0.15s per picture.

YuNet over a Haar cascade because OpenCV 5 no longer ships the cascades, and over a
larger detector because a poster's faces are big and frontal enough not to need one.

Measured on the client posters, at a 1024 input:

- a real photographed face scores 0.86-0.92
- a round nail-art thumbnail, which YuNet takes for a face, scores 0.74-0.80
- line-art icons of faces are not found at all at this size; at 1536 they are, at 0.89,
  which is why the input is not larger

Hence `min_confidence` 0.82 - between the two, with room on the real side.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from genvai.media import Face

INPUT = 1024
"""Square input side. Pictures are letterboxed into it rather than resized to fit.

Fixed, not per image: changing the input size between calls hung OpenCV 5's new graph
engine on the third picture. One size, set once, runs every time.
"""


class YunetFaces:
    """Implements `FaceDetector`."""

    name = "yunet"

    def __init__(self, model: Path, *, min_confidence: float = 0.82) -> None:
        self._model = model
        self._min_confidence = min_confidence
        self._detector: Any | None = None

    def is_available(self) -> bool:
        """False when OpenCV or the model file is missing. Never raises."""
        if not self._model.is_file():
            return False
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return False
        return hasattr(cv2, "FaceDetectorYN")

    def detect(self, image: NDArray[np.uint8]) -> tuple[Face, ...]:
        if image.ndim != 3 or image.size == 0:
            return ()
        detector = self._load()
        if detector is None:
            return ()

        import cv2  # noqa: PLC0415

        height, width = image.shape[:2]
        scale = INPUT / max(height, width)
        small = cv2.resize(
            np.ascontiguousarray(image[:, :, 2::-1]),
            (max(1, round(width * scale)), max(1, round(height * scale))),
        )
        padded = np.zeros((INPUT, INPUT, 3), dtype=np.uint8)
        padded[: small.shape[0], : small.shape[1]] = small

        try:
            _, found = detector.detect(padded)
        except cv2.error:
            return ()
        if found is None:
            return ()

        faces = [
            _face(row, scale, width, height)
            for row in found
            if float(row[14]) >= self._min_confidence
        ]
        return tuple(sorted((f for f in faces if f is not None), key=lambda f: -f.confidence))

    def _load(self) -> Any | None:
        if self._detector is None and self.is_available():
            import cv2  # noqa: PLC0415

            self._detector = cv2.FaceDetectorYN.create(
                str(self._model), "", (INPUT, INPUT), self._min_confidence
            )
        return self._detector


def _face(row: NDArray[np.float32], scale: float, width: int, height: int) -> Face | None:
    """A detection in padded-input pixels, as a face in fractions of the original.

    Clipped to the frame: a face cut by the edge is reported partly outside it.
    """
    left = max(0.0, float(row[0]) / scale / width)
    top = max(0.0, float(row[1]) / scale / height)
    right = min(1.0, float(row[0] + row[2]) / scale / width)
    bottom = min(1.0, float(row[1] + row[3]) / scale / height)
    if right <= left or bottom <= top:
        return None
    return Face(
        rect=(left, top, right - left, bottom - top),
        confidence=min(1.0, max(0.0, float(row[14]))),
    )
