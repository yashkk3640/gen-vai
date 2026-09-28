"""VisionPort backed by RapidOCR.

Optional: `uv sync --extra ocr`. About 10 MB of ONNX models, CPU only, no torch and no
system binary, which keeps the project's "clone and run" promise intact.

Chosen over a vision model after measuring both on a real poster. A 1.8B vision model
took 138 seconds and returned invented services at invented prices - dollars on a rupee
poster, a service that was not on it repeated six times. RapidOCR read the same poster in
4.4 seconds with every price correct. The difference is not accuracy, it is kind: a
language model asked for a number always produces one, while OCR either resolves the
characters or does not.
"""

from pathlib import Path
from typing import Any

from genvai.ocr import Box, read_brief
from genvai.promo import Brief


class RapidOcrVision:
    """Implements `VisionPort`."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine: Any | None = None

    def is_available(self) -> bool:
        """False when the extra is not installed. Never raises, never imports at import."""
        try:
            import rapidocr_onnxruntime  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def read(self, image: Path) -> Brief:
        """Recognise the text on a poster and pair it into a price list.

        An image the engine cannot handle yields an empty brief rather than raising: the
        user can still supply the offers, and losing the automation is a smaller failure
        than losing the command.
        """
        boxes = self.boxes(image)
        return read_brief(boxes) if boxes else Brief()

    def boxes(self, image: Path) -> list[Box]:
        """Raw recognised text with positions, for callers that want to pair differently."""
        engine = self._load()
        if engine is None:
            return []
        try:
            result, _ = engine(str(image))
        except Exception:  # noqa: BLE001 - one unreadable file must not stop a batch
            return []
        return [_as_box(entry) for entry in (result or []) if _usable(entry)]

    def unload(self) -> None:
        self._engine = None

    def _load(self) -> Any | None:
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        except ImportError:
            return None
        self._engine = RapidOCR()
        return self._engine


def _usable(entry: Any) -> bool:
    return isinstance(entry, list | tuple) and len(entry) >= 3 and bool(entry[1])


def _as_box(entry: Any) -> Box:
    """RapidOCR gives four corners; a rectangle is enough to reason about rows."""
    corners, text, confidence = entry[0], entry[1], entry[2]
    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    return Box(
        text=str(text),
        confidence=float(confidence),
        x=min(xs),
        y=min(ys),
        width=max(xs) - min(xs),
        height=max(ys) - min(ys),
    )
