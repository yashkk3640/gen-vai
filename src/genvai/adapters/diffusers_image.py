"""ImageProvider backed by a local Stable Diffusion pipeline.

Optional: needs `uv sync --extra image`, which is about 3 GB of torch. Without it
`is_available()` returns False and the caller falls back to procedural backdrops, so a
machine with no GPU never sees an import error.

Defaults to SD-Turbo, which produces an image in one to four steps rather than the usual
twenty-five. On a 4 GB card that is the difference between a few seconds per scene and a
minute, and the quality gap at this size is smaller than the time gap.

Everything here is lazy. torch is imported inside the methods, not at module scope,
because importing it costs seconds and a project that never generates an image should
never pay that.
"""

from pathlib import Path
from typing import Any

from genvai.config import ImageSettings
from genvai.errors import ProviderUnavailable

GUIDANCE_FREE = 0.0
"""SD-Turbo is trained without classifier-free guidance. Passing a guidance scale makes
it slower and worse, which is a mistake worth not making twice."""

MULTIPLE = 8
"""Latent diffusion works in 8-pixel blocks; anything else is rounded up and cropped."""

FALLBACK_SCALE = 0.5
"""How far to shrink after running out of memory. A smaller picture upscales acceptably;
a failed render does not."""


class DiffusersImageProvider:
    """Implements `ImageProvider`."""

    name = "diffusers"

    def __init__(self, settings: ImageSettings, cache_dir: Path) -> None:
        self._settings = settings
        self._cache_dir = cache_dir
        self._pipeline: Any | None = None

    def is_available(self) -> bool:
        """False when the extra is not installed or no usable device exists.

        Never raises and never imports at module scope: this is called to decide whether
        to use the provider at all, including on machines where torch does not exist.
        """
        try:
            import torch  # noqa: PLC0415
            from diffusers import AutoPipelineForText2Image  # noqa: F401, PLC0415
        except ImportError:
            return False
        return self._settings.device == "cpu" or torch.cuda.is_available()

    def generate(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        negative_prompt: str | None = None,
    ) -> Path:
        """Generate deterministically from `seed`.

        On running out of memory it retries once at half size rather than failing. A
        smaller image upscales; a missing one stops the render.
        """
        import torch  # noqa: PLC0415

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        out = self._cache_dir / f"gen-{seed:016x}-{width}x{height}.png"
        if out.exists():
            return out

        for scale in (1.0, FALLBACK_SCALE):
            try:
                image = self._run(
                    prompt,
                    negative_prompt,
                    _rounded(width * scale),
                    _rounded(height * scale),
                    seed,
                )
            except torch.cuda.OutOfMemoryError:
                self.unload()
                torch.cuda.empty_cache()
                if scale == FALLBACK_SCALE:
                    raise ProviderUnavailable(
                        self.name,
                        f"out of VRAM even at {int(width * FALLBACK_SCALE)}px. "
                        "Reduce GENVAI_IMAGE__WIDTH, or use the procedural provider.",
                    ) from None
                continue

            if image.size != (width, height):
                image = image.resize((width, height))
            image.save(out)
            return out

        raise ProviderUnavailable(self.name, "generation failed")

    def unload(self) -> None:
        """Release the model from VRAM.

        Called at the phase boundary, because at 4 GB this and the text model cannot both
        be resident. See 'Phase-ordered model loading' in docs/decisions.md.
        """
        if self._pipeline is None:
            return
        self._pipeline = None
        try:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return

    # ------------------------------------------------------------------ internals

    def _run(self, prompt: str, negative: str | None, width: int, height: int, seed: int) -> Any:
        import torch  # noqa: PLC0415

        pipeline = self._load()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_inference_steps=max(1, self._settings.steps),
            guidance_scale=GUIDANCE_FREE,
            generator=generator,
        )
        return result.images[0]

    def _load(self) -> Any:
        """Build the pipeline once and keep it until `unload`.

        The memory-saving options are not optional at 4 GB. Attention slicing computes
        attention in chunks, VAE slicing decodes the image in strips, and sequential
        offload keeps only the layer in use on the card - each trades a little speed for
        a lot of headroom.
        """
        if self._pipeline is not None:
            return self._pipeline

        import torch  # noqa: PLC0415
        from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

        device = self._settings.device
        pipeline = AutoPipelineForText2Image.from_pretrained(
            self._settings.model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            safety_checker=None,
        )
        if self._settings.low_vram and device == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        pipeline.set_progress_bar_config(disable=True)
        self._pipeline = pipeline
        return pipeline


def _rounded(value: float) -> int:
    """Up to the next multiple of 8, which is what the latent grid requires."""
    return max(MULTIPLE, int(value + MULTIPLE - 1) // MULTIPLE * MULTIPLE)
