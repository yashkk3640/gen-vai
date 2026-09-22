"""Settings, resolved from environment variables and an optional .env file.

Nested via a double underscore: GENVAI_LLM__MODEL sets llm.model.
Every field has a working default; a fresh clone runs with no .env at all.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    timeout_s: float = 120.0
    temperature: float = 0.4
    max_retries: int = Field(default=3, description="Retries on schema-invalid output.")
    keep_alive: str = Field(
        default="0s",
        description=(
            "Ollama unload delay. Defaults to immediate release so the GPU is free "
            "for the diffusion model; see docs/02-architecture.md."
        ),
    )


class ImageSettings(BaseModel):
    provider: Literal["diffusers", "procedural", "null"] = "procedural"
    model: str = "stabilityai/sd-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    steps: int = 4
    width: int = 768
    height: int = 768
    low_vram: bool = Field(
        default=True, description="Attention slicing + sequential offload. Required under ~6 GB."
    )


class TTSSettings(BaseModel):
    provider: Literal["piper", "null"] = "null"
    voice: str = "en_US-amy-medium"
    speed: float = 1.0


class MusicSettings(BaseModel):
    provider: Literal["local", "null"] = "local"
    library_dir: Path = Path("assets/music_cache")
    allow_download: bool = Field(
        default=False,
        description=(
            "Master switch only. Even when true, each track still needs explicit "
            "per-track confirmation; see docs/06-decisions.md D4."
        ),
    )


class RenderSettings(BaseModel):
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20
    preset: str = "medium"
    preview_height: int = Field(default=480, description="Proxy render height for fast iteration.")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GENVAI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    image: ImageSettings = ImageSettings()
    tts: TTSSettings = TTSSettings()
    music: MusicSettings = MusicSettings()
    render: RenderSettings = RenderSettings()

    projects_dir: Path = Path("projects")


def load_settings() -> Settings:
    """Read settings fresh. Not cached - tests and the CLI both override the environment."""
    return Settings()
