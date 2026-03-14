"""
Empathy Engine — Configuration
Loads settings from .env and emotion_config.yaml
"""

import os
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

# ── Paths ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "emotion_config.yaml"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Environment Settings ──────────────────────────────────
class Settings(BaseSettings):
    azure_speech_key: Optional[str] = Field(default=None)
    azure_speech_region: str = Field(default="westus2")
    azure_voice_name: str = Field(default="en-US-JennyNeural")
    hf_api_token: Optional[str] = Field(default=None)

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"


settings = Settings()


# ── Emotion Config Loader ─────────────────────────────────
def load_emotion_config() -> dict:
    """Load emotion→voice mapping from YAML config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Fallback defaults
    return {
        "emotions": {
            "neutral": {
                "rate": {"base": 0, "range": 0},
                "pitch": {"base": 0, "range": 0},
                "volume": {"base": 1.0, "range": 0},
                "azure_style": "neutral",
            }
        },
        "defaults": {"rate_wpm": 175, "pitch_hz": 0, "volume": 1.0},
    }


emotion_config = load_emotion_config()
