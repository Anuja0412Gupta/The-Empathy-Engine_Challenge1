"""
Empathy Engine — Voice Modulation Engine
Maps detected emotion + intensity → dynamic vocal parameters
with non-linear intensity scaling for expressive differentiation.
"""

import logging
from app.config import emotion_config
from app.schemas import EmotionResult, VoiceParams

logger = logging.getLogger("empathy_engine.voice")

# ── Defaults ──────────────────────────────────────────────
_defaults = emotion_config.get("defaults", {})
BASE_RATE = _defaults.get("rate_wpm", 175)
BASE_VOLUME = _defaults.get("volume", 1.0)

# ── Pitch cap (keep pitch changes minimal) ────────────────
MAX_PITCH_OFFSET = 0.5  # ±0.5 semitones max


def modulate(emotion_result: EmotionResult) -> VoiceParams:
    """
    Convert an EmotionResult into concrete voice parameters.

    Uses the emotion_config.yaml mapping with non-linear intensity scaling:
        scaled_intensity = intensity ^ 1.5  (subtle for low, dramatic for high)
        final_value = base + (scaled_intensity × range)

    Pitch is capped to ±0.5 semitones to keep changes minimal.
    """
    emotions_map = emotion_config.get("emotions", {})
    emotion_key = emotion_result.emotion.lower()
    mapping = emotions_map.get(emotion_key, emotions_map.get("neutral", {}))

    raw_intensity = emotion_result.intensity

    # ── Non-linear curve: low stays subtle, high gets dramatic ─
    scaled_intensity = raw_intensity ** 1.5

    # ── Rate (WPM) ────────────────────────────────────────
    rate_cfg = mapping.get("rate", {"base": 0, "range": 0})
    rate_offset = rate_cfg["base"] + (scaled_intensity * rate_cfg["range"])
    final_rate = BASE_RATE + rate_offset

    # ── Pitch (semitones) — capped to stay minimal ────────
    pitch_cfg = mapping.get("pitch", {"base": 0, "range": 0})
    raw_pitch = pitch_cfg["base"] + (scaled_intensity * pitch_cfg["range"])
    final_pitch = max(-MAX_PITCH_OFFSET, min(raw_pitch, MAX_PITCH_OFFSET))

    # ── Volume (multiplier) ───────────────────────────────
    vol_cfg = mapping.get("volume", {"base": 1.0, "range": 0})
    final_volume = vol_cfg["base"] + (scaled_intensity * vol_cfg["range"])
    final_volume = max(0.1, min(final_volume, 2.0))  # clamp

    # ── Azure style ───────────────────────────────────────
    azure_style = mapping.get("azure_style", "neutral")

    params = VoiceParams(
        rate=round(final_rate, 1),
        pitch=round(final_pitch, 2),
        volume=round(final_volume, 3),
        azure_style=azure_style,
    )

    logger.info(
        f"Voice modulation: emotion={emotion_key} "
        f"raw_intensity={raw_intensity:.2f} scaled={scaled_intensity:.2f} → "
        f"rate={params.rate} pitch={params.pitch} vol={params.volume} style={params.azure_style}"
    )

    return params
