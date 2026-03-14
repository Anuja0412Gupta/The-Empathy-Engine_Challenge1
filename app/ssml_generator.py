"""
Empathy Engine — SSML Generator
Builds Azure-compatible SSML using a SINGLE base voice (from .env)
with per-sentence prosody modulation, express-as styles, and
intensity-scaled styledegree.

Supports multi-sentence SSML where each sentence gets its own
emotion style and prosody parameters.
"""

import logging
from typing import List, Optional
from app.schemas import VoiceParams, SentenceEmotion
from app.config import settings

logger = logging.getLogger("empathy_engine.ssml")

# ── Emotion → Azure express-as style mapping ─────────────
_EMOTION_STYLE_MAP = {
    "joy":      "cheerful",
    "anger":    "angry",
    "sadness":  "sad",
    "fear":     "terrified",
    "surprise": "excited",
    "disgust":  "angry",
    "neutral":  None,
}


def _rate_to_ssml(rate_wpm: float) -> str:
    """Convert WPM to SSML prosody rate percentage relative to default (175)."""
    pct = ((rate_wpm - 175) / 175) * 100
    if pct >= 0:
        return f"+{pct:.0f}%"
    return f"{pct:.0f}%"


def _pitch_to_ssml(semitones: float) -> str:
    """Convert semitone offset to SSML prosody pitch."""
    if semitones >= 0:
        return f"+{semitones:.1f}st"
    return f"{semitones:.1f}st"


def _volume_to_ssml(volume: float) -> str:
    """Convert volume multiplier to SSML prosody volume keyword."""
    if volume >= 1.3:
        return "x-loud"
    elif volume >= 1.1:
        return "loud"
    elif volume >= 0.9:
        return "medium"
    elif volume >= 0.7:
        return "soft"
    else:
        return "x-soft"


def _build_sentence_ssml(
    text: str,
    voice_params: VoiceParams,
    emotion: str,
    intensity: float,
) -> str:
    """Build SSML fragment for a single sentence (no <speak>/<voice> wrapper)."""
    style = _EMOTION_STYLE_MAP.get(emotion)
    content = _escape_xml(text)

    # Convert numeric params to SSML strings
    rate = _rate_to_ssml(voice_params.rate)
    pitch = _pitch_to_ssml(voice_params.pitch)
    volume = _volume_to_ssml(voice_params.volume)

    if style:
        # Scale styledegree by intensity (0.5 to 2.0 max) to match test_azure for high intensities
        styledegree = round(0.5 + (intensity * 1.5), 2)
        styledegree = min(max(styledegree, 0.0), 2.0)
        
        # Apply the explicit prosody and style, similar to test_azure but dynamically scaled
        fragment = (
            f'<mstts:express-as style="{style}" styledegree="{styledegree}">'
            f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
            f'{content}'
            f'</prosody>'
            f'</mstts:express-as>'
        )
    else:
        # Neutral: minimal prosody, no express-as
        styledegree = 0.0
        fragment = (
            f'<prosody rate="{rate}" pitch="{pitch}" volume="{volume}">'
            f'{content}'
            f'</prosody>'
        )

    logger.info(
        f"SSML fragment: emotion={emotion} intensity={intensity:.2f} "
        f"degree={styledegree} rate={rate} pitch={pitch} vol={volume}"
    )
    return fragment


def build_ssml(text: str, voice_params: VoiceParams, emotion: str, intensity: float = 0.5) -> str:
    """
    Generate Azure SSML for a single text block (backward compatible).
    Now uses intensity-scaled styledegree and actual prosody values.
    """
    voice_name = settings.azure_voice_name
    sentence_fragment = _build_sentence_ssml(text, voice_params, emotion, intensity)

    ssml = (
        f'<speak version="1.0"'
        f' xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xmlns:mstts="https://www.w3.org/2001/mstts"'
        f' xml:lang="en-US">'
        f'<voice name="{voice_name}">'
        f'{sentence_fragment}'
        f'</voice>'
        f'</speak>'
    )

    logger.info(f"SSML: voice={voice_name} emotion={emotion} intensity={intensity:.2f}")
    return ssml


def build_ssml_multi(sentence_emotions: List[SentenceEmotion]) -> str:
    """
    Generate Azure SSML with per-sentence emotion styles.

    Each sentence gets its own express-as style and prosody,
    all wrapped in a single <speak>/<voice> block.
    """
    voice_name = settings.azure_voice_name
    fragments = []

    for se in sentence_emotions:
        if se.voice_params is None:
            continue
        fragment = _build_sentence_ssml(
            se.sentence,
            se.voice_params,
            se.emotion.emotion,
            se.emotion.intensity,
        )
        fragments.append(fragment)

    # Add small break between sentences for natural flow
    joined = '<break time="200ms"/>'.join(fragments)

    ssml = (
        f'<speak version="1.0"'
        f' xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xmlns:mstts="https://www.w3.org/2001/mstts"'
        f' xml:lang="en-US">'
        f'<voice name="{voice_name}">'
        f'{joined}'
        f'</voice>'
        f'</speak>'
    )

    logger.info(
        f"Multi-sentence SSML: voice={voice_name} "
        f"sentences={len(sentence_emotions)}"
    )
    return ssml


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
