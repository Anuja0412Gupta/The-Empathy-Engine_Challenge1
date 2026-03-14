"""
Empathy Engine — Emotion Detection Layer
Hybrid approach: VADER (intensity) + HuggingFace (granular classification)
Uses HuggingFace Inference API if HF_API_TOKEN is set,
falls back to local model, then VADER-only.

Supports per-sentence emotion detection for multi-sentence input.
"""

import logging
import re
import requests
from typing import Optional, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.schemas import EmotionResult, SentenceEmotion
from app.config import settings

logger = logging.getLogger("empathy_engine.emotion")

# ── Lazy-loaded singletons ────────────────────────────────
_vader: Optional[SentimentIntensityAnalyzer] = None

HF_MODEL = "j-hartmann/emotion-english-distilroberta-base"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


def _get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        logger.info("Initializing VADER sentiment analyzer…")
        _vader = SentimentIntensityAnalyzer()
    return _vader


def _hf_api_classify(text: str) -> Optional[list]:
    """Call HuggingFace Inference API for emotion classification."""
    token = settings.hf_api_token
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            # API returns [[{label, score}, ...]]
            if isinstance(data, list) and len(data) > 0:
                return data[0] if isinstance(data[0], list) else data
        else:
            logger.warning(
                f"HuggingFace API returned {response.status_code}: {response.text}"
            )
            return None
    except Exception as e:
        logger.warning(f"HuggingFace API error: {e}")
        return None


# ── VADER-only emotion mapping ────────────────────────────
def _vader_to_emotion(compound: float) -> str:
    """Map VADER compound score to a basic emotion label."""
    if compound >= 0.5:
        return "joy"
    elif compound >= 0.1:
        return "surprise"
    elif compound <= -0.5:
        return "anger"
    elif compound <= -0.3:
        return "sadness"
    elif compound <= -0.1:
        return "fear"
    else:
        return "neutral"


# ── Sentence splitting ────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries."""
    # Split on sentence-ending punctuation followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


# ── Public API ────────────────────────────────────────────
def detect_emotion(text: str) -> EmotionResult:
    """
    Detect emotion and intensity from input text.

    1. VADER → compound score magnitude → intensity (0-1)
    2. HuggingFace API → top emotion label + probability distribution
       (falls back to VADER mapping if unavailable)
    """
    # ── VADER intensity ───────────────────────────────────
    vader = _get_vader()
    vader_scores = vader.polarity_scores(text)
    intensity = min(abs(vader_scores["compound"]), 1.0)
    intensity = max(intensity, 0.15)  # minimum intensity floor

    # ── HuggingFace classification (API) ──────────────────
    hf_results = _hf_api_classify(text)

    if hf_results is not None:
        try:
            probabilities = {
                r["label"]: round(float(r["score"]), 4) for r in hf_results
            }
            top = max(hf_results, key=lambda r: r["score"])
            emotion_label = top["label"]
            logger.info(f"HuggingFace API: {emotion_label} ({top['score']:.3f})")
        except Exception as e:
            logger.warning(f"Error parsing HF results: {e}. Falling back to VADER.")
            emotion_label = _vader_to_emotion(vader_scores["compound"])
            probabilities = {emotion_label: round(intensity, 4)}
    else:
        # VADER-only fallback
        emotion_label = _vader_to_emotion(vader_scores["compound"])
        probabilities = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0,
            "fear": 0.0, "surprise": 0.0, "disgust": 0.0, "neutral": 0.0,
        }
        probabilities[emotion_label] = round(intensity, 4)

    logger.info(
        f"Detected emotion={emotion_label} intensity={intensity:.2f} "
        f"(VADER compound={vader_scores['compound']:.3f})"
    )

    return EmotionResult(
        emotion=emotion_label,
        intensity=round(intensity, 4),
        probabilities=probabilities,
    )


def detect_emotions_per_sentence(text: str) -> List[SentenceEmotion]:
    """
    Split text into sentences and detect emotion for each.

    Returns a list of SentenceEmotion objects, one per sentence.
    For single-sentence input, returns a list with one entry.
    """
    sentences = _split_sentences(text)

    # If we couldn't split meaningfully, treat entire text as one sentence
    if not sentences:
        sentences = [text.strip()]

    results = []
    for sentence in sentences:
        emotion_result = detect_emotion(sentence)
        results.append(
            SentenceEmotion(
                sentence=sentence,
                emotion=emotion_result,
            )
        )
        logger.info(
            f"Sentence: \"{sentence[:50]}…\" → "
            f"emotion={emotion_result.emotion} intensity={emotion_result.intensity:.2f}"
        )

    return results
