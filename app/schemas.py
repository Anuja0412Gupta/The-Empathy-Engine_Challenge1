"""
Empathy Engine — Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class EmotionResult(BaseModel):
    """Output of the emotion detection layer."""
    emotion: str = Field(..., description="Detected emotion label")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Emotional intensity 0-1")
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution over all emotion classes",
    )


class VoiceParams(BaseModel):
    """Modulated vocal parameters."""
    rate: float = Field(..., description="Speech rate in WPM")
    pitch: float = Field(..., description="Pitch offset in semitones")
    volume: float = Field(..., ge=0.0, le=2.0, description="Volume multiplier")
    azure_style: str = Field(default="neutral", description="Azure SSML express-as style")


class SentenceEmotion(BaseModel):
    """Per-sentence emotion detection result."""
    sentence: str = Field(..., description="The individual sentence text")
    emotion: EmotionResult = Field(..., description="Emotion detected for this sentence")
    voice_params: Optional[VoiceParams] = Field(None, description="Voice params for this sentence")


class TTSRequest(BaseModel):
    """API request body."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")


class TTSResponse(BaseModel):
    """API response body."""
    text: str
    emotion: EmotionResult
    voice_params: VoiceParams
    audio_filename: str
    audio_url: str
    engine_used: str
    sentence_emotions: List[SentenceEmotion] = Field(
        default_factory=list,
        description="Per-sentence emotion breakdown",
    )
