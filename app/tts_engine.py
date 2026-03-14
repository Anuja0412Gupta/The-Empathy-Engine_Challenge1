"""
Empathy Engine — TTS Engine Selection & Synthesis
Runtime selection: Azure Cognitive Services TTS (if key exists) → pyttsx3 fallback

Supports both single-text and multi-sentence synthesis.
"""

import logging
import uuid
import wave
import struct
from pathlib import Path
from typing import Tuple, List
from app.config import settings, OUTPUT_DIR
from app.schemas import VoiceParams, SentenceEmotion
from app.ssml_generator import build_ssml, build_ssml_multi

logger = logging.getLogger("empathy_engine.tts")


def _azure_available() -> bool:
    """Check if Azure Speech credentials are configured."""
    return bool(settings.azure_speech_key and settings.azure_speech_key.strip())


def synthesize(text: str, voice_params: VoiceParams, emotion: str, intensity: float = 0.5) -> Tuple[str, str]:
    """
    Synthesize speech from text with emotional modulation.

    Returns:
        (filename, engine_used) — filename is relative to output/
    """
    filename = f"empathy_{uuid.uuid4().hex[:10]}.wav"
    output_path = OUTPUT_DIR / filename

    if _azure_available():
        engine_used = _synthesize_azure(text, voice_params, emotion, intensity, output_path)
    else:
        engine_used = _synthesize_pyttsx3(text, voice_params, output_path)

    logger.info(f"Audio saved: {output_path} (engine={engine_used})")
    return filename, engine_used


def synthesize_multi(
    sentence_emotions: List[SentenceEmotion],
    full_text: str,
    voice_params_overall: VoiceParams,
) -> Tuple[str, str]:
    """
    Synthesize speech with per-sentence emotion modulation.

    Returns:
        (filename, engine_used) — filename is relative to output/
    """
    filename = f"empathy_{uuid.uuid4().hex[:10]}.wav"
    output_path = OUTPUT_DIR / filename

    if _azure_available():
        engine_used = _synthesize_azure_multi(sentence_emotions, output_path)
    else:
        # pyttsx3 doesn't support per-sentence SSML — use overall params
        engine_used = _synthesize_pyttsx3(full_text, voice_params_overall, output_path)

    logger.info(f"Audio saved: {output_path} (engine={engine_used})")
    return filename, engine_used


# ── Azure Cognitive Services TTS ──────────────────────────
def _synthesize_azure(
    text: str, voice_params: VoiceParams, emotion: str, intensity: float, output_path: Path
) -> str:
    """Use Azure Cognitive Services for high-quality expressive TTS."""
    try:
        import azure.cognitiveservices.speech as speechsdk

        ssml = build_ssml(text, voice_params, emotion, intensity)
        logger.debug(f"Azure SSML:\n{ssml}")

        speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )

        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Azure TTS synthesis completed successfully.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            logger.error(f"Azure TTS canceled: {cancellation.reason}")
            if cancellation.error_details:
                logger.error(f"Error details: {cancellation.error_details}")
            logger.warning("Falling back to pyttsx3…")
            return _synthesize_pyttsx3(text, voice_params, output_path)

        return "azure"

    except ImportError:
        logger.warning("azure-cognitiveservices-speech not installed, falling back to pyttsx3")
        return _synthesize_pyttsx3(text, voice_params, output_path)
    except Exception as e:
        logger.error(f"Azure TTS error: {e}. Falling back to pyttsx3…")
        return _synthesize_pyttsx3(text, voice_params, output_path)


def _synthesize_azure_multi(
    sentence_emotions: List[SentenceEmotion], output_path: Path
) -> str:
    """Use Azure TTS with per-sentence SSML for multi-emotion synthesis."""
    try:
        import azure.cognitiveservices.speech as speechsdk

        ssml = build_ssml_multi(sentence_emotions)
        logger.debug(f"Azure Multi-SSML:\n{ssml}")

        speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )

        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Azure TTS multi-sentence synthesis completed.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            logger.error(f"Azure TTS canceled: {cancellation.reason}")
            if cancellation.error_details:
                logger.error(f"Error details: {cancellation.error_details}")
            # Fallback: concatenate text, use first sentence's params
            full_text = " ".join(se.sentence for se in sentence_emotions)
            if sentence_emotions and sentence_emotions[0].voice_params:
                return _synthesize_pyttsx3(full_text, sentence_emotions[0].voice_params, output_path)
            return _synthesize_pyttsx3(full_text, VoiceParams(rate=175, pitch=0, volume=1.0), output_path)

        return "azure"

    except ImportError:
        logger.warning("azure-cognitiveservices-speech not installed, falling back to pyttsx3")
        full_text = " ".join(se.sentence for se in sentence_emotions)
        if sentence_emotions and sentence_emotions[0].voice_params:
            return _synthesize_pyttsx3(full_text, sentence_emotions[0].voice_params, output_path)
        return _synthesize_pyttsx3(full_text, VoiceParams(rate=175, pitch=0, volume=1.0), output_path)
    except Exception as e:
        logger.error(f"Azure TTS error: {e}. Falling back to pyttsx3…")
        full_text = " ".join(se.sentence for se in sentence_emotions)
        if sentence_emotions and sentence_emotions[0].voice_params:
            return _synthesize_pyttsx3(full_text, sentence_emotions[0].voice_params, output_path)
        return _synthesize_pyttsx3(full_text, VoiceParams(rate=175, pitch=0, volume=1.0), output_path)


# ── pyttsx3 Offline Fallback ─────────────────────────────
def _synthesize_pyttsx3(text: str, voice_params: VoiceParams, output_path: Path) -> str:
    """Use pyttsx3 for offline TTS with rate/volume modulation."""
    try:
        import pyttsx3

        engine = pyttsx3.init()

        # ── Rate ──────────────────────────────────────────
        engine.setProperty("rate", int(voice_params.rate))

        # ── Volume (pyttsx3 uses 0.0 – 1.0) ──────────────
        vol = min(max(voice_params.volume, 0.0), 1.0)
        engine.setProperty("volume", vol)

        # ── Save to file ─────────────────────────────────
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

        # Verify file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.warning("pyttsx3 produced empty file, generating silent placeholder")
            _generate_silent_wav(output_path)

        return "pyttsx3"

    except Exception as e:
        logger.error(f"pyttsx3 error: {e}")
        _generate_silent_wav(output_path)
        return "pyttsx3 (error — silent placeholder)"


def _generate_silent_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 22050):
    """Generate a short silent WAV as a last-resort placeholder."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
