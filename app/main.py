"""
Empathy Engine — FastAPI App + CLI Entry Point
Supports per-sentence emotion detection and intensity-scaled voice modulation.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import OUTPUT_DIR, ROOT_DIR
from app.schemas import TTSRequest, TTSResponse, SentenceEmotion
from app.emotion_detector import detect_emotion, detect_emotions_per_sentence
from app.voice_modulator import modulate
from app.tts_engine import synthesize, synthesize_multi

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("empathy_engine")


# ── Lifespan: pre-load models ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🧠 Empathy Engine starting — pre-loading models…")
    try:
        from app.emotion_detector import _get_vader
        _get_vader()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-load warning: {e}")
    yield
    logger.info("Empathy Engine shutting down.")


# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="Empathy Engine",
    description="Emotion-Aware Text-to-Speech System with Intensity Scaling",
    version="1.1.0",
    lifespan=lifespan,
)

# Static files & templates
STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Web UI ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Serve the Empathy Engine Web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ── Synthesize API ────────────────────────────────────────
@app.post("/api/synthesize", response_model=TTSResponse)
async def api_synthesize(req: TTSRequest):
    """
    Full pipeline with per-sentence emotion analysis:
    Text → Split Sentences → Detect Emotion per Sentence → Modulate Each → TTS → Audio
    """
    logger.info(f"📝 Input: \"{req.text[:80]}{'…' if len(req.text) > 80 else ''}\"")

    # 1. Per-sentence emotion detection
    sentence_emotions = detect_emotions_per_sentence(req.text)

    # 2. Modulate voice for each sentence
    for se in sentence_emotions:
        se.voice_params = modulate(se.emotion)

    # 3. Overall emotion = dominant (highest intensity) sentence
    dominant = max(sentence_emotions, key=lambda se: se.emotion.intensity)
    overall_emotion = dominant.emotion
    overall_voice_params = dominant.voice_params

    # 4. Synthesize with per-sentence SSML (if multiple sentences)
    if len(sentence_emotions) > 1:
        filename, engine_used = synthesize_multi(
            sentence_emotions, req.text, overall_voice_params
        )
    else:
        filename, engine_used = synthesize(
            req.text, overall_voice_params, overall_emotion.emotion, overall_emotion.intensity
        )

    # 5. Build response
    response = TTSResponse(
        text=req.text,
        emotion=overall_emotion,
        voice_params=overall_voice_params,
        audio_filename=filename,
        audio_url=f"/api/audio/{filename}",
        engine_used=engine_used,
        sentence_emotions=sentence_emotions,
    )

    logger.info(
        f"✅ Done: sentences={len(sentence_emotions)} "
        f"dominant_emotion={overall_emotion.emotion} "
        f"intensity={overall_emotion.intensity:.2f} engine={engine_used}"
    )
    return response


# ── Serve Audio Files ─────────────────────────────────────
@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated WAV file."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return JSONResponse({"error": "Audio file not found"}, status_code=404)
    return FileResponse(
        str(filepath),
        media_type="audio/wav",
        filename=filename,
    )


# ── Batch Endpoint ────────────────────────────────────────
@app.post("/api/batch")
async def api_batch(texts: List[str]):
    """Process multiple texts in a single request."""
    results = []
    for text in texts:
        sentence_emotions = detect_emotions_per_sentence(text)
        for se in sentence_emotions:
            se.voice_params = modulate(se.emotion)

        dominant = max(sentence_emotions, key=lambda se: se.emotion.intensity)
        overall_emotion = dominant.emotion
        overall_voice_params = dominant.voice_params

        if len(sentence_emotions) > 1:
            filename, engine_used = synthesize_multi(
                sentence_emotions, text, overall_voice_params
            )
        else:
            filename, engine_used = synthesize(
                text, overall_voice_params, overall_emotion.emotion, overall_emotion.intensity
            )

        results.append(
            TTSResponse(
                text=text,
                emotion=overall_emotion,
                voice_params=overall_voice_params,
                audio_filename=filename,
                audio_url=f"/api/audio/{filename}",
                engine_used=engine_used,
                sentence_emotions=sentence_emotions,
            ).model_dump()
        )
    return results


# ── Health Check ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "empathy-engine", "version": "1.1.0"}


# ── CLI Mode ──────────────────────────────────────────────
def run_cli(text: str):
    """CLI pipeline: text → per-sentence emotion → modulation → TTS → play."""
    print(f"\n{'='*60}")
    print(f"  🧠 Empathy Engine — CLI Mode (v1.1 with Intensity Scaling)")
    print(f"{'='*60}\n")
    print(f"  📝 Input : {text}\n")

    # 1. Per-sentence detection
    sentence_emotions = detect_emotions_per_sentence(text)

    print(f"  📋 Sentences detected: {len(sentence_emotions)}\n")

    for i, se in enumerate(sentence_emotions, 1):
        se.voice_params = modulate(se.emotion)
        print(f"  ── Sentence {i} ──")
        print(f"      Text     : {se.sentence}")
        print(f"      Emotion  : {se.emotion.emotion}")
        print(f"      Intensity: {se.emotion.intensity:.2f}")
        print(f"      Rate     : {se.voice_params.rate} WPM")
        print(f"      Pitch    : {se.voice_params.pitch:+.2f} semitones")
        print(f"      Volume   : {se.voice_params.volume:.3f}")
        print(f"      Style    : {se.voice_params.azure_style}")
        print()

    # 2. Overall dominant
    dominant = max(sentence_emotions, key=lambda se: se.emotion.intensity)
    print(f"  🎯 Dominant : {dominant.emotion.emotion} (intensity={dominant.emotion.intensity:.2f})")

    # 3. Synthesize
    if len(sentence_emotions) > 1:
        filename, engine_used = synthesize_multi(
            sentence_emotions, text, dominant.voice_params
        )
    else:
        filename, engine_used = synthesize(
            text, dominant.voice_params, dominant.emotion.emotion, dominant.emotion.intensity
        )

    filepath = OUTPUT_DIR / filename
    print(f"\n  🔊 Engine : {engine_used}")
    print(f"  💾 Output : {filepath}")
    print(f"\n{'='*60}\n")

    # 4. Auto-play (Windows)
    if sys.platform == "win32":
        try:
            os.startfile(str(filepath))
            print("  ▶  Playing audio…")
        except Exception:
            print("  ⚠  Could not auto-play. Open the file manually.")


# ── Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empathy Engine — Emotion-Aware TTS")
    parser.add_argument("--cli", type=str, help="Text to synthesize in CLI mode")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.cli:
        run_cli(args.cli)
    else:
        import uvicorn
        print(f"\n🧠 Empathy Engine starting at http://{args.host}:{args.port}\n")
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
