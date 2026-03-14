"""Test Azure TTS with each emotion style to verify which work correctly."""
import os
import azure.cognitiveservices.speech as speechsdk
from app.config import settings

key = settings.azure_speech_key
region = settings.azure_speech_region
voice = settings.azure_voice_name

print(f"Voice: {voice}")
print(f"Region: {region}")
print()

test_cases = [
    ("cheerful", "I am so happy today! This is wonderful!"),
    ("angry", "This is completely unacceptable! I am furious!"),
    ("sad", "I feel so alone and broken. Nothing matters anymore."),
    ("terrified", "Oh no, something terrible is happening. I am so scared."),
    ("excited", "Wow, I cannot believe this! This is amazing!"),
]

for style, text in test_cases:
    ssml = (
        f'<speak version="1.0"'
        f' xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xmlns:mstts="https://www.w3.org/2001/mstts"'
        f' xml:lang="en-US">'
        f'<voice name="{voice}">'
        f'<mstts:express-as style="{style}" styledegree="2">'
        f'<prosody rate="+0%" pitch="+0st" volume="medium">'
        f'{text}'
        f'</prosody>'
        f'</mstts:express-as>'
        f'</voice>'
        f'</speak>'
    )

    outfile = f"output/test_{style}.wav"
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=outfile)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        size = os.path.getsize(outfile)
        print(f"  ✅ {style:12s} → {size:>7} bytes  OK")
    elif result.reason == speechsdk.ResultReason.Canceled:
        c = result.cancellation_details
        print(f"  ❌ {style:12s} → CANCELED: {c.reason} | {c.error_details}")
    else:
        print(f"  ⚠️  {style:12s} → Unknown: {result.reason}")

print()
print("Done! Check output/test_*.wav files to hear the differences.")
