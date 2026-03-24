"""
ORBI Voice Service — OpenAI only
"""
import os
import io
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TTS_MODEL = "tts-1"
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")


async def text_to_speech(text: str) -> bytes:
    response = _client.audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=text, response_format="mp3")
    return response.content


async def speech_to_text(audio_bytes: bytes, filename: str = "audio.m4a") -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    transcript = _client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    return transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
