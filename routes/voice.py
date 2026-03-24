from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from services.voice import speech_to_text, text_to_speech
import base64

router = APIRouter()


class TranscribeRequest(BaseModel):
    audio_base64: str
    filename: str = "audio.m4a"  # Extension tells Whisper the format


class TTSRequest(BaseModel):
    text: str


@router.post("/transcribe")
async def transcribe_audio(req: TranscribeRequest):
    """
    Transcribe audio to text using Whisper.
    Accepts base64-encoded audio (m4a/webm/mp4 etc).
    Returns { transcript: str }.
    """
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        transcript = await speech_to_text(audio_bytes, req.filename)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speak", response_class=Response)
async def speak_text(req: TTSRequest):
    """
    Convert text to speech (OpenAI TTS).
    Returns raw MP3 bytes.
    Useful for standalone TTS without a full chat round-trip.
    """
    try:
        audio_bytes = await text_to_speech(req.text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


