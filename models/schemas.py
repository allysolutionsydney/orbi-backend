from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str
    include_vision: Optional[bool] = False
    image_base64: Optional[str] = None  # for camera input from phone/glasses


class ChatResponse(BaseModel):
    reply: str
    memories_used: Optional[int] = 0
    audio_url: Optional[str] = None
    provider_used: Optional[str] = None  # which AI brain answered


class MemoryItem(BaseModel):
    id: Optional[str] = None
    user_id: str
    content: str
    type: str  # "conversation" | "observation" | "fact" | "preference"
    importance: Optional[float] = 0.5
    tags: Optional[List[str]] = []
    timestamp: Optional[datetime] = None


class MemorySearchRequest(BaseModel):
    user_id: str
    query: str
    limit: Optional[int] = 15


class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None
    occupation: Optional[str] = None
    personality_preference: Optional[str] = "warm_friend"
    interests: Optional[List[str]] = []
    preferences: Optional[dict] = {}
    facts: Optional[List[str]] = []
    orbi_persona_notes: Optional[str] = None

    # AI Brain settings — the core of ORBI's model-agnostic platform
    ai_provider: Optional[str] = "openai"        # "openai" | "anthropic" | "gemini" | "ollama" | "groq"
    ai_model: Optional[str] = None               # specific model override, e.g. "claude-sonnet-4-6"

    created_at: Optional[datetime] = None


class AIProviderUpdate(BaseModel):
    """Used to switch the user's AI brain from the settings screen."""
    ai_provider: str   # "openai" | "anthropic" | "gemini" | "ollama" | "groq"
    ai_model: Optional[str] = None  # optional model override


class VisionRequest(BaseModel):
    user_id: str
    image_base64: str
    prompt: Optional[str] = "What do you see? Be helpful and contextual."
    store_in_memory: Optional[bool] = True


class VisionResponse(BaseModel):
    description: str
    stored: bool = False


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class TranscribeRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None
