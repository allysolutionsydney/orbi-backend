from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str
    include_vision: Optional[bool] = False
    image_base64: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    memories_used: Optional[int] = 0
    audio_url: Optional[str] = None


class MemoryItem(BaseModel):
    id: Optional[str] = None
    user_id: str
    content: str
    type: str
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
    personality_preference: Optional[str] = 'warm_friend'
    interests: Optional[List[str]] = []
    preferences: Optional[dict] = {}
    facts: Optional[List[str]] = []
    orbi_persona_notes: Optional[str] = None
    created_at: Optional[datetime] = None


class VisionRequest(BaseModel):
    user_id: str
    image_base64: str
    prompt: Optional[str] = None
    store_in_memory: Optional[bool] = True


class VisionResponse(BaseModel):
    description: str
    stored: bool = False
