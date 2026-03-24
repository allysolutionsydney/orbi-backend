import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
from supabase import create_client, Client
from openai import OpenAI
from models.schemas import MemoryItem

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str):
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def store_memory(memory: MemoryItem) -> str:
    embedding = get_embedding(memory.content)
    memory_id = str(uuid.uuid4())
    supabase.table("memories").insert({"id": memory_id, "user_id": memory.user_id, "content": memory.content, "type": memory.type, "importance": memory.importance or 0.5, "tags": memory.tags or [], "embedding": embedding, "timestamp": (memory.timestamp or datetime.utcnow()).isoformat()}).execute()
    return memory_id


def search_memories(user_id: str, query: str, limit: int = 15) -> List[MemoryItem]:
    query_embedding = get_embedding(query)
    result = supabase.rpc("match_memories", {"query_embedding": query_embedding, "match_user_id": user_id, "match_count": limit}).execute()
    memories = []
    for row in result.data or []:
        memories.append(MemoryItem(id=row["id"], user_id=row["user_id"], content=row["content"], type=row.get("type", "fact"), importance=row.get("importance", 0.5), tags=row.get("tags", []), timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else None))
    return memories


def get_all_memories(user_id: str, limit: int = 100) -> List[MemoryItem]:
    result = supabase.table("memories").select("*").eq("user_id", user_id).order("timestamp", desc=True).limit(limit).execute()
    return [MemoryItem(id=row["id"], user_id=row["user_id"], content=row["content"], type=row.get("type", "fact"), importance=row.get("importance", 0.5), tags=row.get("tags", []), timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else None) for row in result.data or []]


def delete_memory(memory_id: str, user_id: str):
    supabase.table("memories").delete().eq("id", memory_id).eq("user_id", user_id).execute()


def store_conversation_turn(user_id, user_msg, orbi_reply):
    content = f"User said: {user_msg}\nORBI replied: {orbi_reply}"
    return store_memory(MemoryItem(user_id=user_id, content=content, type="conversation", importance=0.4, tags=["conversation"]))


def auto_extract_and_store(user_id, conversation):
    from services.claude import extract_memory_facts
    facts = extract_memory_facts(conversation)
    stored = []
    for fact in facts:
        memory = MemoryItem(user_id=user_id, content=fact.get("content", ""), type=fact.get("type", "fact"), importance=fact.get("importance", 0.6), tags=fact.get("tags", []))
        if memory.content:
            stored.append(store_memory(memory))
    return stored
