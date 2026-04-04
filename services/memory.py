import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from supabase import create_client, Client
from models.schemas import MemoryItem

logger = logging.getLogger(__name__)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ── OpenAI embeddings (optional) ────────────────────────────────────────────
# If OPENAI_API_KEY is missing or unreachable, semantic search gracefully
# falls back to returning the most recent memories instead of crashing.

_openai_client = None
_embeddings_available = False

try:
    from openai import OpenAI
    _key = os.getenv("OPENAI_API_KEY")
    if _key:
        _openai_client = OpenAI(api_key=_key)
        _embeddings_available = True
    else:
        logger.warning("OPENAI_API_KEY not set — semantic search disabled, using recency fallback")
except Exception as _e:
    logger.warning(f"Could not initialise OpenAI client: {_e}")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Generate a vector embedding via OpenAI.
    Returns None (instead of raising) if the service is unavailable.
    """
    if not _embeddings_available or not _openai_client:
        return None
    try:
        response = _openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Embedding request failed ({e}) — falling back to recency search")
        return None


def store_memory(memory: MemoryItem) -> str:
    """Store a memory with its vector embedding in Supabase."""
    embedding = get_embedding(memory.content)
    memory_id = str(uuid.uuid4())

    data = {
        "id": memory_id,
        "user_id": memory.user_id,
        "content": memory.content,
        "type": memory.type,
        "importance": memory.importance or 0.5,
        "tags": memory.tags or [],
        "timestamp": (memory.timestamp or datetime.utcnow()).isoformat(),
    }
    if embedding is not None:
        data["embedding"] = embedding

    supabase.table("memories").insert(data).execute()
    return memory_id


def search_memories(user_id: str, query: str, limit: int = 15) -> List[MemoryItem]:
    """
    Semantic search — find the most relevant memories for this query.
    Falls back to most-recent memories if OpenAI embeddings are unavailable.
    """
    embedding = get_embedding(query)

    if embedding is not None:
        # ── Vector similarity search ─────────────────────────────────────
        try:
            result = supabase.rpc(
                "match_memories",
                {
                    "query_embedding": embedding,
                    "match_user_id": user_id,
                    "match_count": limit,
                },
            ).execute()
        except Exception as e:
            logger.warning(f"Vector RPC failed ({e}) — falling back to recency")
            result = (
                supabase.table("memories")
                .select("*")
                .eq("user_id", user_id)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
    else:
        # ── Recency fallback (no embeddings) ─────────────────────────────
        result = (
            supabase.table("memories")
            .select("*")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )

    memories = []
    for row in result.data or []:
        try:
            memories.append(MemoryItem(
                id=row["id"],
                user_id=row["user_id"],
                content=row["content"],
                type=row.get("type", "fact"),
                importance=row.get("importance", 0.5),
                tags=row.get("tags", []),
                timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else None,
            ))
        except Exception:
            continue

    return memories


def get_all_memories(user_id: str, limit: int = 100) -> List[MemoryItem]:
    """Retrieve all memories for a user, most recent first."""
    result = (
        supabase.table("memories")
        .select("*")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    return [
        MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            type=row.get("type", "fact"),
            importance=row.get("importance", 0.5),
            tags=row.get("tags", []),
            timestamp=datetime.fromisoformat(row["timestamp"]) if row.get("timestamp") else None,
        )
        for row in result.data or []
    ]


def delete_memory(memory_id: str, user_id: str) -> bool:
    """Delete a specific memory (user can remove things ORBI remembers)."""
    supabase.table("memories").delete().eq("id", memory_id).eq("user_id", user_id).execute()
    return True


def store_conversation_turn(user_id: str, user_msg: str, orbi_reply: str) -> str:
    """Store a conversation exchange as a single memory."""
    content = f"User said: {user_msg}\nORBI replied: {orbi_reply}"
    memory = MemoryItem(
        user_id=user_id,
        content=content,
        type="conversation",
        importance=0.4,
        tags=["conversation"],
    )
    return store_memory(memory)


def auto_extract_and_store(user_id: str, conversation: str):
    """After a conversation, extract important facts and store them."""
    from services.claude import extract_memory_facts
    facts = extract_memory_facts(conversation)

    stored = []
    for fact in facts:
        memory = MemoryItem(
            user_id=user_id,
            content=fact.get("content", ""),
            type=fact.get("type", "fact"),
            importance=fact.get("importance", 0.6),
            tags=fact.get("tags", []),
        )
        if memory.content:
            stored.append(store_memory(memory))

    return stored
