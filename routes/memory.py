from fastapi import APIRouter, HTTPException, Query
from models.schemas import MemoryItem, MemorySearchRequest
from services.memory import (
    store_memory,
    search_memories,
    get_all_memories,
    delete_memory,
    auto_extract_and_store,
)
from supabase import create_client
import os

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


@router.post("/", response_model=dict)
async def add_memory(memory: MemoryItem):
    """Manually store a memory for a user."""
    try:
        memory_id = store_memory(memory)
        return {"id": memory_id, "stored": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=dict)
async def list_memories(user_id: str, limit: int = Query(default=100, le=500)):
    """Retrieve all memories for a user, most recent first."""
    try:
        memories = get_all_memories(user_id, limit=limit)
        return {
            "memories": [m.model_dump() for m in memories],
            "count": len(memories),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=dict)
async def search(req: MemorySearchRequest):
    """Semantic search — find memories most relevant to a query."""
    try:
        memories = search_memories(req.user_id, req.query, limit=req.limit or 15)
        return {
            "memories": [m.model_dump() for m in memories],
            "count": len(memories),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}/{memory_id}")
async def remove_memory(user_id: str, memory_id: str):
    """Delete a specific memory. Users control what ORBI remembers."""
    try:
        delete_memory(memory_id, user_id)
        return {"deleted": True, "id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def clear_all_memories(user_id: str):
    """Wipe all memories for a user (full reset)."""
    try:
        supabase.table("memories").delete().eq("user_id", user_id).execute()
        return {"cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
