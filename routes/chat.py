from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from models.schemas import ChatRequest, ChatResponse, ChatMessage
from services.claude import chat, chat_stream
from services.memory import search_memories, store_conversation_turn, auto_extract_and_store
from services.voice import text_to_speech
from supabase import create_client
import os
import json
from datetime import datetime

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


def get_conversation_history(user_id: str, limit: int = 20) -> list[ChatMessage]:
    """Fetch recent conversation history for this user from Supabase."""
    result = (
        supabase.table("conversations")
        .select("role,content,timestamp")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    rows = list(reversed(result.data or []))
    return [ChatMessage(role=r["role"], content=r["content"]) for r in rows]


def save_message(user_id: str, role: str, content: str):
    """Persist a message to conversation history."""
    supabase.table("conversations").insert({
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }).execute()


def get_user_profile(user_id: str) -> dict:
    result = supabase.table("users").select("*").eq("user_id", user_id).limit(1).execute()
    return result.data[0] if result.data else {}


@router.post("/", response_model=ChatResponse)
async def send_message(req: ChatRequest):
    """Main chat endpoint. Retrieves memories, calls Claude, stores response."""
    try:
        # 1. Fetch context
        history = get_conversation_history(req.user_id)
        memories = search_memories(req.user_id, req.message, limit=15)
        profile = get_user_profile(req.user_id)

        # 2. Get ORBI's reply
        reply = chat(
            message=req.message,
            conversation_history=history,
            memories=memories,
            user_profile=profile,
            image_base64=req.image_base64 if req.include_vision else None,
        )

        # 3. Persist both turns
        save_message(req.user_id, "user", req.message)
        save_message(req.user_id, "assistant", reply)

        # 4. Auto-extract memorable facts in background
        conversation_snippet = f"User: {req.message}\nORBI: {reply}"
        try:
            auto_extract_and_store(req.user_id, conversation_snippet)
        except Exception:
            pass  # Don't fail the response if memory extraction fails

        return ChatResponse(reply=reply, memories_used=len(memories))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice", response_class=Response)
async def send_message_voice(req: ChatRequest):
    """
    Same as /chat but returns MP3 audio of ORBI's response.
    The mobile app plays this directly through the speaker/earbuds.
    """
    try:
        history = get_conversation_history(req.user_id)
        memories = search_memories(req.user_id, req.message, limit=15)
        profile = get_user_profile(req.user_id)

        reply = chat(
            message=req.message,
            conversation_history=history,
            memories=memories,
            user_profile=profile,
        )

        save_message(req.user_id, "user", req.message)
        save_message(req.user_id, "assistant", reply)

        # Convert reply to audio
        audio_bytes = await text_to_speech(reply)

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"X-Orbi-Reply": reply[:200]},  # Text reply in header too
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
async def stream_message(user_id: str, message: str):
    """Server-sent events stream for real-time token-by-token response."""
    async def event_generator():
        history = get_conversation_history(user_id)
        memories = search_memories(user_id, message, limit=15)
        profile = get_user_profile(user_id)

        full_reply = []
        async for token in chat_stream(message, history, memories, profile):
            full_reply.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Store after stream completes
        complete_reply = "".join(full_reply)
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", complete_reply)
        yield f"data: {json.dumps({'done': True, 'full_reply': complete_reply})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Fetch conversation history for the chat screen."""
    result = (
        supabase.table("conversations")
        .select("role,content,timestamp")
        .eq("user_id", user_id)
        .order("timestamp", desc=False)
        .limit(limit)
        .execute()
    )
    return {"messages": result.data or []}


@router.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear conversation history (not memories — those persist)."""
    supabase.table("conversations").delete().eq("user_id", user_id).execute()
    return {"cleared": True}
