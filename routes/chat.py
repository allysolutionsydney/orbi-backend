from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from models.schemas import ChatRequest, ChatResponse, ChatMessage
from services.claude import chat, chat_stream
from services.memory import search_memories, store_conversation_turn, auto_extract_and_store
from services.voice import text_to_speech
from services.ai_provider import list_providers
from supabase import create_client
import os
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


def get_conversation_history(user_id: str, limit: int = 20) -> list[ChatMessage]:
    """Fetch recent conversation history for this user from Supabase."""
    try:
        result = (
            supabase.table("conversations")
            .select("role,content,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(reversed(result.data or []))
        return [ChatMessage(role=r["role"], content=r["content"]) for r in rows]
    except Exception as e:
        logger.warning(f"Could not fetch conversation history: {e}")
        return []


def save_message(user_id: str, role: str, content: str):
    """Persist a message to conversation history. Silently skips on DB errors."""
    try:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "role": role,
            "content": content,
        }).execute()
    except Exception as e:
        logger.warning(f"Could not save message (user may not exist in users table yet): {e}")


def get_user_profile(user_id: str) -> dict:
    """Fetch the user's full profile including their chosen AI provider.
    Returns {} gracefully if the users table doesn't exist yet."""
    try:
        result = supabase.table("users").select("*").eq("user_id", user_id).limit(1).execute()
        return result.data[0] if result.data else {}
    except Exception as e:
        logger.warning(f"Could not fetch user profile (users table may not exist yet): {e}")
        return {}


@router.post("/", response_model=ChatResponse)
async def send_message(req: ChatRequest):
    """
    Main chat endpoint.
    Automatically uses the user's chosen AI provider (OpenAI, Claude, Gemini, etc.)
    """
    try:
        # 1. Fetch context
        history = get_conversation_history(req.user_id)
        memories = search_memories(req.user_id, req.message, limit=15)
        profile = get_user_profile(req.user_id)

        # 2. Get provider preference from user profile
        provider_name = profile.get("ai_provider") or None
        model_name = profile.get("ai_model") or None

        # 3. Get ORBI's reply using the user's chosen AI brain
        reply = chat(
            message=req.message,
            conversation_history=history,
            memories=memories,
            user_profile=profile,
            image_base64=req.image_base64 if req.include_vision else None,
            provider_name=provider_name,
            model_name=model_name,
        )

        # 4. Persist both turns
        save_message(req.user_id, "user", req.message)
        save_message(req.user_id, "assistant", reply)

        # 5. Auto-extract memorable facts in background
        conversation_snippet = f"User: {req.message}\nORBI: {reply}"
        try:
            auto_extract_and_store(req.user_id, conversation_snippet)
        except Exception:
            pass  # Don't fail the response if memory extraction fails

        return ChatResponse(reply=reply, memories_used=len(memories))

    except Exception as e:
        logger.error(f"Chat error ({type(e).__name__}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/voice", response_class=Response)
async def send_message_voice(req: ChatRequest):
    """
    Same as /chat but returns MP3 audio of ORBI's response.
    The mobile app / glasses play this directly through the speaker/earbuds.
    """
    try:
        history = get_conversation_history(req.user_id)
        memories = search_memories(req.user_id, req.message, limit=15)
        profile = get_user_profile(req.user_id)

        provider_name = profile.get("ai_provider") or None
        model_name = profile.get("ai_model") or None

        reply = chat(
            message=req.message,
            conversation_history=history,
            memories=memories,
            user_profile=profile,
            provider_name=provider_name,
            model_name=model_name,
        )

        save_message(req.user_id, "user", req.message)
        save_message(req.user_id, "assistant", reply)

        # Convert reply to audio for glasses/earbuds
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

        provider_name = profile.get("ai_provider") or None
        model_name = profile.get("ai_model") or None

        full_reply = []
        async for token in chat_stream(
            message, history, memories, profile,
            provider_name=provider_name,
            model_name=model_name,
        ):
            full_reply.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

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
        .select("role,content,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return {"messages": result.data or []}


@router.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear conversation history (not memories — those persist)."""
    supabase.table("conversations").delete().eq("user_id", user_id).execute()
    return {"cleared": True}


@router.get("/providers")
async def get_providers():
    """
    Return all available AI providers and their models.
    Used by the mobile app settings screen so users can pick their AI brain.
    """
    return {"providers": list_providers()}
