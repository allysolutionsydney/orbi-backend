from fastapi import APIRouter, HTTPException
from models.schemas import VisionRequest, VisionResponse
from services.vision import analyse_image, extract_text_from_image, identify_objects
from services.memory import search_memories, auto_extract_and_store
from services.claude import chat
from supabase import create_client
import os

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


def get_user_context(user_id: str) -> str:
    """Build a short context string from user profile for vision system."""
    result = supabase.table("users").select("*").eq("user_id", user_id).limit(1).execute()
    if not result.data:
        return ""
    p = result.data[0]
    parts = []
    if p.get("name"):
        parts.append(f"User's name is {p['name']}.")
    if p.get("occupation"):
        parts.append(f"They work as a {p['occupation']}.")
    if p.get("location"):
        parts.append(f"Based in {p['location']}.")
    return " ".join(parts)


@router.post("/analyse", response_model=VisionResponse)
async def analyse(req: VisionRequest):
    """
    Analyse an image and return ORBI's description.
    Optionally takes a custom prompt; defaults to a general helpful description.
    """
    try:
        user_context = get_user_context(req.user_id) if req.user_id else None
        description = await analyse_image(
            req.image_base64,
            prompt=req.prompt or "What do you see? Describe it helpfully and concisely.",
            user_context=user_context,
        )
        return VisionResponse(description=description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=VisionResponse)
async def ocr(req: VisionRequest):
    """Extract all text visible in an image (OCR)."""
    try:
        text = await extract_text_from_image(req.image_base64)
        return VisionResponse(description=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify", response_model=dict)
async def identify(req: VisionRequest):
    """Identify objects/items in an image and return as a list."""
    try:
        items = await identify_objects(req.image_base64)
        return {"objects": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=dict)
async def vision_chat(req: VisionRequest):
    """
    Full conversational response about an image — ORBI sees the image
    and responds using memories + user profile, just like a chat message.
    The image and user's prompt are sent together to Claude.
    """
    try:
        if not req.user_id:
            raise HTTPException(status_code=400, detail="user_id required for vision chat")

        # Fetch context
        memories = search_memories(req.user_id, req.prompt or "image", limit=10)
        profile_result = supabase.table("users").select("*").eq("user_id", req.user_id).limit(1).execute()
        profile = profile_result.data[0] if profile_result.data else {}

        reply = chat(
            message=req.prompt or "What do you see in this image?",
            conversation_history=[],
            memories=memories,
            user_profile=profile,
            image_base64=req.image_base64,
        )

        # Store this vision interaction as a memory
        try:
            snippet = f"User shared an image and asked: {req.prompt}\nORBI replied: {reply}"
            auto_extract_and_store(req.user_id, snippet)
        except Exception:
            pass

        return {"reply": reply}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
