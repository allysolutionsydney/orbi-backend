from fastapi import APIRouter, HTTPException
from models.schemas import UserProfile
from supabase import create_client
import os
import uuid
from datetime import datetime

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


@router.post("/", response_model=dict)
async def create_user(profile: UserProfile):
    """
    Create or upsert a user profile.
    Called on first app launch / after sign-up.
    """
    try:
        user_id = profile.user_id or str(uuid.uuid4())
        data = {
            "user_id": user_id,
            "name": profile.name,
            "age": profile.age,
            "location": profile.location,
            "occupation": profile.occupation,
            "personality_preference": profile.personality_preference,
            "interests": profile.interests or [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        supabase.table("users").upsert(data, on_conflict="user_id").execute()
        return {"user_id": user_id, "created": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=dict)
async def get_user(user_id: str):
    """Fetch a user's full profile."""
    try:
        result = (
            supabase.table("users")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{user_id}", response_model=dict)
async def update_user(user_id: str, updates: dict):
    """
    Partial update of a user profile.
    Body: any subset of profile fields (e.g. { "name": "Alex", "location": "Sydney" })
    """
    try:
        updates["updated_at"] = datetime.utcnow().isoformat()
        # Remove fields that shouldn't be overwritten via this endpoint
        updates.pop("user_id", None)
        updates.pop("created_at", None)

        supabase.table("users").update(updates).eq("user_id", user_id).execute()
        return {"updated": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """
    Delete a user and ALL their data (conversations + memories).
    Cascading delete — use with caution.
    """
    try:
        supabase.table("conversations").delete().eq("user_id", user_id).execute()
        supabase.table("memories").delete().eq("user_id", user_id).execute()
        supabase.table("users").delete().eq("user_id", user_id).execute()
        return {"deleted": True, "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/summary")
async def user_summary(user_id: str):
    """
    Return a quick dashboard summary: profile + counts.
    Used by the Settings screen in the mobile app.
    """
    try:
        profile_result = (
            supabase.table("users")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        profile = profile_result.data[0] if profile_result.data else {}

        memory_result = (
            supabase.table("memories")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        convo_result = (
            supabase.table("conversations")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )

        return {
            "profile": profile,
            "memory_count": memory_result.count or 0,
            "message_count": convo_result.count or 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{user_id}/personality")
async def update_personality(user_id: str, body: dict):
    """
    Update ORBI's personality preference for this user.
    Body: { "personality_preference": "warm_friend" | "professional_assistant" | "motivational_coach" }
    """
    try:
        preference = body.get("personality_preference")
        if not preference:
            raise HTTPException(status_code=400, detail="personality_preference required")
        supabase.table("users").update({
            "personality_preference": preference,
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("user_id", user_id).execute()
        return {"updated": True, "personality_preference": preference}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
