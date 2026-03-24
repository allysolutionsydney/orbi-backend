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
