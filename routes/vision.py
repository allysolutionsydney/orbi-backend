from fastapi import APIRouter, HTTPException
from models.schemas import VisionRequest, VisionResponse
from services.vision import analyse_image, extract_text_from_image, identify_objects
from services.memory import search_memories, auto_extract_and_store
from services.claude import chat
from supabase import create_client
import os

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
