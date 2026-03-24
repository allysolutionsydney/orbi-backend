"""ORBI Vision Service — GPT-4o Vision"""
import os
import json
from typing import Optional, List
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VISION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ORBI_VISION_CONTEXT = "You are ORBI, a warm and helpful AI companion. When analysing images, be clear and useful."


def _build_image_url(image_base64):
    return image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"


async def analyse_image(image_base64, prompt=None, user_context=None):
    system = ORBI_VISION_CONTEXT
    if user_context: system += f" Context: {user_context}"
    response = _client.chat.completions.create(model=VISION_MODEL, max_tokens=512, messages=[{"role": "system", "content": system}, {"role": "user", "content": [{"type": "text", "text": prompt or "What do you see?"}, {"type": "image_url", "image_url": {"url": _build_image_url(image_base64), "detail": "auto"}}]}])
    return response.choices[0].message.content


async def extract_text_from_image(image_base64):
    response = _client.chat.completions.create(model=VISION_MODEL, tokens=1024, messages=[{"role": "user", "content": [{"type": "text", "text": "Extract all text from this image exactly. Return only the text."}, {"type": "image_url", "image_url": {"url": _build_image_url(image_base64), "detail": "high"}}]}])
    return response.choices[0].message.content.strip()


async def identify_objects(image_base64):
    response = _client.chat.completions.create(model=VISION_MODEL, max_tokens=256, messages=[{"role": "user", "content": [{"type": "text", "text": "List objects in this image as a JSON array of strings. Return ONLY JSON."}, {"type": "image_url", "image_url": {"url": _build_image_url(image_base64), "detail": "auto"}}]}])
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"): raw = raw.split("```")[1]; raw = raw[4:] if raw.startswith("json") else raw
    try: return json.loads(raw) if isinstance(json.loads(raw), list) else []
    except: return []
