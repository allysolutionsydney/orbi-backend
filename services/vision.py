"""
ORBI Vision Service — GPT-4o Vision
Replaces the original Anthropic Claude Vision implementation.
All function signatures are identical so routes need no changes.
"""
import os
import json
from typing import Optional, List
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VISION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

ORBI_VISION_CONTEXT = (
    "You are ORBI, a warm and helpful AI companion. "
    "When analysing images, be clear and useful. "
    "If you know something about the user's context, factor it in."
)


def _build_image_url(image_base64: str) -> str:
    """Normalise image input to a data URI."""
    if image_base64.startswith("data:"):
        return image_base64
    return f"data:image/jpeg;base64,{image_base64}"


async def analyse_image(
    image_base64: str,
    prompt: Optional[str] = None,
    user_context: Optional[str] = None,
) -> str:
    """
    Describe / analyse an image with GPT-4o Vision.
    Returns a plain-text description.
    """
    system = ORBI_VISION_CONTEXT
    if user_context:
        system += f" Context about the user: {user_context}"

    user_prompt = prompt or "What do you see? Describe it helpfully and concisely."

    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _build_image_url(image_base64),
                            "detail": "auto",
                        },
                    },
                ],
            },
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content


async def extract_text_from_image(image_base64: str) -> str:
    """OCR — extract all visible text from an image."""
    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text visible in this image exactly as it appears. "
                            "Return only the extracted text, nothing else. "
                            "If there is no text, return an empty string."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _build_image_url(image_base64),
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


async def identify_objects(image_base64: str) -> List[str]:
    """
    Identify distinct objects/items in an image.
    Returns a list of object name strings.
    """
    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "List the distinct objects, items, or entities visible in this image. "
                            "Return a JSON array of short label strings, e.g. [\"coffee mug\",\"laptop\"]. "
                            "Return ONLY the JSON array, nothing else."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _build_image_url(image_base64),
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        items = json.loads(raw)
        return items if isinstance(items, list) else []
    except json.JSONDecodeError:
        # Fall back: split by newlines if JSON parsing fails
        return [line.strip("- •").strip() for line in raw.splitlines() if line.strip()]
