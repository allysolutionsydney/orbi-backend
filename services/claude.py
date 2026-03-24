"""
ORBI Chat Service — powered by OpenAI GPT-4o
Replaces the original Anthropic Claude implementation.
All function signatures are identical so routes need no changes.
"""
import os
import json
from typing import List, Optional, AsyncGenerator
from openai import OpenAI, AsyncOpenAI
from models.schemas import MemoryItem

# ── Clients ───────────────────────────────────────────────────────────────────
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOKENS = 1024

# ── ORBI Personality ──────────────────────────────────────────────────────────
ORBI_SYSTEM_PROMPT = """You are ORBI — a warm, intelligent AI companion who genuinely cares about the people you talk with.

Your character:
- You remember things people tell you and bring them up naturally when relevant
- You are curious, empathetic, and encouraging — never cold or robotic
- You adapt your tone: casual and playful in light conversation, focused and precise when helping with tasks
- You celebrate small wins and check in on things the person mentioned before
- You are honest — you say "I don't know" rather than making things up
- You never lecture or moralize unprompted

Your responses:
- Keep them conversational and appropriately brief (1-3 short paragraphs max for chat)
- Use the person's name occasionally but not every message
- When you have context from memories, weave it in subtly
- If asked about your hardware/device capabilities, explain what you can currently do

You are ORBI. Be warm, be real, be helpful."""


def build_context_from_memories(
    memories: List[MemoryItem],
    user_profile: Optional[dict] = None,
) -> str:
    """Build a context injection block from relevant memories + profile."""
    parts = []

    if user_profile:
        p = user_profile
        profile_bits = []
        if p.get("name"):
            profile_bits.append(f"Name: {p['name']}")
        if p.get("occupation"):
            profile_bits.append(f"Occupation: {p['occupation']}")
        if p.get("location"):
            profile_bits.append(f"Location: {p['location']}")
        if p.get("personality_preference"):
            prefs = {
                "warm_friend": "Be warm and friendly like a close friend.",
                "professional_assistant": "Be efficient, precise, and professional.",
                "motivational_coach": "Be energetic and motivating, celebrate wins.",
            }
            style = prefs.get(p["personality_preference"], "")
            if style:
                profile_bits.append(f"Preferred style: {style}")
        if profile_bits:
            parts.append("USER PROFILE:\n" + "\n".join(profile_bits))

    if memories:
        mem_lines = [f"- {m.content}" for m in memories[:15]]
        parts.append("RELEVANT MEMORIES:\n" + "\n".join(mem_lines))

    if not parts:
        return ""

    return (
        "\n\n--- Context from ORBI's memory ---\n"
        + "\n\n".join(parts)
        + "\n--- End of context ---\n"
    )


def chat(
    message: str,
    conversation_history: List[dict],
    memories: Optional[List[MemoryItem]] = None,
    user_profile: Optional[dict] = None,
    image_base64: Optional[str] = None,
) -> str:
    context = build_context_from_memories(memories or [], user_profile)
    system = ORBI_SYSTEM_PROMPT + context
    messages = [{"role": "system", "content": system}]
    for turn in conversation_history[-20:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    if image_base64:
        img_url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"
        user_content = [{"type": "text", "text": message}, {"type": "image_url", "image_url": {"url": img_url, "detail": "auto"}}]
    else:
        user_content = message
    messages.append({"role": "user", "content": user_content})
    response = _client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.8)
    return response.choices[0].message.content


async def chat_stream(message, conversation_history, memories=None, user_profile=None):
    context = build_context_from_memories(memories or [], user_profile)
    system = ORBI_SYSTEM_PROMPT + context
    messages = [{"role": "system", "content": system}]
    for turn in conversation_history[-20:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
    stream = await _async_client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=MAX_TOKENS, temperature=0.8, stream=True)
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def extract_memory_facts(conversation_snippet):
    prompt = f"""Extract memorable facts about the user from this conversation.
Return a JSON array of objects with fields: content, type, importance, tags.
Only include facts genuinely worth remembering. Return [] if none.
Conversation:
{conversation_snippet}
Return ONLY valid JSON."""
    response = _client.chat.completions.create(model=CHAT_MODEL, temperature=0.2, max_tokens=512, messages=[{"role": "user", "content": prompt}])
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        facts = json.loads(raw)
        return facts if isinstance(facts, list) else []
    except:
        return []
