"""
ORBI Chat Orchestration Service
=================================
This module handles the chat logic: building context from memories + user
profile, calling the AI provider, and returning responses.

The AI provider is now fully swappable — pass provider_name + model_name
and ORBI will use whichever brain the user has chosen.
"""
import json
from typing import List, Optional, AsyncGenerator
from models.schemas import MemoryItem, ChatMessage
from services.ai_provider import get_provider, AIProvider

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


def _build_messages(
    message: str,
    conversation_history: List[ChatMessage],
    memories: Optional[List[MemoryItem]] = None,
    user_profile: Optional[dict] = None,
    image_base64: Optional[str] = None,
) -> List[dict]:
    """Build the full messages list for the AI provider."""
    context = build_context_from_memories(memories or [], user_profile)
    system = ORBI_SYSTEM_PROMPT + context

    messages = [{"role": "system", "content": system}]

    # Add conversation history
    for turn in conversation_history[-20:]:
        role = turn.role if hasattr(turn, "role") else turn["role"]
        content = turn.content if hasattr(turn, "content") else turn["content"]
        messages.append({"role": role, "content": content})

    # Build user message — with optional image (vision input from glasses/camera)
    if image_base64:
        img_url = (
            image_base64
            if image_base64.startswith("data:")
            else f"data:image/jpeg;base64,{image_base64}"
        )
        user_content = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": img_url, "detail": "auto"}},
        ]
    else:
        user_content = message

    messages.append({"role": "user", "content": user_content})
    return messages


def chat(
    message: str,
    conversation_history: List[ChatMessage],
    memories: Optional[List[MemoryItem]] = None,
    user_profile: Optional[dict] = None,
    image_base64: Optional[str] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Send a message to the user's chosen AI provider and return the reply.

    provider_name: "openai" | "anthropic" | "gemini" | "ollama" | "groq" | ...
    model_name:    specific model override (e.g. "gpt-4o-mini", "claude-haiku-4-5-20251001")
    """
    provider = get_provider(provider_name, model_name)
    messages = _build_messages(
        message, conversation_history, memories, user_profile, image_base64
    )
    return provider.chat(messages, max_tokens=MAX_TOKENS, temperature=0.8)


async def chat_stream(
    message: str,
    conversation_history: List[ChatMessage],
    memories: Optional[List[MemoryItem]] = None,
    user_profile: Optional[dict] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields token chunks for SSE streaming."""
    provider = get_provider(provider_name, model_name)
    messages = _build_messages(message, conversation_history, memories, user_profile)
    async for token in provider.stream(messages, max_tokens=MAX_TOKENS, temperature=0.8):
        yield token


def extract_memory_facts(conversation_snippet: str, provider_name: Optional[str] = None) -> List[dict]:
    """
    Extract memorable facts from a conversation snippet.
    Returns a list of dicts: [{content, type, importance, tags}]
    """
    prompt = f"""Extract memorable facts about the user from this conversation.
Return a JSON array of objects with fields
  content    (string) — the fact, written as a statement about the user
  type       (string) - one of: fact, preference, goal, event, conversation
  importance (float)  — 0.0 to 1.0
  tags       (array)  — relevant keyword tags

Only include facts genuinely worth remembering long-term.
Return [] if there are none worth storing.

Conversation:
{conversation_snippet}

Return ONLY valid JSON, nothing else."""

    provider = get_provider(provider_name)
    raw = provider.raw_completion(prompt, max_tokens=512, temperature=0.2).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        facts = json.loads(raw)
        return facts if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []
