"""
ORBI AI Provider Abstraction Layer
===================================
This is the core of ORBI's model-agnostic architecture.
Users can choose their preferred AI brain — OpenAI, Anthropic Claude,
Google Gemini, or any OpenAI-compatible model (Ollama, Together, Groq, etc.)

Adding a new provider:
1. Create a class that extends AIProvider
2. Implement chat(), stream(), and raw_completion()
3. Register it in PROVIDER_REGISTRY
"""