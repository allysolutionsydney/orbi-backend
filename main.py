from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from routes.chat import router as chat_router
from routes.memory import router as memory_router
from routes.user import router as user_router
from routes.vision import router as vision_router
from routes.voice import router as voice_router

app = FastAPI(
    title="ORBI API",
    description="The AI that orbits your life — backend",
    version="0.1.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8081").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(chat_router,   prefix="/chat",   tags=["Chat"])
app.include_router(memory_router, prefix="/memory", tags=["Memory"])
app.include_router(user_router,   prefix="/user",   tags=["User"])
app.include_router(vision_router, prefix="/vision", tags=["Vision"])
app.include_router(voice_router,  prefix="/voice",  tags=["Voice"])


@app.get("/")
def root():
    return {"status": "ORBI is online", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
