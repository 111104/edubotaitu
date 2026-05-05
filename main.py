from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import httpx
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """Сен — EduBot, Astana IT University колледжінің интеллектті оқу ассистентісің.
Сен студенттерге мына модульдер бойынша көмек бересің:
• КМ03/ПМ03 — Бағдарламалау
• КМ04/ПМ04 — Web-сайт
• КМ05/ПМ05 — Рефакторинг кода
• КМ06 — Микроконтроллер
• КМ07 — Мобильді қосымшалар

Студент қай тілде жазса, сол тілде жауап бер.
Тьютор рөлін атқар: дайын жауап берме, бағыттаушы сұрақ қой.
Жауаптың соңында тексеру сұрағын қой."""

MODULE_CONTEXTS = {
    "km03": "Қазір КМ03/ПМ03: бағдарламалау (Python/Java).",
    "km04": "Қазір КМ04/ПМ04: Web-сайт (HTML, CSS, JS).",
    "km05": "Қазір КМ05/ПМ05: рефакторинг, тестілеу.",
    "km06": "Қазір КМ06: микроконтроллер, Arduino.",
    "km07": "Қазір КМ07: мобильді қосымшалар, Flutter.",
}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    module: str = "auto"

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok", "gemini_key": bool(GEMINI_API_KEY)}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        return {"error": "API key not configured"}

    module_ctx = MODULE_CONTEXTS.get(req.module, "")
    system = SYSTEM_PROMPT + ("\n\n" + module_ctx if module_ctx else "")

    contents = []
    for m in req.messages:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7}
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)

    if response.status_code != 200:
        return {"error": f"Gemini қате: {response.status_code} — {response.text}"}

    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"response": text}
    except Exception:
        return {"error": f"Жауап форматы қате: {data}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
