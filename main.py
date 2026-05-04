from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("AIzaSyDDuF5AoJHk6Zn4lF5thZSSu8iydb9oqyo")

SYSTEM_PROMPT = """Сен — EduBot, Astana IT University колледжінің интеллектті оқу ассистентісің.
Сен студенттерге мына модульдер бойынша көмек бересің:
• КМ03 / ПМ03 — Бағдарламалық қамтамасыз ету модульдерін бағдарламалау
• КМ04 / ПМ04 — Web-сайтты жобалау және үздіксіз жұмыс істеуін қамтамасыз ету
• КМ05 / ПМ05 — Бағдарламалық кодтың жұмыс жасау рефакторингін тексеру
• КМ06 — Микроконтроллер негізінде сандық құрылғыларды бағдарламалау
• КМ07 — Мобильді қосымшаларды әзірлеу

ТІЛІҢДІ АНЫҚТА: Студент қай тілде жазса, сол тілде жауап бер (қазақша, орысша немесе ағылшынша).

ПЕДАГОГИКАЛЫҚ СТИЛЬ:
- Дайын жауапты бірден берме — алдымен бағыттаушы сұрақ қой
- Студент қате жасаса — "қате" деме, "ал мынадай жағдайда не болады?" деп сұра
- Тьютор рөлін атқар: бағыттай, түсіндір, тексер
- Жауаптың соңында студентке кері сұрақ қой"""

MODULE_CONTEXTS = {
    "km03": "Қазір КМ03/ПМ03: бағдарламалау (Python/Java, алгоритмдер).",
    "km04": "Қазір КМ04/ПМ04: Web-сайт (HTML, CSS, JavaScript).",
    "km05": "Қазір КМ05/ПМ05: рефакторинг (код сапасы, тестілеу).",
    "km06": "Қазір КМ06: микроконтроллер (Arduino, C/C++).",
    "km07": "Қазір КМ07: мобильді қосымшалар (Flutter/React Native).",
}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    module: str = "auto"

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        return {"error": "API key not configured"}

    module_ctx = MODULE_CONTEXTS.get(req.module, "")
    system = SYSTEM_PROMPT + ("\n\n" + module_ctx if module_ctx else "")

    # Gemini форматына аудару
    contents = []
    for m in req.messages:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1024}
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=30)
        data = response.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"response": text}
    except Exception:
        return {"error": str(data)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
