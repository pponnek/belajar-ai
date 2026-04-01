from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

app = FastAPI()


model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature": 0.7,
        "max_output_tokens": 300
    }
)
chat_session = model.start_chat(history=[])

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    response = chat_session.send_message(req.message)

    return {
        "reply": response.text
    }