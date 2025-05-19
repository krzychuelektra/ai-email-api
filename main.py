from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

# API key from environment
API_KEY = os.environ.get("XAI_API_KEY")

# xAI client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.x.ai/v1"
)

class EmailRequest(BaseModel):
    industry: str
    offer: str
    tone: str

@app.post("/generate")
async def generate_email(data: EmailRequest):
    prompt = f"""
    Write a cold email for a freelancer in the {data.industry} industry.
    The offer is: {data.offer}
    Use a {data.tone.lower()} tone.
    Structure: short intro, clear value, strong call to action.
    """

    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are Grok, a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return {"email": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
