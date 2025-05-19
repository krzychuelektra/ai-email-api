from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import cohere

app = FastAPI()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise Exception("COHERE_API_KEY not found in environment variables")

client = cohere.Client(COHERE_API_KEY)

class EmailRequest(BaseModel):
    industry: str
    offer: str
    tone: str

@app.get("/")
async def root():
    return {"message": "FastAPI Cohere email generator is running!"}

@app.post("/generate")
async def generate_email(data: EmailRequest):
    prompt = (
        f"Write a cold email for a freelancer in the {data.industry} industry.\n"
        f"The offer is: {data.offer}\n"
        f"Use a {data.tone.lower()} tone.\n"
        f"Structure the email with a short introduction, clear value proposition, and a strong call to action."
    )

    try:
        response = client.chat.completions.create(
            model="command-xlarge-nightly",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who writes cold emails."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            stop_sequences=[]
        )
        email_text = response.choices[0].message.content.strip()
        return {"email": email_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
