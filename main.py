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
    prompt = f"""
Write a cold email for a freelancer in the {data.industry} industry.
The offer is: {data.offer}
Use a {data.tone.lower()} tone.
Structure: short intro, clear value, strong call to action.
"""

    try:
        response = client.generate(
            model="xlarge",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["--"]
        )
        return {"email": response.generations[0].text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
