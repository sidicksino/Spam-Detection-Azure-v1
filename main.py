import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

app = FastAPI(
    title="Spam Detector Proxy API",
    description="Secures and routes frontend client requests to Azure ML Online Endpoints"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Azure ML Online Endpoint credentials from environment variables
AZURE_ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

class TextPayload(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main frontend page."""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend HTML file templates/index.html not found.")

@app.post("/predict")
async def predict_spam(payload: TextPayload):
    """Proxies user prediction request to the underlying Azure ML Endpoint."""
    if not AZURE_ENDPOINT_URL or not AZURE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Azure ML credentials are not configured on the host server. Set AZURE_ENDPOINT_URL and AZURE_API_KEY."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }
    
    # Azure ML online endpoint expects format: {"data": ["message_text"]}
    azure_payload = {"data": [payload.message]}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                AZURE_ENDPOINT_URL,
                json=azure_payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Azure ML endpoint returned error: {response.text}"
                )
            
            result = response.json()
            predictions = result.get("predictions", [])
            if not predictions:
                raise HTTPException(status_code=502, detail="Invalid response structure returned by Azure ML endpoint.")
            
            return {"message": payload.message, "prediction": predictions[0]}

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Failed to communicate with Azure ML inference server: {str(e)}"
            )