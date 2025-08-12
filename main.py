import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.models.models import ChatRequest, ChatResponse
from app.triage_service.triage_agent import SKTriageAgent
from app.utils.azure_config import AzureOpenAISettings

settings = AzureOpenAISettings()

agent = SKTriageAgent()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    missing = [
        k
        for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT")
        if not os.getenv(k)
    ]
    if missing:
        print(f"[agent-triage-service] Missing required env vars: {', '.join(missing)}")
    yield


app = FastAPI(title="agent-triage-service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agents/triage")
async def triage_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        return await agent.invoke(request)
    except Exception as e:
        return ChatResponse(
            answer=f"Internal server error: {str(e)}",
            conversation_id="test",
            message_id="test",
        )
