import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.models.models import (
    ChatRequest,
    ChatResponse,
)
from app.triage_service.triage_agent import (
    SKTriageAgent,
)
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

agent = SKTriageAgent()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    missing = [k for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT") if not os.getenv(k)]
    if missing:
        print(f"[agent-triage-service] Missing required env vars: {', '.join(missing)}")
    yield


app = FastAPI(title="agent-triage-service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agents/triage", response_model=ChatResponse)
async def triage_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        return await agent.invoke(request)
    except Exception as e:
        # Return a ChatResponse with error details to satisfy the return type
        return ChatResponse(
            answer=f"Internal server error: {str(e)}",
            conversation_id="test",
            message_id="test",
        )
