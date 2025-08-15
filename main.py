from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from app.models.models import ChatRequest, ChatResponse
from app.triage_service.triage_agent import SKTriageAgent
from app.utils.azure_config import app_settings

agent = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    service = AzureChatCompletion(
        api_key=app_settings.azure_openai.key,
        endpoint=app_settings.azure_openai.endpoint,
        deployment_name=app_settings.azure_openai.deployment,
    )
    app.state.agent = SKTriageAgent(service)
    yield


app = FastAPI(title="agent-triage-service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agents/triage")
async def triage_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        return await app.state.agent.invoke(request)
    except Exception as e:
        return ChatResponse(
            answer=f"Internal server error: {str(e)}",
            conversation_id="test",
            message_id="test",
        )
