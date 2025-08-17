import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from app.agents.agent_with_plugins import SKSampleAgent
from app.agents.triage_agent import SKTriageAgent
from app.models.models import ChatRequest, ChatResponse
from app.utils.azure_config import app_settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    service = AzureChatCompletion(
        api_key=app_settings.azure_openai.key,
        endpoint=app_settings.azure_openai.endpoint,
        deployment_name=app_settings.azure_openai.deployment,
    )
    app.state.agent = SKTriageAgent(service)
    app.state.sample_agent = SKSampleAgent(service)
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

@app.post("/v1/agents/sample")
async def sample_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        return await app.state.sample_agent.invoke(request)
    except Exception as e:
        return ChatResponse(
            answer=f"Internal server error: {str(e)}",
            conversation_id="test",
            message_id="test",
        )
