import os
from dataclasses import dataclass
from typing import Sequence, cast
from uuid import uuid4

from app.models.models import (
    ChatMessageContent,
    ChatRequest,
    ChatResponse,
)
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory  # Add this import
from semantic_kernel.contents.chat_message_content import (
    ChatMessageContent as SKChatMessageContent,
)
from semantic_kernel.contents.utils.author_role import AuthorRole as SKAuthorRole


@dataclass
class _AgentMessageAdapter:
    role: str
    content: str | list[SKChatMessageContent]


def _adapt_content(c: ChatMessageContent) -> str | list[SKChatMessageContent]:
    if isinstance(c, str):
        return c
    return cast(list[SKChatMessageContent], c)


def _role(role: str) -> SKAuthorRole:
    return SKAuthorRole.USER if role == "user" else SKAuthorRole.ASSISTANT


def to_sk_messages(messages: Sequence[_AgentMessageAdapter]) -> list[SKChatMessageContent]:
    out: list[SKChatMessageContent] = []
    for m in messages:
        if isinstance(m.content, str):
            out.append(SKChatMessageContent(role=_role(m.role), content=m.content))
        else:
            # Already SKChatMessageContent list (e.g., tool messages)
            out.extend(m.content)
    return out


class SKTriageAgent:
    """Semantic Kernel triage agent (no storage; service-only)."""

    def __init__(self) -> None:
        self._agents_cache: dict[str, ChatCompletionAgent] = {}

    def _get_triage_agent(self) -> ChatCompletionAgent:
        if "triage" in self._agents_cache:
            return self._agents_cache["triage"]

        # Read env directly in the service (no dependency on app_settings)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not endpoint or not api_key or not deployment:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT/KEY/DEPLOYMENT for triage service.")

        billing_agent = ChatCompletionAgent(
            service=AzureChatCompletion(api_key=api_key, endpoint=endpoint, deployment_name=deployment),
            name="BillingAgent",
            instructions="You handle billing issues like charges, payment methods, cycles, fees, discrepancies, and payment failures.",
            function_choice_behavior=FunctionChoiceBehavior.Auto(),  # type: ignore
        )
        refund_agent = ChatCompletionAgent(
            service=AzureChatCompletion(api_key=api_key, endpoint=endpoint, deployment_name=deployment),
            name="RefundAgent",
            instructions="Assist users with refund inquiries, including eligibility, policies, processing, and status updates.",
            function_choice_behavior=FunctionChoiceBehavior.Auto(),  # type: ignore
        )
        triage_agent = ChatCompletionAgent(
            service=AzureChatCompletion(api_key=api_key, endpoint=endpoint, deployment_name=deployment),
            name="TriageAgent",
            instructions="Evaluate user requests and forward them to BillingAgent or RefundAgent for targeted assistance. Provide the full answer to the user containing any information from the agents.",
            plugins=[billing_agent, refund_agent],
            function_choice_behavior=FunctionChoiceBehavior.Auto(),  # type: ignore
        )
        self._agents_cache["triage"] = triage_agent
        return triage_agent

    async def invoke(self, request: ChatRequest) -> ChatResponse:
        triage_agent = self._get_triage_agent()
        adapted: Sequence[_AgentMessageAdapter] = [
            _AgentMessageAdapter(role=m.role, content=_adapt_content(m.content)) for m in request.messages
        ]
        chat_message_list = to_sk_messages(adapted)
        chat_history = ChatHistory(messages=chat_message_list)  # Wrap in ChatHistory 
        result = await triage_agent.get_response(chat_history)
        conversation_id = request.conversation_id or str(uuid4())
        message_id = str(uuid4())
        answer = str(result.content) if result else "No response from agent"
        return ChatResponse(answer=answer, conversation_id=conversation_id, message_id=message_id)
