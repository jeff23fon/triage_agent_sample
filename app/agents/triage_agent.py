import logging
from uuid import uuid4

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole

from app.models.models import ChatRequest, ChatResponse, TextBlock


class SKTriageAgent:
    """Semantic Kernel triage agent (no storage; service-only)."""

    def __init__(self, service: AzureChatCompletion) -> None:
        self._kernel = Kernel()
        self._kernel.add_service(service)
        self._agents_cache: dict[str, ChatCompletionAgent] = {}

    def _create_agent(
        self,
        name: str,
        instructions: str,
        plugins: list[object] | dict[str, object] | None = None,
        function_choice_behavior: FunctionChoiceBehavior | None = None,
        kernel: Kernel | None = None,
    ) -> ChatCompletionAgent:
        if function_choice_behavior is None:
            function_choice_behavior = FunctionChoiceBehavior.Auto()  # type: ignore
        return ChatCompletionAgent(
            name=name,
            instructions=instructions,
            plugins=plugins if plugins is not None else None,
            function_choice_behavior=function_choice_behavior,
            kernel=kernel,
        )

    def _get_triage_agent(self) -> ChatCompletionAgent:
        if "triage" in self._agents_cache:
            return self._agents_cache["triage"]

        billing_agent = self._create_agent(
            name="BillingAgent",
            instructions="""You handle billing issues like charges, payment methods, cycles, fees,
            discrepancies, and payment failures.""",
            kernel=self._kernel,
        )
        refund_agent = self._create_agent(
            name="RefundAgent",
            instructions="""Assist users with refund inquiries, including eligibility, policies,
            processing, and status updates.""",
            kernel=self._kernel,
        )
        triage_agent = self._create_agent(
            name="TriageAgent",
            instructions="""Evaluate user requests and forward them to BillingAgent or RefundAgent 
            for targeted assistance. Provide the full answer to the user containing any information 
            from the agents.""",
            plugins=[billing_agent, refund_agent],
            kernel=self._kernel,
        )
        self._agents_cache["triage"] = triage_agent
        return triage_agent

    def _chat_request_to_sk_history(self, request: ChatRequest) -> ChatHistory:
        sk_messages: list[ChatMessageContent] = []
        for m in request.messages:
            role = AuthorRole.USER if m.role == "user" else AuthorRole.ASSISTANT
            if isinstance(m.content, str):
                sk_messages.append(ChatMessageContent(role=role, content=m.content))
            else:
                items: list[TextContent | ImageContent] = []
                for item in m.content:
                    if isinstance(item, TextBlock):
                        items.append(TextContent(text=item.text))
                    else:
                        items.append(ImageContent(url=item.image_url.url))
                if items:
                    sk_messages.append(ChatMessageContent(role=role, items=items))  # type: ignore
        return ChatHistory(messages=sk_messages)

    async def invoke(self, request: ChatRequest) -> ChatResponse:
        triage_agent: ChatCompletionAgent = self._get_triage_agent()
        chat_history: ChatHistory = self._chat_request_to_sk_history(request)
        conversation_id = request.conversation_id or str(uuid4())
        message_id = str(uuid4())
        try:
            result: ChatMessageContent = await triage_agent.get_response(chat_history)  # type: ignore
        except Exception as e:
            logging.error(f"Error occurred while invoking triage agent: {e}")
            result = ChatMessageContent(
                role=AuthorRole.SYSTEM, content="An error occurred while processing your request."
            )
        answer = str(result.content) if result else "No response from agent"  # type: ignore
        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            message_id=message_id,
        )
