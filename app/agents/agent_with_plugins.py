from typing import Union

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistorySummarizationReducer, ChatHistoryTruncationReducer
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import kernel_function

from app.models.models import ChatRequest, ChatResponse

# Control which reducer to use: "truncation" or "summarization"
CHAT_HISTORY_REDUCER_TYPE = "summarization"  # or "truncation"


class SamplePlugin:
    @kernel_function(description="Returns a greeting.")
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"
    
    @kernel_function(description="Returns the menu items and their prices.")
    def get_menu(self) -> str:
        menu = {
            "Burger": 8.99,
            "Pizza": 12.50,
            "Salad": 7.25,
            "Soda": 2.50,
            "Coffee": 3.00,
        }
        return "\n".join([f"{item}: ${price:.2f}" for item, price in menu.items()])


class SKSampleAgent:
    def __init__(self, service: AzureChatCompletion):
        self._kernel = Kernel()
        self._kernel.add_service(service)
        self._kernel.add_plugin(SamplePlugin(), plugin_name="sample")
        self._agent = ChatCompletionAgent(
            name="SampleAgent",
            instructions="Use the sample plugin to greet users.",
            plugins=["sample"],
            kernel=self._kernel,
        )
        # Choose reducer based on config
        self._history: ChatHistorySummarizationReducer | ChatHistoryTruncationReducer
        if CHAT_HISTORY_REDUCER_TYPE == "summarization":
            self._history = ChatHistorySummarizationReducer(
                service=service,
                target_count=10,
                threshold_count=5,
                auto_reduce=True,
            )
        else:
            self._history = ChatHistoryTruncationReducer(
                target_count=10,
                threshold_count=5,
                auto_reduce=True,
            )

    async def invoke(self, request: ChatRequest) -> ChatResponse:
        self._history.messages.clear()
        for m in request.messages:
            role = {
                "user": AuthorRole.USER,
                "assistant": AuthorRole.ASSISTANT,
                "system": AuthorRole.SYSTEM,
            }.get(m.role, AuthorRole.USER)
            if isinstance(m.content, str):
                msg = ChatMessageContent(role=role, content=m.content)
            else:
                items: list[Union[TextContent, ImageContent]] = []
                for item in m.content:
                    if item.type == "text":
                        items.append(TextContent(text=item.text))
                    elif item.type == "image_url":
                        items.append(ImageContent(uri=item.image_url.url))
                msg = ChatMessageContent(role=role, items=items)  # type: ignore
            await self._history.add_message_async(msg)

        result: ChatMessageContent = await self._agent.get_response(self._history)  # type: ignore
        return ChatResponse(
            answer=str(result.content),  # type: ignore
            conversation_id=request.conversation_id or "",
            message_id="sample-id",
        )
