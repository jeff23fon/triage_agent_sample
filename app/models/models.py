from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlData(BaseModel):
    url: str
    detail: Optional[str] = None


class ImageUrlBlock(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrlData


ChatMessageContent = Union[TextBlock, ImageUrlBlock]


class ChatMessage(BaseModel):
    role: str
    content: str | list[ChatMessageContent]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    settings: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    message_id: str
