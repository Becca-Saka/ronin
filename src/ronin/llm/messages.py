import base64
from dataclasses import dataclass, field


@dataclass
class TextMessage:
    content: str
    role: str = "user"

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class AssistantMessage(TextMessage):
    role: str = "assistant"


@dataclass
class SystemMessage(TextMessage):
    role: str = "system"


@dataclass
class ImageMessage:
    text: str
    png_bytes: bytes
    role: str = "user"

    def to_dict(self) -> dict:
        encoded = base64.b64encode(self.png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{encoded}"
        return {
            "role": self.role,
            "content": [
                {"type": "text", "text": self.text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }


@dataclass
class MultiPartMessage:
    """For messages with multiple images or mixed content."""

    role: str = "user"
    parts: list[dict] = field(default_factory=list)

    def add_text(self, text: str) -> "MultiPartMessage":
        self.parts.append({"type": "text", "text": text})
        return self

    def add_image(self, png_bytes: bytes) -> "MultiPartMessage":
        encoded = base64.b64encode(png_bytes).decode("utf-8")
        self.parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
        return self

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.parts}


# Union type for type hints
Message = (
    TextMessage | AssistantMessage | SystemMessage | ImageMessage | MultiPartMessage
)
