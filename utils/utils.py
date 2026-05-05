import base64
from textual.app import App
from textual.theme import Theme
from abc import ABC
from dataclasses import dataclass, field
from openai import AsyncOpenAI


# Encodes image to base64
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@dataclass
class AgentContext:
    """Dataclass that holds all useful information about the current state."""

    # LLM-RELATED CONFIG
    model_name: str
    thinking_effort: str
    system_prompt: str
    tools: list[dict]
    client: AsyncOpenAI

    messages: list[dict]
    debug_logs: str = ""
    active_tool_calls: list = field(default_factory=list)
    debug_mode: bool = False

    metadata: dict = field(default_factory=dict)
    token_metrics: dict = field(default_factory=lambda: {
        'latest_up': 0,
        'latest_down': 0,
        'latest_cost': 0,
    })

    def append_message(self, role: str, message) -> None:
        self.append_raw({"role": role, "content": message})

    def append_raw(self, msg_dict: dict) -> None:
        self.messages.append(msg_dict)

    def update_token_metrics(self, up, down, cost) -> None:
        self.token_metrics["latest_up"] = up
        self.token_metrics["latest_down"] = down
        self.token_metrics["latest_cost"] = cost


class AgentPlugin(ABC):
    name: str = ""

    async def on_start(self, context: AgentContext) -> None: pass

    async def on_turn_start(self, context: AgentContext) -> None: pass

    async def on_stream_chunk(self, chunk: str,
                              context: AgentContext) -> None: pass

    async def on_turn_end(self, context: AgentContext) -> None: pass

    def get_system_prompt(self) -> str | None: return None

    def get_tools(self) -> list: return []

SARI_THEME = Theme(
        name="sari",
        primary="#85A598",
        secondary="#A89A85",
        warning="#fe8019",
        error="#fb4934",
        success="#b8bb26",
        accent="#fabd2f",
        foreground="#a3a3a3",
        background="#000000",
        surface="#3c3836",
        panel="#504945",
        dark=True,
        variables={
            "block-cursor-foreground": "#fbf1c7",
            "input-selection-background": "#000000",
            "button-color-foreground": "#282828",
        },
    )