from openai import AsyncOpenAI
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, HorizontalGroup
from textual.widgets import TabbedContent, TabPane
from agent_tools3 import (NAVIGATION_TOOLS, MANIPULATION_TOOLS, PERCEPTION_TOOLS,
                           MEMORY_TOOLS, SWITCH_MODE_TOOL)
from utils.utils import SARI_THEME, AgentContext
from utils.tui_widgets import (
    MemoryDisplay, ModeDisplay, LLMInput, WelcomeHeader,
)
import os

# Configuration
DEBUG = False
MODEL_NAME = "qwen/qwen3.5-27b"
# MODEL_NAME = "deepseek/deepseek-v4-flash"

ALL_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS + MEMORY_TOOLS + [SWITCH_MODE_TOOL]
# ALL_TOOLS = []

current_mode = "navigation"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ['SARI_OPENROUTER_API_KEY']
)

chat_log = []

class SariApp(App):

    TITLE = "Sari Term"
    CSS_PATH = "sari_tui.tcss"

    ctx = AgentContext(
        messages=[],
        system_prompt="",
        tools=ALL_TOOLS,
        model_name=MODEL_NAME,
        metadata={
            "current_mode": "navigation",
        },
        thinking_effort="low",
        client=client,
    )

    def compose(self) -> ComposeResult:
        # yield Header()
        with TabbedContent():
            with TabPane("Main Agent"):
                yield VerticalScroll(id="user_llm_screen")
                with HorizontalGroup():
                    yield MemoryDisplay(self.ctx)
                    yield ModeDisplay(self.ctx)
                yield LLMInput(self.ctx)

    def on_mount(self) -> None:
        self.sub_title = ""
        self.register_theme(SARI_THEME)
        self.theme = "sari"
        self.query_one(VerticalScroll).mount(WelcomeHeader())


if __name__ == "__main__":
    app = SariApp()
    app.run()
