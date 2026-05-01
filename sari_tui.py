from textual import work
from openai import AsyncOpenAI, APIConnectionError
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup, VerticalScroll, HorizontalGroup
from textual.widgets import Markdown, LoadingIndicator, RichLog, Label, Static, TabbedContent, TabPane
from agent_tools3 import (NAVIGATION_TOOLS, MANIPULATION_TOOLS, PERCEPTION_TOOLS,
                           MEMORY_TOOLS, SWITCH_MODE_TOOL)
from utils.utils import SARI_THEME
from utils.llm_streaming import stream_from_llm_api as _stream_from_llm_api
import os
from utils.tui_widgets import (
    LLMThinkingSummary, LLMToolCallDisplay, UserPrompt,
    MemoryDisplay, ModeDisplay, LLMInput, WelcomeHeader,
)

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

class LLMResponse(VerticalGroup):

    BORDER_TITLE = MODEL_NAME

    def __init__(self, prompt: str | None, mode: str = "navigation") -> None:
        self.prompt = prompt
        self.mode = mode
        super().__init__()

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()

        llm_thinking = LLMThinkingSummary()
        llm_thinking.display = False
        yield llm_thinking

        with HorizontalGroup():
            yield Static("⏺", id="llm_resp_bullet")
            with VerticalGroup():
                yield Markdown(id="llm_response_text")
                yield Label("↑ ... Tok | ↓ ... Tok", id="token_usage_label")

        if DEBUG:
            yield RichLog(highlight=True, id="raw_log")

        try:
            self.stream_from_llm_api()
        except APIConnectionError:
            self.notify("Error connecting to LLM API.", severity="error")

    @work(exclusive=True)
    async def stream_from_llm_api(self):
        def _set_mode(mode: str):
            global current_mode
            current_mode = mode

        await _stream_from_llm_api(self, client, MODEL_NAME, chat_log, ALL_TOOLS, DEBUG, _set_mode)


class SariApp(App):

    TITLE = "Sari Term"
    CSS_PATH = "sari_tui.tcss"

    def compose(self) -> ComposeResult:
        # yield Header()
        with TabbedContent():
            with TabPane("Main Agent"):
                yield VerticalScroll(id="user_llm_screen")
                with HorizontalGroup():
                    yield MemoryDisplay()
                    yield ModeDisplay()
                yield LLMInput()

    def on_mount(self) -> None:
        self.sub_title = ""
        self.register_theme(SARI_THEME)
        self.theme = "sari"
        self.query_one(VerticalScroll).mount(WelcomeHeader())


if __name__ == "__main__":
    app = SariApp()
    app.run()
