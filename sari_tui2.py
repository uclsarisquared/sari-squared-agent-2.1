from textual import work, on
from openai import AsyncOpenAI, APIConnectionError
from textual.app import App, ComposeResult
from textual.suggester import SuggestFromList
from textual.containers import VerticalGroup, VerticalScroll, HorizontalGroup
from textual.widgets import Markdown, LoadingIndicator,  \
    RichLog, Collapsible, Label, Input, Static, TabbedContent, TabPane
from agent_tools3 import (NAVIGATION_TOOLS, MANIPULATION_TOOLS, PERCEPTION_TOOLS,
                           MEMORY_TOOLS, SWITCH_MODE_TOOL, load_semantic_memory)
from utils.utils import SARI_THEME
from utils.llm_streaming import stream_from_llm_api as _stream_from_llm_api
import os
from utils.tui_widgets import WELCOME_TEXT, COMMAND_LIST

# Configuration
DEBUG = False
MODEL_NAME = "qwen/qwen3.5-27b"

ALL_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS + MEMORY_TOOLS + [SWITCH_MODE_TOOL]
# ALL_TOOLS = []

current_mode = "navigation"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ['SARI_OPENROUTER_API_KEY']
)

chat_log = []

# ---------------------------------------------------------------------------
# TUI Widgets
# ---------------------------------------------------------------------------

class LLMThinkingSummary(VerticalGroup):

    def compose(self) -> ComposeResult:
        with Collapsible(title="🤔 Thinking...", id="thinking_collapsible", collapsed=False):
            yield Markdown()

    def update_thinking_text(self, delta: str) -> None:
        self.query_one(Markdown).append(delta)

    def collapse(self) -> None:
        self.query_one(Collapsible).collapsed = True

    def done_thinking(self) -> None:
        self.collapse()
        self.query_one(Collapsible).title = "💡 Done thinking"


class LLMToolCallDisplay(VerticalGroup):

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__()

    def on_mount(self):
        self.border_title = self.tool_name

    def append_func_args(self, args: str) -> None:
        self.border_title += args

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()
        yield Label("OK")
        self.add_class("tool_success")

    def tool_done(self):
        self.query_one(LoadingIndicator).display = False


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
            yield Markdown(id="llm_response_text")

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


class UserPrompt(HorizontalGroup):

    BORDER_TITLE = "User Prompt"

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Markdown(markdown="❯ " + self.prompt, id="user_prompt")


class MemoryDisplay(HorizontalGroup):
    """Shows a live count of stored semantic memory facts."""

    def compose(self) -> ComposeResult:
        count = len(load_semantic_memory())
        yield Label(f"Semantic memory: {count} fact(s)", id="memory_display")

    def refresh_count(self) -> None:
        count = len(load_semantic_memory())
        self.query_one("#memory_display", Label).update(f"Semantic memory: {count} fact(s)")


class ModeDisplay(HorizontalGroup):

    def compose(self) -> ComposeResult:
        yield Label(f"Agent mode: {current_mode}", id="mode_display")

    def update_mode(self, mode: str) -> None:
        self.query_one("#mode_display", Label).update(f"Agent mode: {mode}")
        self.app.query_one(MemoryDisplay).refresh_count()


class LLMInput(HorizontalGroup):

    @on(Input.Submitted)
    def on_button_pressed(self) -> None:
        user_input = self.query_one(Input).value

        if not user_input:
            return

        self.parent.query_one(VerticalScroll).mount(UserPrompt(user_input))
        self.parent.sub_title = user_input
        self.query_one(Input).value = ""
        self.parent.query_one(VerticalScroll).mount(LLMResponse(user_input, current_mode))

    def compose(self) -> ComposeResult:
        yield Static(content="❯",id="input_arrow")
        yield Input(
            placeholder="Enter Sari prompt here...",
            suggester=SuggestFromList(COMMAND_LIST),
            compact=True,
            id="input_text"
        )


class WelcomeHeader(VerticalGroup):

    BORDER_TITLE = "Sari Term v1.0"

    def compose(self) -> ComposeResult:
        yield Markdown(markdown=WELCOME_TEXT, id="welcome_header")

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
