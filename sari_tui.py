from openai import AsyncOpenAI
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, HorizontalGroup
from textual.widgets import TabbedContent, TabPane, RichLog
from agent_tools3 import (NAVIGATION_TOOLS, MANIPULATION_TOOLS, PERCEPTION_TOOLS,
                           MEMORY_TOOLS, SWITCH_MODE_TOOL)
from utils.utils import SARI_THEME, AgentContext, read_markdown
from utils.tui_widgets import (
    MemoryDisplay, ModeDisplay, LLMUserInput, WelcomeHeader,
)
import os

# Configuration
DEBUG = False
# MODEL_NAME = "qwen/qwen3.5-27b"
MODEL_NAME = "deepseek/deepseek-v4-flash"
PLUGIN_DIR = "plugins"
LOAD_PLUGINS = [
    "debug_tools",
    "subagents"
]
BASE_PROMPT = read_markdown("memory/SARI.md")

ALL_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS + MEMORY_TOOLS + [SWITCH_MODE_TOOL]
# ALL_TOOLS = []

current_mode = "navigation"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ['SARI_OPENROUTER_API_KEY']
)


class SariApp(App):

    TITLE = "Sari Term"
    CSS_PATH = "sari_tui.tcss"

    def __init__(self):
        self.ctx = AgentContext(
            base_system_prompt=BASE_PROMPT,
            model_name=MODEL_NAME,
            thinking_effort="low",
            client=client,
            main_app=self,
            plugin_dir=PLUGIN_DIR,
            plugins=LOAD_PLUGINS,
            metadata={
                'current_mode': 'navigation',
            }
        )

        super().__init__()

    def compose(self) -> ComposeResult:
        # yield Header()
        with TabbedContent(initial="main_agent"):
            with TabPane("🔍", id="debug_log_pane"):
                yield RichLog(highlight=True, id="debug_log")
            with TabPane("Main Agent", id="main_agent"):
                yield VerticalScroll(id="user_llm_screen")
                with HorizontalGroup(id="plugin_debug_display"):
                    pass
                    # yield MemoryDisplay(self.ctx)
                    # yield ModeDisplay(self.ctx)
                yield LLMUserInput(self.ctx)

    def on_mount(self) -> None:
        self.sub_title = ""
        self.register_theme(SARI_THEME)
        self.theme = "sari"
        self.query_one(VerticalScroll).mount(WelcomeHeader(self.ctx))


if __name__ == "__main__":
    app = SariApp()
    app.run()
