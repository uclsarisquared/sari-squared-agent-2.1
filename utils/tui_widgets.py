from textual import on, events
from textual.app import ComposeResult
from textual.containers import VerticalGroup, HorizontalGroup, VerticalScroll
from textual.widgets import Markdown, LoadingIndicator, Collapsible, Label, Input, Static, RichLog
from textual.suggester import SuggestFromList
from agent_tools3 import load_semantic_memory
from utils.utils import AgentContext

COMMAND_LIST = [
    "/subagents enable",
    "/subagents disable",
    "/effort medium",
    "/effort low",
    "/effort high",
    "/effort none",
]

WELCOME_TEXT = """
# Let's shop.

Enter a prompt below. Use /<command> to run a command. Available commands are:
- `/subagents` enable/disable
- `/effort` high/medium/low/none
"""

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
        self.border_title = "🛠️ " + self.tool_name

    def update_display_header(self, args: str) -> None:
        self.border_title += args

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()
        yield Label("OK")

    def tool_done(self, resp: str):
        self.add_class("tool_success")
        self.query_one(Label).content = resp
        self.query_one(LoadingIndicator).display = False


class UserPrompt(HorizontalGroup):

    BORDER_TITLE = "User Prompt"

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(content=" ❯", id="input_arrow")
        yield Markdown(markdown=self.prompt, id="user_prompt")


class MemoryDisplay(HorizontalGroup):
    """Shows a live count of stored semantic memory facts."""

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        super().__init__()

    def compose(self) -> ComposeResult:
        count = len(load_semantic_memory())
        yield Label(f"Semantic memory: {count} fact(s)", id="memory_display")

    def refresh_count(self) -> None:
        # TODO: should use ctx.metadata
        count = len(load_semantic_memory())
        self.query_one("#memory_display", Label).update(f"Semantic memory: {count} fact(s)")


class ModeDisplay(HorizontalGroup):

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label(f"Agent mode: {self.ctx.metadata['current_mode']}", id="mode_display")

    def update_mode(self, mode: str) -> None:
        # TODO: should use ctx.metadata
        self.query_one("#mode_display", Label).update(f"Agent mode: {mode}")
        self.app.query_one(MemoryDisplay).refresh_count()


class LoadingIcon(Label):
    SPINNER_ICONS = ["·", "✻", "✽", "✶", "✳", "✢"]
    __spinner_idx = 0
    __spinner_timer = None

    def compose(self) -> ComposeResult:
        yield Label("·", id="spinner_icon")

    def on_mount(self) -> None:
        self.__spinner_timer = self.set_interval(
            0.5,
            self.update_spinner,
            name="spinner"
        )

    def stop_spinner(self):
        self.__spinner_timer.pause()
        label = self.query_one(Label)
        label.content = "⏺"
        label.add_class("spinner_done")


    def update_spinner(self) -> None:
        self.query_one(Label).content = self.SPINNER_ICONS[
            self.__spinner_idx
        ]
        self.__spinner_idx = (self.__spinner_idx + 1) % len(self.SPINNER_ICONS)


class SlashCommandDisplay(HorizontalGroup):

    def __init__(self, command, cmd_output) -> None:
        self.command = command
        self.cmd_output = cmd_output
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup():
            with HorizontalGroup(id="slash_command_display"):
                yield Static("❯", id="slash_command_arrow")
                yield Label(self.command, id="slash_cmd_text")
            with HorizontalGroup(id="slash_cmd_output"):
                yield Static("⎿", id="slash_cmd_dd")
                yield Label(self.cmd_output, id="slash_cmd_out")



class LLMUserInput(HorizontalGroup):

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        super().__init__()

    @on(Input.Submitted)
    def on_button_pressed(self) -> None:
        from .llm_streaming import LLMResponse
        user_input = self.query_one(Input).value

        if not user_input:
            return

        if user_input.startswith("/"):
            self.query_one(Input).value = ""
            self.ctx.main_app.query_one(
                "#user_llm_screen",
                VerticalScroll
            ).mount(SlashCommandDisplay("/test", "hello world"))
            return
            # parts = user_input[1:].split(" ")
            # root_cmd = parts[0]
            # if root_cmd in self.ctx.tui_command_handlers:
            #     handler, _ = self.ctx.tui_command_handlers[root_cmd]
            #     handler(parts[1:])
            #     return

        # self.parent is the main app
        vscroll = self.parent.query_one("#user_llm_screen", VerticalScroll)
        vscroll.mount(UserPrompt(user_input))
        self.parent.sub_title = user_input
        self.query_one(Input).value = ""
        vscroll.mount(
            LLMResponse(
                user_input, self.ctx
            )
        )

        # Force scrolls down
        vscroll.scroll_end()

    def compose(self) -> ComposeResult:
        yield Static(content="❯", id="input_arrow")
        yield Input(
            placeholder=f"Send prompt to {self.ctx.model_name}...",
            suggester=SuggestFromList(self.ctx.get_possible_commands()),
            compact=True,
            id="input_text"
        )


class WelcomeHeader(VerticalGroup):

    BORDER_TITLE = "Sari Term v1.0"

    def compose(self) -> ComposeResult:
        yield Markdown(markdown=WELCOME_TEXT, id="welcome_header")
