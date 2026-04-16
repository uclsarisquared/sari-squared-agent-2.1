from textual import work, events
from openai import AsyncOpenAI, APIConnectionError
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup, VerticalScroll, HorizontalGroup
from textual.widgets import Markdown, LoadingIndicator, TextArea, Header, \
    RichLog, Button, Collapsible, Label
from agent_tools import *
import json

# Configuration
DEBUG = False
MODEL_NAME = "gpt-5.4-mini"

# Make sure to set OPENAI_API_KEY environment variable
client = AsyncOpenAI()

chat_log = []

def append_to_chat_log(role, message):
    global chat_log

    chat_log.append(
        {
            "role": role,
            "content": message
        }
    )

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
        # self.border_subtitle = "🛠️ TOOL CALL"

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

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()


        llm_thinking = LLMThinkingSummary()
        llm_thinking.display = False
        yield llm_thinking

        yield Markdown()

        if DEBUG:
            # with Collapsible(title="🔎 Show raw logs", id="log_collapsible"):
            yield RichLog(highlight=True, id="raw_log")

        try:
            self.stream_from_llm_api()
        except APIConnectionError:
            self.notify(
                "Error connecting to LLM API.",
                severity="error"
            )

    @work(exclusive=True)
    async def stream_from_llm_api(self):

        tool_call_string = ""
        tool_call_id = None
        tool_call_display = None

        append_to_chat_log("user", self.prompt)

        # noinspection PyTypeChecker
        stream = await client.responses.create(
            model=MODEL_NAME,
            input=chat_log,
            reasoning={
                "effort": "low",
                "summary": "auto"
            },
            tools=AGENT_TOOLS,
            stream=True,
        )

        async for event in stream:
            self.query_one(LoadingIndicator).display = False

            if DEBUG:
                # Debug mode logs the raw events from API
                self.query_one(RichLog).write(str(event))
                self.query_one(RichLog).write(event.type)

            match event.type:
                case "response.output_text.delta":
                    await self.query_one(Markdown).append(str(event.delta))

                case "response.reasoning_summary_part.added":
                    self.query_one(LLMThinkingSummary).display = True

                case "response.reasoning_summary_text.delta":
                    self.query_one(LLMThinkingSummary).update_thinking_text(event.delta)

                case "response.output_item.done":
                    # LLM response finished generating

                    # Only if it's a "text generation" event
                    if event.item.type == "message":
                        completed_text = event.item.content[0].text
                        append_to_chat_log("assistant", completed_text)


                case "response.reasoning_summary_part.done":
                    # LLM finished reasoning, hide `Thinking` collapsible
                    self.query_one(LLMThinkingSummary).done_thinking()

                case "response.output_item.added":
                    if event.item.type == "function_call":
                        tool_call_id = event.item.call_id
                        tool_call_display = LLMToolCallDisplay(event.item.name)

                        await self.mount(
                            tool_call_display
                        )

                case "response.function_call_arguments.delta":
                    tool_call_string += event.delta

                case "response.function_call_arguments.done":
                    args = json.loads(tool_call_string)
                    tool_call_display.append_func_args(tool_call_string)

                    tool_response = handle_agent_tool_call(args, tool_call_id)
                    tool_call_display.tool_done()

                    await self.parent.mount(
                        LLMResponse(
                            tool_response
                        )
                    )



class UserPrompt(VerticalGroup):

    BORDER_TITLE = "User Prompt"

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Markdown(markdown=self.prompt, classes="user_prompt")


class LLMInput(HorizontalGroup):

    def on_button_pressed(self, event: Button) -> None:
        if event.button.id == "enter_button":
            user_input = self.query_one(TextArea).text

            if not user_input:
                return

            self.parent.query_one(VerticalScroll).mount(
                UserPrompt(user_input)
            )

            self.parent.sub_title = user_input

            self.query_one(TextArea).text = ""

            self.parent.query_one(VerticalScroll).mount(
                LLMResponse(user_input)
            )

    def compose(self) -> ComposeResult:
        yield TextArea(placeholder = "Enter Sari prompt here...", id="input_text")
        yield Button(label="↵", id="enter_button")


class SariApp(App):

    TITLE = "Sari Term"
    CSS_PATH = "sari_tui.tcss"

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="user_llm_screen")
        yield LLMInput()

    def on_mount(self) -> None:
        self.sub_title = ""
        self.theme = "gruvbox"

if __name__ == "__main__":
    app = SariApp()
    app.run()
