from textual import work, events
from openai import AsyncOpenAI, APIConnectionError
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup, VerticalScroll, HorizontalGroup
from textual.widgets import Markdown, LoadingIndicator, TextArea, Header, \
    RichLog, Button, Collapsible, Label
from agent_tools2 import NAVIGATION_TOOLS, MANIPULATION_TOOLS, PERCEPTION_TOOLS, SWITCH_MODE_TOOL, TOOL_MODE_MAP, dispatch_tool
import json
import os

# Configuration
DEBUG = False
MODEL_NAME = "qwen/qwen3.5-27b"

ALL_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS + [SWITCH_MODE_TOOL]

current_mode = "navigation"

# Make sure to set OPENAI_API_KEY environment variable
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ['SARI_OPENROUTER_API_KEY']
    #base_url=os.environ['UCL_MODEL_BASE_URL']+":8000/v1",
    #api_key="key" # use for model server without auth
)

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

    def __init__(self, prompt: str | None, mode: str = "navigation") -> None:
        self.prompt = prompt
        self.mode = mode
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
        global current_mode

        tool_call_string = ""
        tool_call_id = None
        tool_call_name = None
        tool_call_display = None

        # tool-continuation calls pass prompt=None; skip adding a user message
        if self.prompt is not None:
            append_to_chat_log("user", self.prompt)

        # noinspection PyTypeChecker
        stream = await client.responses.create(
            model=MODEL_NAME,
            input=chat_log,
            max_output_tokens=65536,
            reasoning={
                "effort": "low",
                "summary": "auto"
            },
            tools=ALL_TOOLS,
            stream=True,
            parallel_tool_calls=False  # This disables parallel calling
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
                    if event.item.type == "message" and event.item.content:
                        # Text response — add to chat log
                        append_to_chat_log("assistant", event.item.content[0].text)

                case "response.reasoning_summary_part.done":
                    # LLM finished reasoning, hide `Thinking` collapsible
                    self.query_one(LLMThinkingSummary).done_thinking()

                case "response.output_item.added":
                    if event.item.type == "function_call":
                        tool_call_id = event.item.call_id
                        tool_call_name = event.item.name
                        tool_call_display = LLMToolCallDisplay(event.item.name)

                        await self.mount(
                            tool_call_display
                        )

                case "response.function_call_arguments.delta":
                    tool_call_string += event.delta

                case "response.function_call_arguments.done":
                    args = json.loads(tool_call_string)
                    tool_call_display.append_func_args(tool_call_string)
                    tool_call_string = ""

                    if tool_call_display.tool_name == "switch_mode":
                        new_mode = args["mode"]
                        current_mode = new_mode
                        self.mode = new_mode
                        self.app.query_one(ModeDisplay).update_mode(new_mode)
                        result = {"switched_to": new_mode}
                    elif (required_mode := TOOL_MODE_MAP.get(tool_call_display.tool_name)) and required_mode != self.mode:
                        result = {"error": f"Mode mismatch: '{tool_call_display.tool_name}' requires '{required_mode}' mode. Call switch_mode(\"{required_mode}\") first."}
                    else:
                        try:
                            result = await dispatch_tool(tool_call_display.tool_name, args)
                        except Exception as e:
                            result = {"error": str(e)}
                    tool_call_display.tool_done()

                    # Append function_call and function_call_output together after
                    # dispatch_tool returns so chat_log is never left with an
                    # unmatched function_call during the await.
                    chat_log.append({
                        "type": "function_call",
                        "call_id": tool_call_id,
                        "name": tool_call_name,
                        "arguments": json.dumps(args),
                    })

                    if isinstance(result, dict) and "image_base64" in result:
                        # Image result: add function_call_output with a placeholder,
                        # then pass the actual image as an input_image content block
                        # so it doesn't bloat the context as a plain text string.
                        chat_log.append({
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": "Screenshot captured.",
                        })
                        chat_log.append({
                            "role": "user",
                            "content": [{
                                "type": "input_image",
                                "image_url": f"data:{result['mimeType']};base64,{result['image_base64']}",
                            }],
                        })
                    else:
                        chat_log.append({
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": json.dumps(result, default=list),
                        })

                    # Continue the conversation; prompt=None skips adding a user message
                    await self.parent.mount(LLMResponse(None, self.mode))



class UserPrompt(VerticalGroup):

    BORDER_TITLE = "User Prompt"

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Markdown(markdown=self.prompt, classes="user_prompt")


class ModeDisplay(HorizontalGroup):

    def compose(self) -> ComposeResult:
        yield Label(f"Agent mode: {current_mode}", id="mode_display")

    def update_mode(self, mode: str) -> None:
        self.query_one("#mode_display", Label).update(f"Agent mode: {mode}")


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
                LLMResponse(user_input, current_mode)
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
        yield ModeDisplay()
        yield LLMInput()

    def on_mount(self) -> None:
        self.sub_title = ""
        self.theme = "gruvbox"

if __name__ == "__main__":
    app = SariApp()
    app.run()
