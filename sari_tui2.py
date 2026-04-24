from textual import work
from openai import AsyncOpenAI, APIConnectionError
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup, VerticalScroll, HorizontalGroup
from textual.widgets import Markdown, LoadingIndicator, TextArea, Header, \
    RichLog, Button, Collapsible, Label
from agent_tools3 import (NAVIGATION_TOOLS, MANIPULATION_TOOLS, PERCEPTION_TOOLS,
                           MEMORY_TOOLS, SWITCH_MODE_TOOL, TOOL_MODE_MAP, dispatch_tool,
                           load_semantic_memory, load_episodic_memory, save_episodic_memory)
import json
import os
import re

# Configuration
DEBUG = False
MODEL_NAME = "qwen/qwen3.5-27b"

ALL_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS + MEMORY_TOOLS + [SWITCH_MODE_TOOL]

current_mode = "navigation"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ['SARI_OPENROUTER_API_KEY']
)

chat_log = []

# ---------------------------------------------------------------------------
# Episodic synthesis prompt — matches the Gemini version's output schema
# ---------------------------------------------------------------------------

EPISODIC_SYNTHESIS_INSTRUCTION = (
    "You are a memory synthesis module for an embodied agent. "
    "Given a recent agent-environment exchange, return ONLY a valid JSON object "
    "(no markdown fences, no extra text) with exactly these four keys:\n"
    "  dense_summary  — 1-2 sentence summary of what happened\n"
    "  surprise       — anything unexpected or that contradicted prior beliefs\n"
    "  what_worked    — actions or strategies that were effective\n"
    "  what_to_avoid  — actions or approaches that failed or should not be repeated"
)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def build_system_instruction() -> str:
    """Build the system prompt, injecting current semantic and episodic memory."""
    facts = load_semantic_memory()
    episodic = load_episodic_memory()

    parts = [
        "You are Sari, an embodied agent operating in a 3D environment. "
        "Use your tools to navigate, manipulate objects, and perceive the scene. "
        "Use memory tools to store important facts and recall them across interactions. "
        "Always switch to the correct mode before using navigation, manipulation, or perception tools."
    ]

    if facts:
        facts_str = "\n".join(f"- {f}" for f in facts)
        parts.append(f"\n## SEMANTIC MEMORY\n{facts_str}")

    if episodic:
        parts.append(
            f"\n## EPISODIC MEMORY (last session)\n"
            f"- Summary: {episodic.get('dense_summary', 'N/A')}\n"
            f"- Surprise: {episodic.get('surprise', 'N/A')}\n"
            f"- What worked: {episodic.get('what_worked', 'N/A')}\n"
            f"- What to avoid: {episodic.get('what_to_avoid', 'N/A')}"
        )

    return "\n".join(parts)


async def synthesize_episodic_memory() -> None:
    """Run a post-turn LLM call to synthesize episodic memory from the last exchange."""
    # Collect the tail of the chat log as readable text, skipping image blocks
    exchange_parts = []
    for msg in chat_log[-10:]:
        role = msg.get("role") or msg.get("type", "")
        content = msg.get("content") or msg.get("output") or msg.get("arguments", "")
        if isinstance(content, str) and content.strip():
            exchange_parts.append(f"{role}: {content.strip()}")

    if not exchange_parts:
        return

    exchange_text = "\n\n".join(exchange_parts)

    try:
        response = await client.responses.create(
            model=MODEL_NAME,
            instructions=EPISODIC_SYNTHESIS_INSTRUCTION,
            input=[{"role": "user", "content": exchange_text}],
            max_output_tokens=512,
        )

        text = ""
        for item in response.output:
            if hasattr(item, "content"):
                for part in item.content:
                    if hasattr(part, "text"):
                        text += part.text

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            entry = json.loads(match.group())
            save_episodic_memory(entry)
    except Exception:
        pass  # episodic synthesis is best-effort; never block the main flow


def append_to_chat_log(role, message):
    global chat_log
    chat_log.append({"role": role, "content": message})


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

        yield Markdown()

        if DEBUG:
            yield RichLog(highlight=True, id="raw_log")

        try:
            self.stream_from_llm_api()
        except APIConnectionError:
            self.notify("Error connecting to LLM API.", severity="error")

    @work(exclusive=True)
    async def stream_from_llm_api(self):
        global current_mode

        tool_call_string = ""
        tool_call_id = None
        tool_call_name = None
        tool_call_display = None
        spawned_continuation = False

        if self.prompt is not None:
            append_to_chat_log("user", self.prompt)

        # noinspection PyTypeChecker
        stream = await client.responses.create(
            model=MODEL_NAME,
            instructions=build_system_instruction(),
            input=chat_log,
            max_output_tokens=65536,
            reasoning={
                "effort": "low",
                "summary": "auto"
            },
            tools=ALL_TOOLS,
            stream=True,
            parallel_tool_calls=False
        )

        async for event in stream:
            self.query_one(LoadingIndicator).display = False

            if DEBUG:
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
                        append_to_chat_log("assistant", event.item.content[0].text)

                case "response.reasoning_summary_part.done":
                    self.query_one(LLMThinkingSummary).done_thinking()

                case "response.output_item.added":
                    if event.item.type == "function_call":
                        tool_call_id = event.item.call_id
                        tool_call_name = event.item.name
                        tool_call_display = LLMToolCallDisplay(event.item.name)
                        await self.mount(tool_call_display)

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

                    chat_log.append({
                        "type": "function_call",
                        "call_id": tool_call_id,
                        "name": tool_call_name,
                        "arguments": json.dumps(args),
                    })

                    if isinstance(result, dict) and "image_base64" in result:
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

                    spawned_continuation = True
                    await self.parent.mount(LLMResponse(None, self.mode))

        # Turn is complete — synthesize episodic memory in the background
        if not spawned_continuation:
            await synthesize_episodic_memory()


class UserPrompt(VerticalGroup):

    BORDER_TITLE = "User Prompt"

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Markdown(markdown=self.prompt, classes="user_prompt")


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

    def on_button_pressed(self, event: Button) -> None:
        if event.button.id == "enter_button":
            user_input = self.query_one(TextArea).text

            if not user_input:
                return

            self.parent.query_one(VerticalScroll).mount(UserPrompt(user_input))
            self.parent.sub_title = user_input
            self.query_one(TextArea).text = ""
            self.parent.query_one(VerticalScroll).mount(LLMResponse(user_input, current_mode))

    def compose(self) -> ComposeResult:
        yield TextArea(placeholder="Enter Sari prompt here...", id="input_text")
        yield Button(label="↵", id="enter_button")


class SariApp(App):

    TITLE = "Sari Term"
    CSS_PATH = "sari_tui.tcss"

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="user_llm_screen")
        yield MemoryDisplay()
        yield ModeDisplay()
        yield LLMInput()

    def on_mount(self) -> None:
        self.sub_title = ""
        self.theme = "gruvbox"


if __name__ == "__main__":
    app = SariApp()
    app.run()
