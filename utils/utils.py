import base64
from textual.theme import Theme
from abc import ABC
from textual.app import App
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from collections.abc import Callable
import importlib.util
import pathlib


# Encodes image to base64
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@dataclass
class AgentContext:
    """Dataclass that holds all useful information about the current state and
    handles updating the terminal."""

    # LLM-RELATED CONFIG
    model_name: str
    thinking_effort: str
    # AsyncOpenAI Client
    client: AsyncOpenAI
    # Textual SariApp
    main_app: App
    # Loaded plugins
    plugins: list[AgentPlugin]
    base_system_prompt: str

    messages: list[dict]
    debug_logs: str = ""
    tools: list[dict] = field(default_factory=list)
    active_tool_calls: list = field(default_factory=list)
    debug_mode: bool = False

    # PLUGIN-RELATED
    system_prompt: str = ""
    metadata: dict = field(default_factory=dict)

    __widget_update_handlers: list[Callable] = field(default_factory=list)
    __tool_call_handlers: dict = field(default_factory=dict)

    def __post_init__(self):
        self.reload_system_prompt()
        self.reload_tools()

    def reload_system_prompt(self) -> None:
        self.system_prompt = self.base_system_prompt
        for plugin in self.plugins:
            self.system_prompt += "\n" + plugin.SYSTEM_PROMPT

    def reload_tools(self) -> None:
        for plugin in self.plugins:
            self.tools += plugin.AGENT_TOOLS
            # Load tool call handlers
            for tool in plugin.AGENT_TOOLS:
                name = tool["name"]
                try:
                    self.__tool_call_handlers[name] = getattr(plugin, name.lower())
                except AttributeError:
                    raise AttributeError(
                        "Is the tool handler function name the same as the tool but lowercase?"
                    )

    async def dispatch_tool(self, tool_name: str, args: dict) -> dict:
        return await self.__tool_call_handlers[tool_name](args)

    def append_message(self, role: str, message) -> None:
        self.append_msg_raw({"role": role, "content": message})

    def append_msg_raw(self, msg_dict: dict) -> None:
        self.messages.append(msg_dict)

    def register_update_handler(self, handler: Callable) -> None:
        self.__widget_update_handlers.append(handler)

    def update_widgets(self) -> None:
        for handler_func in self.__widget_update_handlers:
            handler_func()


class AgentPlugin(ABC):
    PLUGIN_NAME: str = ""
    AGENT_TOOLS = []
    SYSTEM_PROMPT: str = ""

    async def on_start(self, context: AgentContext) -> None: pass

    async def on_turn_start(self, context: AgentContext) -> None: pass

    async def on_stream_chunk(self, chunk: str,
                              context: AgentContext) -> None: pass

    async def on_turn_end(self, context: AgentContext) -> None: pass


def load_plugins(directory_name: str) -> list[AgentPlugin]:
    # 1. Define the path to the folder
    folder = pathlib.Path(directory_name)

    plugins = []

    # 2. Iterate over all .py files (excluding __init__.py)
    for path in folder.glob("*.py"):
        if path.name == "__init__.py":
            continue

        # 3. Create a module name from the file name
        module_name = path.stem

        # 4. Load the module spec and the module itself
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 5. Look for the 'setup' function and call it
        if hasattr(module, 'setup'):
            setup_func = getattr(module, 'setup')
            plugins.append(setup_func())

    return plugins

def read_markdown(file_path: str) -> str:
    path = pathlib.Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    return path.read_text(encoding="utf-8")

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