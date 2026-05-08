import base64
from textual.theme import Theme
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from collections.abc import Callable
import importlib.util
import pathlib
from textual.widgets import RichLog

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sari_tui import SariApp

# Encodes image to base64
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class AgentContext:
    """
    Dataclass that holds all useful information about the current state of
    the entire program.
    """

    def __init__(self, model_name: str, thinking_effort: str, client: AsyncOpenAI, main_app: SariApp, base_system_prompt: str, plugin_dir = None, metadata: dict = {}) -> None:
        # LLM-RELATED CONFIG
        self.model_name = model_name
        self.thinking_effort = thinking_effort
        # AsyncOpenAI Client
        self.client = client
        # Textual SariApp
        self.main_app = main_app
        # Loaded plugins
        self.base_system_prompt = base_system_prompt
        self.metadata: dict = metadata

        self.plugins: list[AgentPlugin] = []
        self.messages: list[dict] = []
        self.tools: list[dict] = []
        self.active_tool_calls: list = []
        self.debug_mode: bool = False

        # PLUGIN-RELATED
        self.system_prompt: str = ""
        self.tui_command_handlers: dict[str, Callable] = {}

        self.__widget_update_handlers: list[Callable] = []
        self.__tool_call_handlers: dict = {}

        if plugin_dir:
            self.load_plugins(plugin_dir)
            self.reload_system_prompt()
            self.reload_tools()

    def inherit_plugins(self, context: AgentContext) -> None:
        self.plugins = context.plugins
        self.reload_system_prompt()
        self.reload_tools()

    def reload_system_prompt(self) -> None:
        self.system_prompt = self.base_system_prompt
        for plugin in self.plugins:
            self.system_prompt += "\n" + plugin.SYSTEM_PROMPT

    def reload_tools(self) -> None:
        for plugin in self.plugins:
            tool_dict = plugin.assemble_tool_dicts()
            self.tools += tool_dict
            # Load tool call handlers
            for tool in tool_dict:
                name = tool["name"]
                try:
                    # TODO: Check for conflicting tool names
                    self.__tool_call_handlers[name] = getattr(plugin, name.lower())
                except AttributeError:
                    raise AttributeError(
                        "Did you setup a function to handle each tool? "
                        "Is the tool handler's function name the same as the tool but lowercase?"
                    )

    async def dispatch_tool(self, tool_name: str, args: dict) -> dict:
        return await self.__tool_call_handlers[tool_name](args)

    def append_message(self, role: str, message) -> None:
        self.append_msg_raw({"role": role, "content": message})

    def append_msg_raw(self, msg_dict: dict) -> None:
        self.messages.append(msg_dict)

    def register_update_handler(self, handler: Callable) -> None:
        self.__widget_update_handlers.append(handler)

    def log(self, msg: str) -> None:
        self.main_app.query_one("#debug_log", RichLog).write(msg)

    def update_widgets(self) -> None:
        for handler_func in self.__widget_update_handlers:
            handler_func()

    def load_plugins(self, directory_name: str) -> None:
        # 1. Define the path to the folder
        folder = pathlib.Path(directory_name)

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
                self.plugins.append(setup_func(self))



class AgentPlugin(ABC):
    @property
    @abstractmethod
    def PLUGIN_NAME(self) -> str:
        """Name of the plugin"""
        pass

    @property
    @abstractmethod
    def AGENT_TOOLS(self) -> list[ToolDefinition]:
        """List of tools available to the agent"""
        pass

    @property
    @abstractmethod
    def SYSTEM_PROMPT(self) -> str:
        """System prompt that gets appended to the base system prompt (SARI.md)"""
        pass


    def __init__(self, context: AgentContext):
        self.ctx = context

    def assemble_tool_dicts(self) -> list[dict]:
        tools = []
        for tooldef in self.AGENT_TOOLS:
            tools.append(tooldef.to_dict())
        return tools

    def slash_command(self, root_cmd: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.ctx.tui_command_handlers[root_cmd] = func
            return wrapper
        return decorator


    async def on_start(self, context: AgentContext) -> None: pass

    async def on_turn_start(self, context: AgentContext) -> None: pass

    async def on_stream_chunk(self, chunk: str,
                              context: AgentContext) -> None: pass

    async def on_turn_end(self, context: AgentContext) -> None: pass






@dataclass
class ToolDefinition:
    name: str
    description: str
    input_arguments: dict[str, Any]
    required_arguments: list[str]
    strict: bool = True

    def to_dict(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.input_arguments,
                "required": self.required_arguments,
                "additionalProperties": False,
            },
            "strict": self.strict,
        }


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
        accent="#F1A361",
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