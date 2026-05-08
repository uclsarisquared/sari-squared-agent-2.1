from __future__ import annotations

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


class PluginManager:
    """Manages plugin loading, activation, tool dispatch, and system prompt assembly."""

    def __init__(self, base_system_prompt: str, plugin_dir: str | None = None, plugins: list[str] | None = None, context: AgentContext | None = None) -> None:
        self.base_system_prompt = base_system_prompt
        self.loaded_plugins: list[AgentPlugin] = []
        self.unloaded_plugins: list[AgentPlugin] = []
        self.system_prompt: str = base_system_prompt
        self.tools: list[dict] = []
        self.tui_command_handlers: dict[str, tuple[Callable, dict]] = {}
        self.__tool_call_handlers: dict = {}

        if plugin_dir and context is not None:
            self._load_plugins(plugin_dir, plugins, context)
            self.reload_system_prompt()
            self.reload_tools()

    def _load_plugins(self, directory_name: str, plugins: list[str] | None, context: AgentContext) -> None:
        folder = pathlib.Path(directory_name)
        paths = (
            [folder / f"{name}.py" for name in plugins]
            if plugins is not None
            else [p for p in folder.glob("*.py") if p.name != "__init__.py"]
        )
        for path in paths:
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'setup'):
                self.loaded_plugins.append(module.setup(context))

    def reload_system_prompt(self) -> None:
        self.system_prompt = self.base_system_prompt
        for plugin in self.loaded_plugins:
            self.system_prompt += "\n" + plugin.SYSTEM_PROMPT

    def reload_tools(self) -> None:
        self.tools = []
        self.__tool_call_handlers = {}
        for plugin in self.loaded_plugins:
            tool_dict = plugin.assemble_tool_dicts()
            self.tools += tool_dict
            for tool in tool_dict:
                name = tool["name"]
                try:
                    self.__tool_call_handlers[name] = getattr(plugin, name.lower())
                except AttributeError:
                    raise AttributeError(
                        "Did you setup a function to handle each tool? "
                        "Is the tool handler's function name the same as the tool but lowercase?"
                    )

    async def dispatch_tool(self, tool_name: str, args: dict) -> dict:
        return await self.__tool_call_handlers[tool_name](args)

    def get_possible_commands(self) -> list[str]:
        result = []
        for root_cmd, (_, sub_cmds) in self.tui_command_handlers.items():
            self._flatten_commands(f"/{root_cmd}", sub_cmds, result)
        return result

    def _flatten_commands(self, prefix: str, sub_cmds: dict, result: list) -> None:
        if not sub_cmds:
            result.append(prefix)
            return
        for key, nested in sub_cmds.items():
            self._flatten_commands(f"{prefix} {key}", nested, result)

    def rebind_context(self, context: AgentContext) -> None:
        """Update every plugin's ctx reference."""
        for plugin in self.loaded_plugins + self.unloaded_plugins:
            plugin.ctx = context

    def activate_plugin(self, plugin_name: str) -> None:
        """Move a plugin from unloaded → loaded and rebuild prompt/tools."""
        for i, plugin in enumerate(self.unloaded_plugins):
            if plugin.PLUGIN_NAME == plugin_name:
                self.loaded_plugins.append(self.unloaded_plugins.pop(i))
                self.reload_system_prompt()
                self.reload_tools()
                return
        raise ValueError(f"Plugin '{plugin_name}' not found in unloaded plugins.")

    def deactivate_plugin(self, plugin_name: str) -> None:
        """Move a plugin from loaded → unloaded and rebuild prompt/tools."""
        for i, plugin in enumerate(self.loaded_plugins):
            if plugin.PLUGIN_NAME == plugin_name:
                self.unloaded_plugins.append(self.loaded_plugins.pop(i))
                self.reload_system_prompt()
                self.reload_tools()
                return
        raise ValueError(f"Plugin '{plugin_name}' not found in loaded plugins.")


class AgentContext:
    """Holds all state for the running agent: LLM config, messages, and plugin manager."""

    def __init__(self, model_name: str, thinking_effort: str, client: AsyncOpenAI, main_app: SariApp, base_system_prompt: str, plugin_dir=None, plugins: list[str] | None = None, metadata: dict = {}) -> None:
        self.model_name = model_name
        self.thinking_effort = thinking_effort
        self.client = client
        self.main_app = main_app
        self.base_system_prompt = base_system_prompt
        self.metadata: dict = metadata

        self.plugin_manager: PluginManager | None = None
        self.messages: list[dict] = []
        self.active_tool_calls: list = []
        self.debug_mode: bool = False

        self.__widget_update_handlers: list[Callable] = []

        if plugin_dir:
            self.plugin_manager = PluginManager(base_system_prompt, plugin_dir, plugins, self)

    # --- Proxy properties so existing call-sites keep working ---

    @property
    def plugins(self) -> list:
        return self.plugin_manager.loaded_plugins if self.plugin_manager else []

    @property
    def system_prompt(self) -> str:
        return self.plugin_manager.system_prompt if self.plugin_manager else self.base_system_prompt

    @property
    def tools(self) -> list[dict]:
        return self.plugin_manager.tools if self.plugin_manager else []

    @property
    def tui_command_handlers(self) -> dict:
        return self.plugin_manager.tui_command_handlers if self.plugin_manager else {}

    # --- Plugin management ---

    def inherit_plugins(self, context: AgentContext, only: list[str] | None = None) -> None:
        self.plugin_manager = PluginManager(self.base_system_prompt)
        all_plugins = context.plugin_manager.loaded_plugins + context.plugin_manager.unloaded_plugins
        for plugin in all_plugins:
            if only is None or plugin.PLUGIN_NAME in only:
                self.plugin_manager.loaded_plugins.append(plugin)
            else:
                self.plugin_manager.unloaded_plugins.append(plugin)
        self.plugin_manager.reload_system_prompt()
        self.plugin_manager.reload_tools()

    async def dispatch_tool(self, tool_name: str, args: dict) -> dict:
        return await self.plugin_manager.dispatch_tool(tool_name, args)

    def get_possible_commands(self) -> list[str]:
        return self.plugin_manager.get_possible_commands() if self.plugin_manager else []

    # --- Messaging ---

    def append_message(self, role: str, message) -> None:
        self.append_msg_raw({"role": role, "content": message})

    def append_msg_raw(self, msg_dict: dict) -> None:
        self.messages.append(msg_dict)

    # --- Widget / UI helpers ---

    def register_update_handler(self, handler: Callable) -> None:
        self.__widget_update_handlers.append(handler)

    def log(self, msg: str) -> None:
        # if self.main_app._is_mounted:
        self.main_app.query_one("#debug_log", RichLog).write(msg)

    def update_widgets(self) -> None:
        for handler_func in self.__widget_update_handlers:
            handler_func()



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

    def slash_command(self, root_cmd: str, sub_cmds: dict = None):
        def decorator(func):
            self.ctx.tui_command_handlers[root_cmd] = (func, sub_cmds or {})
            return func
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