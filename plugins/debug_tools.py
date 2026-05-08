import asyncio
from utils.utils import AgentPlugin, AgentContext, ToolDefinition

class DebugTools(AgentPlugin):
    PLUGIN_NAME = "debug_tools"
    AGENT_TOOLS = [
        ToolDefinition(
            name="DUMMY_MOVE_FWD",
            description="Walk forward x units in Sari Sandbox.",
            input_arguments={
                "meters": {
                    "type": "integer",
                    "description": "The amount of units the embodied agent will move forward in the Sandbox.",
                },
            },
            required_arguments=["meters"],
        )
    ]
    SLASH_COMMANDS = [
        "print_raw_messages"
    ]
    SYSTEM_PROMPT = """
    Please warn the user that debug tools are loaded, as these tools are
    made for internal development rather than Sari Agent usage.
    """

    # Tool call handler be the same name as the tool, but all lowercase
    async def dummy_move_fwd(self, args: dict) -> dict:
        await asyncio.sleep(3)
        self.ctx.log("Called move_fwd()")
        return {
            "status": f"move {args['meters']}m success",
        }

    async def print_raw_messages(self, args: list[str]) -> str:
        self.ctx.log(str(self.ctx.messages))
        return "Successfully logged."

def setup(context) -> AgentPlugin:
    return DebugTools(context)