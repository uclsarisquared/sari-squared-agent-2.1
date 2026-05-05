import asyncio

from utils.utils import AgentPlugin, AgentContext

class DebugTools(AgentPlugin):
    PLUGIN_NAME = "Debug Tools v1.0"
    AGENT_TOOLS = [
        {
            "type": "function",
            "name": "MOVE_FWD",
            "description": "Walk forward x units in Sari Sandbox.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "meters": {
                        "type": "integer",
                        "description": "The amount of units the embodied agent will move forward in the sandbox.",
                    },
                },
                "required": ["meters"],
                "additionalProperties": False
            },
        },
    ]
    SYSTEM_PROMPT = """
    Please warn the user that debug tools are loaded, as these tools are
    made for internal development rather than Sari Agent usage.
    """

    # Tool call handler be the same name as the tool, but all lowercase
    async def move_fwd(self, args: dict) -> dict:
        await asyncio.sleep(3)
        return {
            "status": f"move {args['meters']} success",
        }

def setup() -> AgentPlugin:
    return DebugTools()