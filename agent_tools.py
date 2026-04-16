# TODO: use `allowed_tools` to restrict agent to low-level tools
import json

MOVE_FWD_FUNCTION_NAME = "move_forward"

AGENT_TOOLS = [
    {
        "type": "function",
        "name": MOVE_FWD_FUNCTION_NAME,
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

def handle_agent_tool_call(arguments, call_id):
    return json.dumps({
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps({
            "success": True
        })
    })