"""
agent_tools2.py — Claude API tool definitions mirroring old_agent_server.py.

Defines AGENT_TOOLS (list of function schemas for the Claude API) and
handle_agent_tool_call() which executes each tool by talking to the Sari
Sandbox WebSocket at ws://localhost:8080/commands.
"""

import asyncio
import base64
import io
import json
import re

import websockets
from PIL import Image

# ---------------------------------------------------------------------------
# WebSocket helpers (mirrors old_agent_server.py)
# ---------------------------------------------------------------------------

WS_URI = "ws://localhost:8080/commands"
UNIT_CAP = 10


SCREENSHOT_TIMEOUT = 15  # seconds; screenshots can be slow to render


async def _send(command: dict, timeout: float | None = 10.0) -> bytes | str:
    async with websockets.connect(WS_URI, max_size=None) as ws:
        await ws.send(json.dumps(command))
        return await asyncio.wait_for(ws.recv(), timeout=timeout)


def _parse_agent_state(text: str) -> dict:
    coords = re.findall(r'\((.*?)\)', text, re.DOTALL)
    lines = text.split("\n")
    is_colliding = lines[2].split(": ")[-1].strip() == "True" if len(lines) > 2 else False
    return {
        "translation": tuple(map(float, coords[0].split(", "))) if coords else (),
        "rotation":    tuple(map(float, coords[1].split(", "))) if len(coords) > 1 else (),
        "isColliding": is_colliding,
    }


def _parse_hand_state(text: str) -> dict:
    coords = re.findall(r'\((.*?)\)', text, re.DOTALL)
    lines  = text.split("\n")
    return {
        "leftTranslation":  tuple(map(float, coords[0].split(", "))) if len(coords) > 0 else (),
        "leftRotation":     tuple(map(float, coords[1].split(", "))) if len(coords) > 1 else (),
        "rightTranslation": tuple(map(float, coords[2].split(", "))) if len(coords) > 2 else (),
        "rightRotation":    tuple(map(float, coords[3].split(", "))) if len(coords) > 3 else (),
        "leftHoveredObject":  lines[2].split(": ")[-1].strip() if len(lines) > 2 else "",
        "leftGrippedState":   lines[3].split(": ")[-1].strip() == "True" if len(lines) > 3 else False,
        "rightHoveredObject": lines[7].split(": ")[-1].strip() if len(lines) > 7 else "",
        "rightGrippedState":  lines[8].split(": ")[-1].strip() == "True" if len(lines) > 8 else False,
    }


async def _repeat_transform_agent(translation, rotation, units: int) -> dict:
    state = {}
    for _ in range(min(units, UNIT_CAP)):
        raw = await _send({"command": "TransformAgent", "translation": translation, "rotation": rotation})
        state = _parse_agent_state(raw)
    return state


async def _repeat_transform_hands(l_trans, l_rot, r_trans, r_rot, units: int) -> dict:
    state = {}
    for _ in range(min(units, UNIT_CAP)):
        raw = await _send({
            "command": "TransformHands",
            "leftTranslation":  l_trans,
            "leftRotation":     l_rot,
            "rightTranslation": r_trans,
            "rightRotation":    r_rot,
        })
        state = _parse_hand_state(raw)
    return state


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

NAVIGATION_TOOLS = [
    {
        "type": "function",
        "name": "move_forward",
        "description": "Move the agent forward. Each unit moves the agent 0.1 metres forward. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.1-metre steps to move forward (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_backward",
        "description": "Move the agent backward. Each unit moves the agent 0.1 metres backward. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.1-metre steps to move backward (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_left",
        "description": "Strafe the agent left. Each unit moves the agent 0.1 metres left. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.1-metre steps to strafe left (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "move_right",
        "description": "Strafe the agent right. Each unit moves the agent 0.1 metres right. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.1-metre steps to strafe right (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "pan_left",
        "description": "Rotate the camera left (counterclockwise). Each unit rotates 2.5 degrees. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 2.5-degree steps to pan left (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "pan_right",
        "description": "Rotate the camera right (clockwise). Each unit rotates 2.5 degrees. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 2.5-degree steps to pan right (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "tilt_up",
        "description": "Tilt the camera upward. Each unit tilts 2.5 degrees up. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 2.5-degree steps to tilt up (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "tilt_down",
        "description": "Tilt the camera downward. Each unit tilts 2.5 degrees down. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 2.5-degree steps to tilt down (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "transform_agent",
        "description": (
            "Apply an arbitrary single-step translation and rotation to the agent. "
            "Coordinates are in the agent's local frame."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "tx": {"type": "number", "description": "X translation (metres, left/right)."},
                "ty": {"type": "number", "description": "Y translation (metres, up/down)."},
                "tz": {"type": "number", "description": "Z translation (metres, forward/backward)."},
                "rx": {"type": "number", "description": "Pitch delta (degrees, tilt up/down)."},
                "ry": {"type": "number", "description": "Yaw delta (degrees, pan left/right)."},
                "rz": {"type": "number", "description": "Roll delta (degrees)."},
            },
            "required": ["tx", "ty", "tz", "rx", "ry", "rz"],
            "additionalProperties": False,
        },
    },
]

MANIPULATION_TOOLS = [
    # {
    #     "type": "function",
    #     "name": "extend_left_hand_forward",
    #     "description": "Extend the left hand forward. Each unit moves the hand 0.025 metres forward. Maximum 10 units per call.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False,
    #     },
    # },
    # {
    #     "type": "function",
    #     "name": "pull_left_hand_backward",
    #     "description": "Retract the left hand backward. Each unit moves the hand 0.025 metres backward. Maximum 10 units per call.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False,
    #     },
    # },
    {
        "type": "function",
        "name": "extend_right_hand_forward",
        "description": "Extend the right hand forward. Each unit moves the hand 0.025 metres forward. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "pull_right_hand_backward",
        "description": "Retract the right hand backward. Each unit moves the hand 0.025 metres backward. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    # {
    #     "type": "function",
    #     "name": "raise_left_hand",
    #     "description": "Raise the left hand upward. Each unit raises the hand 0.025 metres. Maximum 10 units per call.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False,
    #     },
    # },
    # {
    #     "type": "function",
    #     "name": "lower_left_hand",
    #     "description": "Lower the left hand downward. Each unit lowers the hand 0.025 metres. Maximum 10 units per call.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False,
    #     },
    # },
    {
        "type": "function",
        "name": "raise_right_hand",
        "description": "Raise the right hand upward. Each unit raises the hand 0.025 metres. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "lower_right_hand",
        "description": "Lower the right hand downward. Each unit lowers the hand 0.025 metres. Maximum 10 units per call.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 0.025-metre steps (1–10)."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    # {
    #     "type": "function",
    #     "name": "rotate_left_hand_clockwise",
    #     "description": "Rotate the left hand clockwise. Each unit rotates 15 degrees.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {"type": "integer", "description": "Number of 15-degree steps."},
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False,
    #     },
    # },
    # {
    #     "type": "function",
    #     "name": "rotate_left_hand_counterclockwise",
    #     "description": "Rotate the left hand counterclockwise. Each unit rotates 15 degrees.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {"type": "integer", "description": "Number of 15-degree steps."},
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False,
    #     },
    # },
    {
        "type": "function",
        "name": "rotate_right_hand_clockwise",
        "description": "Rotate the right hand clockwise. Each unit rotates 15 degrees.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 15-degree steps."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "rotate_right_hand_counterclockwise",
        "description": "Rotate the right hand counterclockwise. Each unit rotates 15 degrees.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "units": {"type": "integer", "description": "Number of 15-degree steps."},
            },
            "required": ["units"],
            "additionalProperties": False,
        },
    },
    # {
    #     "type": "function",
    #     "name": "toggle_left_grip",
    #     "description": "Toggle the left hand grip open ↔ closed.",
    #     "strict": True,
    #     "parameters": {
    #         "type": "object",
    #         "properties": {},
    #         "required": [],
    #         "additionalProperties": False,
    #     },
    # },
    {
        "type": "function",
        "name": "toggle_right_grip",
        "description": "Toggle the right hand grip open ↔ closed.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "transform_hands",
        "description": (
            "Apply arbitrary single-step translations and rotations to both hands. "
            "Coordinates are in the agent's local frame."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "l_tx": {"type": "number", "description": "Left hand X translation delta (metres)."},
                "l_ty": {"type": "number", "description": "Left hand Y translation delta (metres)."},
                "l_tz": {"type": "number", "description": "Left hand Z translation delta (metres)."},
                "l_rx": {"type": "number", "description": "Left hand pitch delta (degrees)."},
                "l_ry": {"type": "number", "description": "Left hand yaw delta (degrees)."},
                "l_rz": {"type": "number", "description": "Left hand roll delta (degrees)."},
                "r_tx": {"type": "number", "description": "Right hand X translation delta (metres)."},
                "r_ty": {"type": "number", "description": "Right hand Y translation delta (metres)."},
                "r_tz": {"type": "number", "description": "Right hand Z translation delta (metres)."},
                "r_rx": {"type": "number", "description": "Right hand pitch delta (degrees)."},
                "r_ry": {"type": "number", "description": "Right hand yaw delta (degrees)."},
                "r_rz": {"type": "number", "description": "Right hand roll delta (degrees)."},
            },
            "required": ["l_tx", "l_ty", "l_tz", "l_rx", "l_ry", "l_rz",
                         "r_tx", "r_ty", "r_tz", "r_rx", "r_ry", "r_rz"],
            "additionalProperties": False,
        },
    },
]

PERCEPTION_TOOLS = [
    {
        "type": "function",
        "name": "get_current_view",
        "description": "Capture the agent's egocentric view as a base64-encoded PNG image.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_scene_json",
        "description": "Request the scene's structured JSON state from the sandbox.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_agent_state",
        "description": "Query the agent's current position and rotation without moving.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_hand_state",
        "description": "Query both hands' current positions and grip states without moving.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "reset_environment",
        "description": "Reset the sandbox environment to its initial state.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]

TOOL_MODE_MAP: dict[str, str] = (
    {tool["name"]: "navigation"   for tool in NAVIGATION_TOOLS}
    | {tool["name"]: "manipulation" for tool in MANIPULATION_TOOLS}
    | {tool["name"]: "perception"   for tool in PERCEPTION_TOOLS}
)

SWITCH_MODE_TOOL = {
    "type": "function",
    "name": "switch_mode",
    "description": (
        "Switch the agent's active tool mode. "
        "You MUST call this before using tools from a different category. "
        "Available modes: 'navigation' (movement and camera), "
        "'manipulation' (hands and grips), "
        "'perception' (camera view, scene state, environment reset)."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["navigation", "manipulation", "perception"],
                "description": "The mode to switch to.",
            }
        },
        "required": ["mode"],
        "additionalProperties": False,
    },
}

# Combined list kept for convenience; sari_tui.py uses the per-category lists directly.
AGENT_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

async def _dispatch(name: str, args: dict):
    # Navigation
    if name == "move_forward":
        return await _repeat_transform_agent((0, 0, 0.1), (0, 0, 0), int(args["units"]))
    if name == "move_backward":
        return await _repeat_transform_agent((0, 0, -0.1), (0, 0, 0), int(args["units"]))
    if name == "move_left":
        return await _repeat_transform_agent((-0.1, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "move_right":
        return await _repeat_transform_agent((0.1, 0, 0), (0, 0, 0), int(args["units"]))

    # Camera
    if name == "pan_left":
        return await _repeat_transform_agent((0, 0, 0), (0, -2.5, 0), int(args["units"]))
    if name == "pan_right":
        return await _repeat_transform_agent((0, 0, 0), (0, 2.5, 0), int(args["units"]))
    if name == "tilt_up":
        return await _repeat_transform_agent((0, 0, 0), (-2.5, 0, 0), int(args["units"]))
    if name == "tilt_down":
        return await _repeat_transform_agent((0, 0, 0), (2.5, 0, 0), int(args["units"]))

    # Hand extension/retraction
    if name == "extend_left_hand_forward":
        return await _repeat_transform_hands((0, 0, 0.025), (0, 0, 0), (0, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "pull_left_hand_backward":
        return await _repeat_transform_hands((0, 0, -0.025), (0, 0, 0), (0, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "extend_right_hand_forward":
        return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, 0.025), (0, 0, 0), int(args["units"]))
    if name == "pull_right_hand_backward":
        return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, -0.025), (0, 0, 0), int(args["units"]))

    # Hand raise/lower
    if name == "raise_left_hand":
        return await _repeat_transform_hands((0, 0.025, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "lower_left_hand":
        return await _repeat_transform_hands((0, -0.025, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "raise_right_hand":
        return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0.025, 0), (0, 0, 0), int(args["units"]))
    if name == "lower_right_hand":
        return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, -0.025, 0), (0, 0, 0), int(args["units"]))

    # Hand rotation
    if name == "rotate_left_hand_clockwise":
        return await _repeat_transform_hands((0, 0, 0), (0, 15, 0), (0, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "rotate_left_hand_counterclockwise":
        return await _repeat_transform_hands((0, 0, 0), (0, -15, 0), (0, 0, 0), (0, 0, 0), int(args["units"]))
    if name == "rotate_right_hand_clockwise":
        return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 15, 0), int(args["units"]))
    if name == "rotate_right_hand_counterclockwise":
        return await _repeat_transform_hands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, -15, 0), int(args["units"]))

    # Grips
    if name == "toggle_left_grip":
        raw = await _send({"command": "ToggleLeftGrip"})
        return {"gripped": "True" in raw}
    if name == "toggle_right_grip":
        raw = await _send({"command": "ToggleRightGrip"})
        return {"gripped": "True" in raw}

    # Raw transforms
    if name == "transform_agent":
        raw = await _send({
            "command": "TransformAgent",
            "translation": (args["tx"], args["ty"], args["tz"]),
            "rotation":    (args["rx"], args["ry"], args["rz"]),
        })
        return _parse_agent_state(raw)
    if name == "transform_hands":
        raw = await _send({
            "command": "TransformHands",
            "leftTranslation":  (args["l_tx"], args["l_ty"], args["l_tz"]),
            "leftRotation":     (args["l_rx"], args["l_ry"], args["l_rz"]),
            "rightTranslation": (args["r_tx"], args["r_ty"], args["r_tz"]),
            "rightRotation":    (args["r_rx"], args["r_ry"], args["r_rz"]),
        })
        return _parse_hand_state(raw)

    # Perception
    if name == "get_current_view":
        raw = await _send({
            "command": "RequestScreenshot",
            "prefix": "",
            "suffix": "",
            "folder_name": "screenshots",
            "save_image": False,
        }, timeout=SCREENSHOT_TIMEOUT)
        img_bytes = raw if isinstance(raw, bytes) else raw.encode()
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((512, 512), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": b64, "mimeType": "image/png"}
    if name == "get_scene_json":
        return await _send({"command": "RequestJson"})

    # Utility
    if name == "get_agent_state":
        raw = await _send({"command": "TransformAgent", "translation": (0, 0, 0), "rotation": (0, 0, 0)})
        return _parse_agent_state(raw)
    if name == "get_hand_state":
        raw = await _send({
            "command": "TransformHands",
            "leftTranslation":  (0, 0, 0),
            "leftRotation":     (0, 0, 0),
            "rightTranslation": (0, 0, 0),
            "rightRotation":    (0, 0, 0),
        })
        return _parse_hand_state(raw)
    if name == "reset_environment":
        await _send({"command": "ResetEnvironment"})
        return {"success": True}

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

async def dispatch_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool and return the raw result dict."""
    return await _dispatch(tool_name, arguments)
