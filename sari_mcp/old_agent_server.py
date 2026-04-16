# For dev tool use the following arguments
# run --with python-3.10 --with mcp --with python-socketio --with requests --with websocket-client mcp_server.py
 
from mcp.server.fastmcp import FastMCP
import socketio
import time
from mcp import types
 
# Create an MCP server
mcp = FastMCP(
    name="sari-functions-mcp",
)
 
sio = socketio.Client(logger=False, engineio_logger=False)
 
sio.connect("http://localhost:6060")
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _emit_and_wait(event: str, payload=None, timeout: float = 2.0) -> bool:
    """Emit a socket event and block until the server acknowledges or times out.
 
    Returns True if the server reported success, False otherwise.
    """
    result = {"status": None}
 
    def ack_callback(response):
        result["status"] = response.get("success", False)
 
    sio.emit(event, payload, callback=ack_callback)
 
    t0 = time.time()
    while result["status"] is None and (time.time() - t0 < timeout):
        sio.sleep(0.01)
 
    return bool(result["status"])
 
 
def _emit_and_read(event: str, payload=None, timeout: float = 5.0) -> str:
    """Emit a socket event and block until the server returns OCR / text data.
 
    Returns the extracted text string, or an empty string on failure / timeout.
    """
    result = {"data": None}
 
    def ack_callback(response):
        result["data"] = response.get("text", "")
 
    sio.emit(event, payload, callback=ack_callback)
 
    t0 = time.time()
    while result["data"] is None and (time.time() - t0 < timeout):
        sio.sleep(0.01)
 
    return result["data"] or ""
 
 
# ---------------------------------------------------------------------------
# PERCEPTION TOOLS
# ---------------------------------------------------------------------------
 
@mcp.tool()
def center_object_on_screen(target_name: str) -> bool:
    """Center the agent's camera on the specified object using visual feedback
    from an object detector.
 
    Args:
        target_name (str): The name or description of the object the agent
            should center in view.
 
    Returns:
        bool: True if the object was successfully centered, False otherwise.
    """
    return _emit_and_wait("CENTER_OBJ", target_name)
 
 
# ---------------------------------------------------------------------------
# MANIPULATION TOOLS
# ---------------------------------------------------------------------------
 
@mcp.tool()
def grab_and_read_item(hand: str) -> str:
    """Extend, grasp, and inspect the object directly in front of the agent
    using the specified hand. Returns OCR-extracted details from the object.
 
    Args:
        hand (str): The hand to use for grasping the object. Must be "left".
 
    Returns:
        str: OCR-extracted text / details from the inspected object.
    """
    return _emit_and_read("GRAB_READ", {"hand": hand})
 
 
@mcp.tool()
def extend_left_hand_forward(units: int) -> bool:
    """Extend the agent's left hand forward. Each unit moves the hand
    0.025 metres forward.
 
    Args:
        units (int): Number of 0.025-unit increments to move the left hand
            forward.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("EXT_LFT_FWD", units)
 
 
@mcp.tool()
def extend_right_hand_forward(units: int) -> bool:
    """Extend the agent's right hand forward. Each unit moves the hand
    0.025 metres forward.
 
    Args:
        units (int): Number of 0.025-unit increments to move the right hand
            forward.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("EXT_RGT_FWD", units)
 
 
@mcp.tool()
def pull_left_hand_backward(units: int) -> bool:
    """Pull the agent's left hand backward. Each unit moves the hand
    0.025 metres backward.
 
    Args:
        units (int): Number of 0.025-unit increments to move the left hand
            backward.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("PUL_LFT_BCK", units)
 
 
@mcp.tool()
def pull_right_hand_backward(units: int) -> bool:
    """Pull the agent's right hand backward. Each unit moves the hand
    0.025 metres backward.
 
    Args:
        units (int): Number of 0.025-unit increments to move the right hand
            backward.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("PUL_RGT_BCK", units)
 
 
@mcp.tool()
def raise_left_hand(units: int) -> bool:
    """Raise the agent's left hand upward. Each unit raises the hand
    0.025 metres.
 
    Args:
        units (int): Number of 0.025-unit increments to raise the left hand.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("RAISE_LFT", units)
 
 
@mcp.tool()
def raise_right_hand(units: int) -> bool:
    """Raise the agent's right hand upward. Each unit raises the hand
    0.025 metres.
 
    Args:
        units (int): Number of 0.025-unit increments to raise the right hand.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("RAISE_RGT", units)
 
 
@mcp.tool()
def lower_left_hand(units: int) -> bool:
    """Lower the agent's left hand downward. Each unit lowers the hand
    0.025 metres.
 
    Args:
        units (int): Number of 0.025-unit increments to lower the left hand.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("LOWER_LFT", units)
 
 
@mcp.tool()
def lower_right_hand(units: int) -> bool:
    """Lower the agent's right hand downward. Each unit lowers the hand
    0.025 metres.
 
    Args:
        units (int): Number of 0.025-unit increments to lower the right hand.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("LOWER_RGT", units)
 
 
@mcp.tool()
def toggle_left_grip() -> bool:
    """Toggle the grip of the agent's left hand (open ↔ closed).
 
    Returns:
        bool: True if the grip was toggled successfully, False otherwise.
    """
    return _emit_and_wait("GRIP_LFT")
 
 
@mcp.tool()
def toggle_right_grip() -> bool:
    """Toggle the grip of the agent's right hand (open ↔ closed).
 
    Returns:
        bool: True if the grip was toggled successfully, False otherwise.
    """
    return _emit_and_wait("GRIP_RGT")
 
 
@mcp.tool()
def rotate_and_read(hand: str) -> str:
    """Rotate an already-grabbed object clockwise using the specified hand and
    inspect it. Returns OCR-extracted details from the newly visible face.
 
    Args:
        hand (str): The hand holding the object. Must be "left" or "right".
 
    Returns:
        str: OCR-extracted text / details from the rotated object.
    """
    return _emit_and_read("ROT_READ", {"hand": hand})
 
 
# ---------------------------------------------------------------------------
# NAVIGATION TOOLS
# ---------------------------------------------------------------------------
 
@mcp.tool()
def move_forward(units: int) -> bool:
    """Move the agent forward. Each unit moves the agent 0.1 metres forward.
 
    Args:
        units (int): Number of 0.1-unit increments to move forward.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("MOVE_FWD", units)
 
 
@mcp.tool()
def move_backward(units: int) -> bool:
    """Move the agent backward. Each unit moves the agent 0.1 metres backward.
 
    Args:
        units (int): Number of 0.1-unit increments to move backward.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("MOVE_BCK", units)
 
 
@mcp.tool()
def move_left(units: int) -> bool:
    """Move the agent to the left. Each unit moves the agent 0.1 metres left.
 
    Args:
        units (int): Number of 0.1-unit increments to move left.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("MOVE_LFT", units)
 
 
@mcp.tool()
def move_right(units: int) -> bool:
    """Move the agent to the right. Each unit moves the agent 0.1 metres right.
 
    Args:
        units (int): Number of 0.1-unit increments to move right.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("MOVE_RGT", units)
 
 
@mcp.tool()
def pan_left(units: int) -> bool:
    """Pan the agent's camera to the left. Each unit rotates the camera
    2.5 degrees counterclockwise.
 
    Args:
        units (int): Number of 2.5-degree increments to pan left.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("PAN_LFT", units)
 
 
@mcp.tool()
def pan_right(units: int) -> bool:
    """Pan the agent's camera to the right. Each unit rotates the camera
    2.5 degrees clockwise.
 
    Args:
        units (int): Number of 2.5-degree increments to pan right.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("PAN_RGT", units)
 
 
@mcp.tool()
def pan_up(units: int) -> bool:
    """Pan the agent's camera upward. Each unit tilts the camera 2.5 degrees
    upward.
 
    Args:
        units (int): Number of 2.5-degree increments to pan up.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("PAN_UP", units)
 
 
@mcp.tool()
def pan_down(units: int) -> bool:
    """Pan the agent's camera downward. Each unit tilts the camera 2.5 degrees
    downward.
 
    Args:
        units (int): Number of 2.5-degree increments to pan down.
 
    Returns:
        bool: True if the action succeeded, False otherwise.
    """
    return _emit_and_wait("PAN_DOWN", units)
 
 
# ---------------------------------------------------------------------------
# SHARED TOOLS
# ---------------------------------------------------------------------------
 
@mcp.tool()
def stop() -> bool:
    """Stop the agent's execution once all goals have been met.
 
    Returns:
        bool: True if the stop signal was acknowledged, False otherwise.
    """
    return _emit_and_wait("STOP")
 
 
if __name__ == "__main__":
    mcp.run(transport='stdio')