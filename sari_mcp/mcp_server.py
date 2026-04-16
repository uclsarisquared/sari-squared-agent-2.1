# For dev tool use the following arguments
# run --with python 3.10 --with mcp --with python-socketio --with requests --with websocket-client --with moondream mcp_server.py

# Debug by running
# npx @modelcontextprotocol/inspector venv/bin/python sari_mcp/mcp_server.py

from mcp.server.fastmcp import FastMCP, Image
import socketio
import time
import base64
from mcp import types
import json

# Create an MCP server
mcp = FastMCP(
    name="sari-mcp",
)

sio = socketio.Client(logger=False, engineio_logger=False)

sio.connect("http://localhost:6060")

SERVER_ERROR_RESP = {
    "isError": True,
    "content": [
        types.TextContent(type="text", text="Server Error")
    ]
}

def base_movement_func(distance: float, direction: str):
    resp = sio.call(
        f"MOVE_{direction}",
        distance
    )

    if resp:
        if resp["success"]:
            return [types.ImageContent(type="image", data=resp["image"],
                                       mimeType="image/jpeg")]
        else:
            raise Exception("Error collecting egocentric view.")
    else:
        raise Exception("Unity client not connected.")

@mcp.tool()
def move_forward(distance: float):
    """Move the player forward by the specified distance in meters. This is mainly used for minor adjustments to the player's position.
        Args:
        distance (float): The distance to move in meters.
    """
    base_movement_func(distance, "FWD")



@mcp.tool()
def move_backward(distance: float):
    """Move the player backward by the specified distance in meters. This is mainly used for minor adjustments to the player's position.
        Args:
        distance (float): The distance to move in meters.
    """
    base_movement_func(distance, "BCK")


@mcp.tool()
def move_left(distance: float):
    """Move the player left by the specified distance in meters. This is mainly used for minor adjustments to the player's position.
        Args:
        distance (float): The distance to move in meters.
    """
    base_movement_func(distance, "LFT")


@mcp.tool()
def move_right(distance: float):
    """Move the player right by the specified distance in meters. This is mainly used for minor adjustments to the player's position.
        Args:
        distance (float): The distance to move in meters.
    """
    base_movement_func(distance, "RGT")


@mcp.tool()
def rotate_left(angle: float):
    """
    Rotate the player counterclockwise by the specified angle in degrees. This is mainly used for minor adjustments to the player's position.
    Args:
    angle (float): The angle to turn in degrees.
    """
    resp = sio.call(
        "TURN_LFT",
        angle
    )

    if resp["success"]:
        return [
            types.ImageContent(type="image", data=json.dumps(resp),
                               mimeType="image/jpeg")
        ]
    return resp


@mcp.tool()
def rotate_right(angle: float):
    """Rotate the player clockwise by the specified angle in degrees. This is mainly used for minor adjustments to the player's position.
        Args:
        angle (float): The angle to turn in degrees.
    """
    resp = sio.call(
        "TURN_RGT",
        angle
    )

    if resp["success"]:
        return [
            types.ImageContent(type="image", data=json.dumps(resp),
                               mimeType="image/jpeg")
        ]
    return resp


@mcp.tool()
def look_up(angle: float):
    """Look up by the specified angle in degrees.
        The player can look up to a maximum of 90 degrees. If the specified angle exceeds the maximum, the player will look up to the maximum angle.
        Args:
        angle (float): The angle to look up in degrees.
    """
    resp = sio.call(
        "LOOK_UP",
        angle
    )

    if resp["success"]:
        return [
            types.ImageContent(type="image", data=json.dumps(resp),
                               mimeType="image/jpeg")
        ]
    return resp


@mcp.tool()
def look_down(angle: float):
    """Look down by the specified angle in degrees.
        The player can look down to a maximum of 90 degrees. If the specified angle exceeds the maximum, the player will look down to the maximum angle.
        Args:
        angle (float): The angle to look down in degrees.
    """
    resp = sio.call(
        "LOOK_DOWN",
        angle
    )

    if resp["success"]:
        return [
            types.ImageContent(type="image", data=json.dumps(resp),
                               mimeType="image/jpeg")
        ]
    return resp


# #HIGH ABSTRACTION TOOLS
# @mcp.tool()
# def move_to_item(product_name: str):
#     """Move the player to the aisle of the specified product.
#         Args:
#         product_name (str): The name of the product to locate.
#     """
#     result = {"status": None}  # mutable object to store result
#     # Define callback function to capture server response
#     def ack_callback(response):
#         result["status"] = response.get("success", False)
#     # Send command with callback
#     sio.emit(
#         "MOVE_TO_ITEM",
#         product_name,
#         callback = ack_callback
#     )
#     # Wait for callback (simple busy-wait)
#     timeout = 20.0  # seconds
#     t0 = time.time()
#     while result["status"] is None and (time.time() - t0 < timeout):
#         sio.sleep(0.01)  # allows SocketIO background thread to run
#     return bool(result["status"])

@mcp.tool()
def get_item_at_pixel(img_x: int, img_y: int):
    """
    Attempts to pick up an object based on normalized screen coordinates (0-100).

    Args:
        img_x: Horizontal position (STRICTLY 0-100). 0 = Left edge, 100 = Right edge.
        img_y: Vertical position (STRICTLY 0-100). 0 = TOP edge, 100 = BOTTOM edge.
               (Note: Standard computer vision orientation).

    Returns:
        bool: True if item was within 1.5m and successfully added to inventory.
              False if no item exists at coordinates OR item is too far (>1.5m).

    Important: If this returns False despite an object being visible, the object
    is likely out of reach. Move the player closer and try again.
    """
    if not (0 <= img_x <= 100 and 0 <= img_y <= 100):
        return False
    result = {"status": None}  # mutable object to store result

    # Define callback function to capture server response
    def ack_callback(response):
        result["status"] = response.get("success", False)

    # Send command with callback
    sio.emit(
        "PICK_ITEM",
        {"x": img_x, "y": img_y},
        callback=ack_callback
    )
    # Wait for callback (simple busy-wait)
    timeout = 10.0  # seconds
    t0 = time.time()
    while result["status"] is None and (time.time() - t0 < timeout):
        sio.sleep(0.01)  # allows SocketIO background thread to run
    return bool(result["status"])


# Moondream-assisted point at object tool

# @mcp.tool()
# def get_item_in_view(item: str):
#     """Attempts to pick up the object in view based on the object's name.
#     Arguments:
#         item: The name of the item to pick up.
#     Returns:
#         bool: True if item was successfully picked up, False if no item exists at the current view or if the item is out of reach.
#     """
#     x, y = point_at_object(item)
#     return get_item_at_pixel(x, y)


# Point-cloud tools

# @mcp.tool()
# def get_item_points():
#     """Get the key points of the items in view as a list of (x, y, z) coordinates in the player's local space.
#         Returns:


# temporary simplified walk_to_node
# @mcp.tool()
# def walk_to_node(node_name: int):
#     """Walk to the specified node in the store. The pre-defined nodes are the following:
#        1: Cornflakes
#        2: Ritz Crackers
#        3: Choco Crunchies
#
#         Args:
#         node_name (int): The number of the node to walk to.
#
#         returns: True if the player successfully walked to the node, False otherwise.
#     """
#     result = {"status": None}  # mutable object to store result
#
#     # Define callback function to capture server response
#     def ack_callback(response):
#         result["status"] = response.get("success", False)
#
#     # Send command with callback
#     sio.emit(
#         "MOVE_TO_ITEM",
#         node_name,
#         callback=ack_callback
#     )
#     # Wait for callback (simple busy-wait)
#     timeout = 10.0  # seconds
#     t0 = time.time()
#     while result["status"] is None and (time.time() - t0 < timeout):
#         sio.sleep(0.01)  # allows SocketIO background thread to run
#     return bool(result["status"])


# @mcp.tool()
# def move_to_checkout():
#     """Move the player to the checkout area."""
#     return True

# #Interaction tools
# @mcp.tool()
# def get_item_at_pixel(x: int, y: int, hand: bool):
#     """Get the name of the item at the specified pixel coordinates in the player's view.
#         Args:
#         x (int): The x-coordinate of the pixel.
#         y (int): The y-coordinate of the pixel.
#         which hand (bool): True for right hand, False for left hand.
#     """
#     return "item_name"
# @mcp.tool()
# def rotate_item_on_hand(x, y, z):
#     """Rotate the item in the player's hand by the specified angles.
#         Args:
#         x (float): Rotation angle around the x-axis in degrees.
#         y (float): Rotation angle around the y-axis in degrees.
#         z (float): Rotation angle around the z-axis in degrees.
#     """
#     return True
# @mcp.tool()
# # def drop_item_from_hand(hand: bool):
#     """Drop the item currently held in the player's hand."""
#     return True

@mcp.tool()
def get_current_view() -> list:
    """Get the jpg of the virtual agent's egocentric view.
        Returns:
        An MCP Image object containing the current view in JPEG format.
        The image's resolution is 1920x1080 pixels.
    """

    resp = sio.call(
        "GET_VIEW",
    )

    return [
        types.ImageContent(type="image", data=str(resp), mimeType="image/jpeg")
    ]


# @mcp.tool()
# def get_current_view_command():
#    """Get the jpg of the player's current view."""
#    return pathlib.Path("C:/Sari/MCP/currentview/current_view.jpg").read_bytes()
# Resources
# @mcp.resource("path/to/screenshots")
# def get_screenshots():
#     """Get the latest screenshots from the player's view."""
#     return ["screenshot1.png", "screenshot2.png"]
# @mcp.resource("path/to/item_database")
# def get_item_database():
#     """Get the item database."""
#     return "item_database.json"
# @mcp.resource("path/to/item_description")
# def get_item_description():
#     """Get the item description file."""
#     return "item_description.txt"
# #Start the MCP server

if __name__ == "__main__":
    mcp.run(transport='stdio')