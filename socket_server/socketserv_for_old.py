import socketio
import asyncio

# Run server: uvicorn socketserv_for_old:app --port 6060

sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio)

unity_done = asyncio.Event()
unity_response = None
unity_sid = None


@sio.event
def connect(sid, environ, auth):
    global unity_sid

    print('connect', sid, environ['QUERY_STRING'])
    if 'sari_instance' in environ['QUERY_STRING']:
        unity_sid = sid
        print(f">> Unity client registered: {sid}")


def reset_unity_flags():
    global unity_response, unity_done
    unity_response = None
    unity_done.clear()


async def forward_to_unity(event: str, payload=None, timeout: float = 10.0):
    """Forward an event to the Unity client and block until UNITY_RESPONSE arrives."""
    global unity_done, unity_response

    unity_done = asyncio.Event()

    if payload is not None:
        await sio.emit(event, payload, to=unity_sid)
    else:
        await sio.emit(event, to=unity_sid)

    try:
        await asyncio.wait_for(unity_done.wait(), timeout=timeout)
        resp = unity_response
        reset_unity_flags()
        return resp
    except asyncio.TimeoutError:
        print(f">> Timeout waiting for Unity response to {event}")
        reset_unity_flags()
        return None


async def handle_success_event(sid, event: str, payload=None, timeout: float = 10.0) -> dict:
    """Handle an event whose response is a success/failure acknowledgment.

    Returns: {"success": bool} or {"success": False, "error": str}
    """
    print(f'[sid: {sid}] {event}({payload})')

    if unity_sid is None:
        print(">> Unity client not connected.")
        return {"success": False, "error": "Unity client not connected"}

    print(f">> Forwarding {event} to Unity client...")
    resp = await forward_to_unity(event, payload, timeout=timeout)

    if resp is None:
        return {"success": False, "error": "Timeout or no response from Unity"}
    if isinstance(resp, bool):
        return {"success": resp}
    if isinstance(resp, dict):
        return {"success": bool(resp.get("success", False))}
    return {"success": bool(resp)}


async def handle_text_event(sid, event: str, payload=None, timeout: float = 15.0) -> dict:
    """Handle an event whose response is OCR / text data.

    Returns: {"text": str}
    """
    print(f'[sid: {sid}] {event}({payload})')

    if unity_sid is None:
        print(">> Unity client not connected.")
        return {"text": "", "error": "Unity client not connected"}

    print(f">> Forwarding {event} to Unity client...")
    resp = await forward_to_unity(event, payload, timeout=timeout)

    if resp is None:
        return {"text": ""}
    if isinstance(resp, str):
        return {"text": resp}
    if isinstance(resp, dict):
        return {"text": resp.get("text", "")}
    return {"text": str(resp)}


# ---------------------------------------------------------------------------
# PERCEPTION
# ---------------------------------------------------------------------------

@sio.on('CENTER_OBJ')
async def center_object(sid, target_name):
    """Center the agent's camera on the named object using the object detector."""
    return await handle_success_event(sid, 'CENTER_OBJ', target_name)


# ---------------------------------------------------------------------------
# MANIPULATION
# ---------------------------------------------------------------------------

@sio.on('GRAB_READ')
async def grab_read(sid, data):
    """Extend, grasp, and OCR-inspect the object in front of the agent."""
    return await handle_text_event(sid, 'GRAB_READ', data)


@sio.on('EXT_LFT_FWD')
async def ext_lft_fwd(sid, units):
    """Extend the left hand forward by the given number of units."""
    return await handle_success_event(sid, 'EXT_LFT_FWD', units)


@sio.on('EXT_RGT_FWD')
async def ext_rgt_fwd(sid, units):
    """Extend the right hand forward by the given number of units."""
    return await handle_success_event(sid, 'EXT_RGT_FWD', units)


@sio.on('PUL_LFT_BCK')
async def pul_lft_bck(sid, units):
    """Pull the left hand backward by the given number of units."""
    return await handle_success_event(sid, 'PUL_LFT_BCK', units)


@sio.on('PUL_RGT_BCK')
async def pul_rgt_bck(sid, units):
    """Pull the right hand backward by the given number of units."""
    return await handle_success_event(sid, 'PUL_RGT_BCK', units)


@sio.on('RAISE_LFT')
async def raise_lft(sid, units):
    """Raise the left hand upward by the given number of units."""
    return await handle_success_event(sid, 'RAISE_LFT', units)


@sio.on('RAISE_RGT')
async def raise_rgt(sid, units):
    """Raise the right hand upward by the given number of units."""
    return await handle_success_event(sid, 'RAISE_RGT', units)


@sio.on('LOWER_LFT')
async def lower_lft(sid, units):
    """Lower the left hand downward by the given number of units."""
    return await handle_success_event(sid, 'LOWER_LFT', units)


@sio.on('LOWER_RGT')
async def lower_rgt(sid, units):
    """Lower the right hand downward by the given number of units."""
    return await handle_success_event(sid, 'LOWER_RGT', units)


@sio.on('GRIP_LFT')
async def grip_lft(sid, *args):
    """Toggle the left-hand grip (open <-> closed)."""
    return await handle_success_event(sid, 'GRIP_LFT')


@sio.on('GRIP_RGT')
async def grip_rgt(sid, *args):
    """Toggle the right-hand grip (open <-> closed)."""
    return await handle_success_event(sid, 'GRIP_RGT')


@sio.on('ROT_READ')
async def rot_read(sid, data):
    """Rotate a grabbed object clockwise and OCR-inspect the new face."""
    return await handle_text_event(sid, 'ROT_READ', data)


# ---------------------------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------------------------

@sio.on('MOVE_FWD')
async def move_fwd(sid, units):
    """Move the agent forward by the given number of units (0.1 m each)."""
    return await handle_success_event(sid, 'MOVE_FWD', units)


@sio.on('MOVE_BCK')
async def move_bck(sid, units):
    """Move the agent backward by the given number of units (0.1 m each)."""
    return await handle_success_event(sid, 'MOVE_BCK', units)


@sio.on('MOVE_LFT')
async def move_lft(sid, units):
    """Move the agent left by the given number of units (0.1 m each)."""
    return await handle_success_event(sid, 'MOVE_LFT', units)


@sio.on('MOVE_RGT')
async def move_rgt(sid, units):
    """Move the agent right by the given number of units (0.1 m each)."""
    return await handle_success_event(sid, 'MOVE_RGT', units)


@sio.on('PAN_LFT')
async def pan_lft(sid, units):
    """Pan the camera left by the given number of units (2.5 degrees each)."""
    return await handle_success_event(sid, 'PAN_LFT', units)


@sio.on('PAN_RGT')
async def pan_rgt(sid, units):
    """Pan the camera right by the given number of units (2.5 degrees each)."""
    return await handle_success_event(sid, 'PAN_RGT', units)


@sio.on('PAN_UP')
async def pan_up(sid, units):
    """Tilt the camera upward by the given number of units (2.5 degrees each)."""
    return await handle_success_event(sid, 'PAN_UP', units)


@sio.on('PAN_DOWN')
async def pan_down(sid, units):
    """Tilt the camera downward by the given number of units (2.5 degrees each)."""
    return await handle_success_event(sid, 'PAN_DOWN', units)


# ---------------------------------------------------------------------------
# SHARED
# ---------------------------------------------------------------------------

@sio.on('STOP')
async def stop(sid, *args):
    """Signal the agent to stop once all goals are met."""
    return await handle_success_event(sid, 'STOP')


# ---------------------------------------------------------------------------
# UNITY RESPONSE HANDLER
# ---------------------------------------------------------------------------

@sio.on('UNITY_RESPONSE')
def handle_unity_response(sid, data):
    global unity_response

    if isinstance(data, str):
        preview = data[:20] + ('...' if len(data) > 20 else '')
    else:
        preview = str(data)

    print(f">> Received response from Unity client: {preview}")

    unity_response = data

    if unity_done:
        unity_done.set()


@sio.event
def disconnect(sid, reason):
    global unity_sid

    print(sid, 'disconnected |', reason)
    if sid == unity_sid:
        unity_sid = None
        print(">> Unity client disconnected.")


if __name__ == '__main__':
    # Run server: uvicorn socketserv_for_old:app --port 6060
    pass
