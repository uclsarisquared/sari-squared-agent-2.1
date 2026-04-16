import socketio
import os
import asyncio

sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio)

unity_done = asyncio.Event()
unity_response = None
unity_sid = None

@sio.event
def connect(sid, environ, auth):
    global unity_sid

    print('connect ', sid, environ['QUERY_STRING'])
    if 'sari_instance' in environ['QUERY_STRING']:
        unity_sid = sid

def reset_unity_flags():
    global unity_response, unity_done
    unity_response = None
    unity_done.clear()

async def base_movement_function(sid, amount, direction: str):
    global unity_sid

    print(f'[sid: {sid}] MOVE_{direction}({amount})')

    if unity_sid is not None:
        print(">> Forwarding to Unity client...")
        await sio.emit(f"MOVE_{direction}", (amount,), to=unity_sid)
        resp = await socket_get_egocentric_view()
        return resp
    else:
        print(">> Unity client not connected.")
        return {"success": False, "reason": "Unity client not connected"}


@sio.on('MOVE_FWD')
async def move_fwd(sid, amount):
    return await base_movement_function(sid, amount, "FWD")

@sio.on('MOVE_BCK')
async def move_bck(sid, amount):
    return await base_movement_function(sid, amount, "BCK")


@sio.on('MOVE_LFT')
async def move_lft(sid, amount):
    return await base_movement_function(sid, amount, "LFT")


@sio.on('MOVE_RGT')
async def move_rgt(sid, amount):
    return await base_movement_function(sid, amount, "RFT")


@sio.on('TURN_LFT')
def turn_lft(sid, amount):
    print(f'[sid: {sid}] TURN_LFT({amount})')
    if unity_sid is not None:
        print(">> Sending to Unity client...")
        sio.emit("TURN_LFT", (amount,), to=unity_sid)
        return {"success": True}
    else:
        print(">> Unity client not connected.")
        return {"success": False}


@sio.on('TURN_RGT')
def turn_rgt(sid, amount):
    print(f'[sid: {sid}] TURN_RGT({amount})')
    if unity_sid is not None:
        print(">> Sending to Unity client...")
        sio.emit("TURN_RGT", (amount,), to=unity_sid)
        return {"success": True}
    else:
        print(">> Unity client not connected.")
        return {"success": False}


@sio.on('LOOK_UP')
def look_up(sid, amount):
    print(f'[sid: {sid}] LOOK_UP({amount})')
    if unity_sid is not None:
        print(">> Sending to Unity client...")
        sio.emit("LOOK_UP", (amount,), to=unity_sid)
        return {"success": True}
    else:
        print(">> Unity client not connected.")
        return {"success": False}


@sio.on('LOOK_DOWN')
def look_down(sid, amount):
    print(f'[sid: {sid}] LOOK_DOWN({amount})')
    if unity_sid is not None:
        print(">> Sending to Unity client...")
        sio.emit("LOOK_DOWN", (amount,), to=unity_sid)
        return {"success": True}
    else:
        print(">> Unity client not connected.")
        return {"success": False}


@sio.on('MOVE_TO_ITEM')
async def move_to_item(sid, item):
    global unity_response, unity_done
    print(f'[sid: {sid}] MOVE_TO_ITEM({item})')

    if unity_sid is None:
        print(">> Unity client not connected.")
        return {"success": False, "error": "Unity client not connected"}
    print(">> Sending to Unity client...")

    unity_done = asyncio.Event()

    await sio.emit("MOVE_TO_ITEM", item, to=unity_sid)

    try:
        await asyncio.wait_for(unity_done.wait(), timeout=10)
        print(f">> Unity response: {unity_response}")
        return {"success": True if unity_response else False}
    except asyncio.TimeoutError:
        print(">> Timeout waiting for Unity response.")
        return {"success": False, "error": "Timeout waiting for Unity response"}


@sio.on('PICK_ITEM')
async def pick_item(sid, data):
    global unity_response, unity_done

    x = data.get("x")
    y = data.get("y")

    print(f'[sid: {sid}] PICK_ITEM({x}, {y})')

    if unity_sid is None:
        print(">> Unity client not connected.")
        return {"success": False, "error": "Unity client not connected"}

    unity_done = asyncio.Event()

    print(">> Sending to Unity client...")
    await sio.emit("PICK_ITEM",[x,y], to=unity_sid)

    try:
        await asyncio.wait_for(unity_done.wait(), timeout=10)
        return {"success": True if unity_response else False}
    except asyncio.TimeoutError:
        print(">> Timeout waiting for Unity response.")
        return {"success": False, "error": "Timeout waiting for Unity response"}


@sio.on('GET_VIEW')
async def get_view(sid):
    global unity_response, unity_done

    if unity_sid is None:
        print(">> Unity client not connected.")
        return {"success": False, "error": "Unity client not connected"}

    unity_done = asyncio.Event()

    print(">> Forwarding command to Unity client...")
    return socket_get_egocentric_view()


async def socket_get_egocentric_view() -> dict:
    await sio.emit("GET_VIEW", to=unity_sid)

    try:
        await asyncio.wait_for(unity_done.wait(), timeout=10)
        data = {"success": True, "image": unity_response}
        reset_unity_flags()
        return data
    except asyncio.TimeoutError:
        print(">> Timeout waiting for Unity response.")
        reset_unity_flags()
        return {
            "success": False,
            "error": "Timeout waiting for Unity response"
        }

@sio.on('UNITY_RESPONSE')
def handle_unity_response(sid, data):
    global unity_response

    print(f">> Received response from Unity client: {data[:20]}{'...' if len(data) > 20 else ''}")

    # a success dict will just be {'success': True}
    unity_response = data

    if unity_done:
        unity_done.set()  # Signal that the response has been received



@sio.event
def disconnect(sid, reason):
    print(sid, 'disconnected |', reason)


if __name__ == '__main__':
    # run server by doing running: uvicorn socket_server:app --port 6060
    pass