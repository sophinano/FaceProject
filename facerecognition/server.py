# 설치가 필요하다면: pip install websockets
import asyncio
import websockets

async def server(websocket, path):
    async for message in websocket:
        print(f"Received from client: {message}")
        response = f"Server received: {message}"
        await websocket.send(response)

start_server = websockets.serve(server, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
print("WebSocket server is running on ws://localhost:8765")
asyncio.get_event_loop().run_forever()
