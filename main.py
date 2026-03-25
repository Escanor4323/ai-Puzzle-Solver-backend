"""PuzzleMind backend entry point.

Minimal FastAPI application with a health-check endpoint and a
WebSocket echo endpoint.  No business-logic modules are imported here;
they are wired in during Phase 2.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PuzzleMind Backend",
    version="0.1.0",
    description="AI puzzle game backend with face recognition and persistent memory.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://localhost:5173",
        "http://127.0.0.1:1420",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Return a simple health-check response."""
    return {"status": "ok", "version": "0.1.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accept a WebSocket connection and echo messages back."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8008,
        reload=True,
    )
