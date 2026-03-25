# PuzzleMind — Backend

FastAPI backend for an AI-driven puzzle game with face recognition,
persistent player memory, and jailbreak detection. Runs as a standalone
server during development and compiles to a Tauri sidecar for distribution.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env   # then edit .env with your API keys

# 4. Run the dev server
python main.py
# → Server starts at http://127.0.0.1:8008
# → Health check: http://127.0.0.1:8008/health
```

## Project Structure

```
├── main.py              # FastAPI entry point
├── config.py            # Pydantic settings
├── security/            # Hardware lock, encryption
├── ai/                  # Face engine, LLM, jailbreak detector, memory
├── game/                # Engine, puzzles, Elo, hints
├── player/              # Profiles, sessions, relationships
├── data/                # Database, vector store, knowledge graph, models
├── events/              # Async event bus
├── ws/                  # WebSocket router & protocol
├── tests/               # Test suite
└── build_scripts/       # Production build helpers
```

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload --port 8008

# Run tests
pytest tests/
```
