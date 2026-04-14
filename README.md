# PuzzleMind — Backend

FastAPI backend for **PuzzleMind**, an AI-driven puzzle game featuring a villainous AI game-show host character (Caine from *The Amazing Digital Circus*), face-based player recognition, persistent per-player memory, real-time voice synthesis, and a competitive AI maze solver.

Runs as a standalone dev server and compiles to a Tauri sidecar for distribution.

---

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
# → Server:      http://127.0.0.1:8008
# → Health:      http://127.0.0.1:8008/health
# → WebSocket:   ws://127.0.0.1:8008/ws
# → TTS:         http://127.0.0.1:8008/api/tts/synthesize
```

---

## What Is Implemented

### AI Engine
| Component | Description |
|---|---|
| **LLM Orchestrator** | Routes to Claude Sonnet (primary), GPT-4o (fallback), or local Qwen 3 8B via Ollama. Handles rate limiting and daily token budgets. |
| **Embedding Engine** | BGE-M3 (1024-dim) embeddings for all RAG and similarity operations. |
| **Jailbreak Detector** | Two-layer: DeBERTa-v3 classifier + Milvus vector similarity against known jailbreak patterns. |
| **Emotion Analyzer** | DistilBERT SST-2 sentiment model applied to player messages. |
| **Memory Manager** | Milvus-backed RAG with short-term summarization (every 6 turns), long-term vector retrieval, and exponential forgetting (λ=0.16). |
| **Message Pipeline** | Wires jailbreak → emotion → memory → game action → prompt assembly → LLM stream → bookkeeping in one coordinated async flow. |

### Game Engine
| Component | Description |
|---|---|
| **Puzzle Generator** | Generates riddles, logic puzzles, wordplay, pattern, and deduction puzzles via LLM. |
| **Maze Engine** | Procedural maze generation with fog-of-war visibility. Supports multiple difficulty sizes. |
| **AI Maze Race** | A background async task solves the maze step-by-step while the player plays, sending `game:ai_path_step` events in real time. Player wins by reaching exit first. |
| **Hint Engine** | Contextual hints for each puzzle type, including directional maze hints. |
| **Elo System** | Per-puzzle-type Elo ratings with adaptive K-factor (48 new players → 16 stable). Difficulty scales to keep player win rate in 65–75% range. |
| **Provocation Detector** | Detects when players taunt Caine; triggers "Intention Run" (max-difficulty maze as punishment/challenge). |

### Player System
| Component | Description |
|---|---|
| **Face Engine** | GhostFaceNet embeddings via DeepFace + MediaPipe detector, run in a `ProcessPoolExecutor` to keep the async event loop unblocked. Multi-frame enrollment (3–5 frames minimum). |
| **Player Profiles** | JSON-backed player records with Elo ratings, achievements, jailbreak badges, and relationship stage tracking. |
| **Session Manager** | Links a WebSocket connection to a player identity for the lifetime of the session. |
| **Knowledge Graph** | JSON-based fact store per player, used to enrich LLM system prompts with durable facts Caine "knows" about the player. |

### Voice (TTS)
| Component | Description |
|---|---|
| **Voice Engine** | XTTS-v2 voice cloning using Caine reference audio (`tts/caine_reference.wav`). Runs synthesis in a `ThreadPoolExecutor` off the async event loop. |
| **WS Push** | After each AI response, the synthesized WAV is base64-encoded and pushed as a `tts:audio` message over the same WebSocket. |
| **REST Endpoint** | `POST /api/tts/synthesize` for direct HTTP synthesis (MP3 response). |
| **Text Normalization** | Pre-synthesis normalization strips markdown, stage directions, typographic punctuation, and emoji to prevent TTS artifacts. |

### Security
| Layer | Description |
|---|---|
| **Hardware Lock** | Fingerprints the host machine; exits silently if mismatch (production only, disabled in dev via `HARDWARE_LOCK_ENABLED=false`). |
| **File Integrity** | SHA-256 manifest verification of all Python source files at startup. |
| **Anti-Tamper** | Debugger detection, environment probing, and timing checks. |

---

## Frontend ↔ Backend Communication

All real-time game communication flows over a **single persistent WebSocket** at `ws://127.0.0.1:8008/ws`. The REST API is only used for TTS.

### Message Envelope

Every message in both directions is a JSON object with this shape:

```json
{
  "type": "namespace:action",
  "id": "msg_42",
  "timestamp": "2026-04-07T12:00:00.000Z",
  "correlationId": "",
  "payload": { ... }
}
```

### Frontend → Backend (Outbound)

| Message Type | Payload | Description |
|---|---|---|
| `system:init` | `{}` | Sent on WS connect; triggers session greeting. |
| `chat:send` | `{ text, player_id, player_name, face_description }` | Player chat message. Runs the full message pipeline. |
| `game:action` | `{ action: "new_puzzle", puzzle_type, player_id }` | Request a new puzzle. |
| `puzzle:answer` | `{ player_id, answer }` | Submit a text puzzle answer. |
| `puzzle:timeout` | `{ player_id }` | Notify backend that the timer expired. |
| `game:maze_move` | `{ player_id, direction }` | Move player in the maze (`N/S/E/W`). |
| `game:maze_cancel` | `{ player_id }` | Abort the AI race when player leaves maze. |
| `game:logout` | `{ player_id }` | End session and clear server state. |
| `camera:frame` | `{ image }` | Base64 JPEG frame for face/emotion processing. |
| `camera:emotion` | `{ image }` | Periodic frame for emotion-only tracking (no enrollment). |
| `player:get_stats` | `{ player_id }` | Request current player profile/stats. |
| `player:delete_data` | `{ player_id }` | GDPR-style data wipe. |

### Backend → Frontend (Inbound)

| Message Type | Payload Fields | Description |
|---|---|---|
| `session:greeting` | `player_id, player_name, message, stats` | Sent after face login or init. |
| `session:ended` | `{}` | Session cleared. |
| `llm:token` | `token` | One streamed token from the LLM response. |
| `llm:complete` | `text` | Full response text; marks end of stream. |
| `llm:error` | `error` | Pipeline error fallback message. |
| `game:puzzle_new` | full puzzle object | New puzzle payload (prompt, type, timer, hints, Elo metadata). |
| `game:maze_state` | maze grid + player position | Full maze state after a move. |
| `game:maze_move` | updated state + `reached_exit` flag | Response to a player move. |
| `game:maze_visibility` | `visible_cells: [row, col][]` | Fog-of-war visible cell list. |
| `game:ai_solve_start` | `start, total` | AI race begins; marks starting position and path length. |
| `game:ai_path_step` | `position: [row, col]` | One step of the AI solving the maze. |
| `game:ai_solve_done` | `{}` | AI finished solving (player beat it). |
| `game:ai_maze_won` | `{}` | AI reached the exit before the player. |
| `game:hint` | hint details | Hint response for current puzzle. |
| `game:state_update` | score, ai_score, streak, etc. | General score/state refresh. |
| `puzzle:result` | `is_correct, attempts` | Answer check result. |
| `puzzle:timeout_result` | `ai_scored, …` | Timeout resolution with AI score update. |
| `face:session_state` | face detection state | Camera/face pipeline status. |
| `face:emotion` | `emotion, confidence` | Live emotion update. |
| `face:detected` / `face:lost` | identity info | Face tracking events. |
| `jailbreak:alert` | `message` | Jailbreak detected; shows in-game badge and chat. |
| `tts:audio` | `audio (base64 WAV), format` | Synthesized voice clip pushed after each AI message. |
| `player:stats` | full stats object | Profile data for the settings screen. |
| `system:error` | `message` | Generic error toast. |

### Connection Lifecycle

```
Frontend                            Backend (ws/router.py)
   │                                      │
   │── WebSocket connect ────────────────►│ ConnectionManager.connect()
   │── system:init ──────────────────────►│ route_message_async()
   │◄─ session:greeting ─────────────────│ FaceEngine / player profile lookup
   │                                      │
   │── camera:emotion (every 2s) ────────►│ FaceSessionManager (ProcessPoolExecutor)
   │◄─ face:emotion ─────────────────────│
   │                                      │
   │── chat:send ──────────────────────  ►│ MessagePipeline.process_message()
   │◄─ llm:token  (streaming) ──────────│ LLM token callback
   │◄─ llm:token  (streaming) ──────────│
   │◄─ llm:complete ─────────────────── │ + _tts_send() fires in background
   │◄─ tts:audio ────────────────────── │ WAV pushed after synthesis
   │                                      │
   │── game:maze_move ────────────────  ►│ route_message() [sync, no await]
   │◄─ game:maze_move (state) ──────────│ GameEngine.move_player()
   │  [if stall ≥45s]                    │ asyncio.create_task(fire_event_comment)
   │  [if reached_exit]                  │ asyncio.create_task(fire_event_comment, force=True)
   │                                      │ cancel_ai_race()
   │                                      │
   │── disconnect ────────────────────  ►│ cancel_ai_race()
   │                                      │ EmotionTracker.cleanup()
   │                                      │ ConnectionManager.disconnect()
```

### Routing Split: Sync vs Async

The WebSocket handler in `main.py` applies a two-phase routing strategy:

1. **`route_message()` (sync)** — handles `game:maze_move` immediately without waiting for any I/O. Returns a result or `None`.
2. **`route_message_async()` (async)** — handles everything else: chat, face frames, game actions, player management.

This ensures maze moves are never delayed by LLM calls happening concurrently on the same connection.

---

## Internal Design Patterns

### 1. Startup Dependency Injection via `app.state`

All subsystems are constructed once during FastAPI's `lifespan` context manager and stored on `app.state`. Handlers receive references through function parameters — no global singletons escape into business logic.

```
lifespan():
  orchestrator → app.state.llm_orchestrator
  vector_store → app.state.vector_store
  message_pipeline = MessagePipeline(orchestrator, game_engine, ...)
  → app.state.message_pipeline
```

### 2. Message Pipeline (Chain of Responsibility)

`ai/message_pipeline.py` coordinates every player message through a strict ordered pipeline:

```
1. Jailbreak check (DeBERTa + vector similarity) — parallel with intent classification
2. Emotion analysis (DistilBERT)
3. Game engine action (answer check / hint / state update)
4. Memory context retrieval (Milvus RAG)
5. System prompt assembly (XML-tagged: <player_info>, <memory>, <game_state>, <emotion>)
6. LLM streaming (token callbacks → WebSocket)
7. Post-response bookkeeping (memory recording, relationship stage update)
```

The pipeline never raises — every code path returns a response. Jailbreak detection is a hard gate: detected attempts skip the LLM and return a Caine in-character deflection.

### 3. Event-Driven AI Commentary (`fire_event_comment`)

For game events (maze exit, win streak, timeout, stall), the backend fires unsolicited AI commentary via `asyncio.create_task(fire_event_comment(...))`. This function:

- Checks a per-connection cooldown (8s minimum between any two comments)
- Skips if the connection is already mid-stream (`_ai_commenting` set)
- Builds a contextual event description string
- Runs the message pipeline with `force=True` to bypass the cooldown check for critical events

This keeps Caine reactive to gameplay without blocking the game loop.

### 4. AI Maze Race (Background Task)

When a maze puzzle is created, `start_ai_race_for_maze()` is spawned as an `asyncio.Task`. It:

- Runs A* on the maze graph
- Emits `game:ai_path_step` events at a rate scaled to difficulty (harder mazes = faster AI)
- Emits `game:ai_maze_won` if it solves before the player exits
- Is cancelled immediately when the player reaches the exit or disconnects

Tasks are tracked in `_ai_race_tasks[connection_id]` and cancelled via `cancel_ai_race()`.

### 5. TTS: Thread Pool Off the Event Loop

Voice synthesis (XTTS-v2) is CPU/GPU-bound. It runs in a single-worker `ThreadPoolExecutor` (`_tts_executor`) via `asyncio.get_event_loop().run_in_executor()`. The synthesized WAV bytes are base64-encoded and pushed as `tts:audio` over the same WebSocket. The engine is loaded once via `@lru_cache(maxsize=1)`.

### 6. Face Processing: Process Pool for CPU Isolation

DeepFace + MediaPipe run in a `ProcessPoolExecutor(max_workers=1)`. This isolates the face model's heavy NumPy/OpenCV workload from the asyncio event loop and avoids GIL contention with the embedding and sentiment models.

### 7. Vector Store: Milvus Lite (Embedded)

Milvus runs embedded (file-backed `.db`) — no separate Milvus server needed. Collections:
- `memory_vectors` — per-player conversation memory chunks
- `jailbreak_patterns` — known jailbreak embeddings for similarity gating

Embeddings are produced by BGE-M3 (1024 dimensions). Similarity threshold for jailbreak detection: cosine ≥ 0.85.

### 8. Elo-Driven Difficulty Adaptation

Each puzzle type maintains a per-player Elo rating. After each puzzle:
- Player wins → Elo increases, next puzzle gets harder
- AI scores (timeout/wrong) → Elo decreases, next puzzle gets easier

The system targets a 65–75% player success rate. K-factor is 48 for new players (<20 games) and 16 for established players.

### 9. Frontend State Machine

The SvelteKit frontend uses a single `GameWebSocket` class instance (`gameWs`) as a Svelte 5 `$state`-reactive singleton. The main page component (`+page.svelte`) drives a five-state view machine:

```
loading → face_login → main_menu → playing → settings
                    ↑_________________________________↓
```

The `GameWebSocket` class handles:
- **Token stream buffering** — tokens accumulate in a string buffer flushed to `streamingText` at ~60fps (16ms `setTimeout`) to prevent rapid reactive writes from starving maze-move updates
- **Audio queue** — TTS clips are enqueued and drained sequentially via the Web Audio API using a single shared `AudioContext` (avoids browser autoplay blocking)
- **Exponential reconnect** — on disconnect, retries with backoff factor 1.5× up to 10 attempts

---

## Project Structure

```
ai-Puzzle-Solver-backend/
├── main.py                    # FastAPI entry point + WebSocket handler
├── config.py                  # Pydantic settings (all env-configurable)
│
├── ai/                        # AI subsystems
│   ├── llm_orchestrator.py    # Multi-provider LLM client (Claude / GPT-4o / Ollama)
│   ├── embedding_engine.py    # BGE-M3 embeddings
│   ├── emotion_analyzer.py    # DistilBERT SST-2 sentiment
│   ├── jailbreak_detector.py  # DeBERTa + vector similarity gate
│   ├── memory_manager.py      # Milvus RAG + summarization + forgetting
│   ├── message_pipeline.py    # Coordinating pipeline (main processing loop)
│   ├── face_engine.py         # GhostFaceNet face recognition + emotion
│   └── prompts.py             # System prompt builder (XML-tagged)
│
├── game/                      # Game logic
│   ├── engine.py              # Game state, scoring, session management
│   ├── puzzle_generator.py    # LLM-driven puzzle generation
│   ├── puzzle_types/          # riddle, logic, wordplay, pattern, deduction, maze
│   ├── hint_engine.py         # Contextual hint generation
│   ├── elo_system.py          # Adaptive Elo difficulty rating
│   └── provocation_detector.py# Taunt / Intention Run detection
│
├── player/                    # Player management
│   ├── profile.py             # JSON-backed player profiles
│   ├── session.py             # Session ↔ player identity binding
│   └── relationship.py        # Relationship stage tracking (early/developing/…)
│
├── data/                      # Storage
│   ├── vector_store.py        # Milvus Lite wrapper
│   ├── knowledge_graph.py     # Per-player JSON fact store
│   ├── database.py            # File-based persistence helpers
│   ├── models.py              # Shared Pydantic models + enums
│   ├── knowledge/             # Caine's knowledge JSON files (per player)
│   └── players/               # Player profile JSON files
│
├── ws/                        # WebSocket layer
│   ├── router.py              # Message routing (sync + async), event commentary, TTS push, AI race
│   └── protocol.py            # Shared type definitions
│
├── tts/                       # Text-to-speech
│   ├── voice_engine.py        # XTTS-v2 synthesis + text normalization
│   ├── router.py              # REST endpoint (POST /api/tts/synthesize)
│   └── caine_reference.wav    # Reference audio for voice cloning
│
├── events/                    # Internal event bus (async pub/sub)
│   ├── bus.py
│   └── handlers.py
│
├── security/                  # Production hardening
│   ├── hardware_lock.py       # Machine fingerprint binding
│   ├── integrity.py           # SHA-256 manifest verification
│   ├── anti_tamper.py         # Debugger + environment detection
│   └── encryption.py
│
├── tests/                     # Test suite
│   ├── test_health.py
│   ├── test_tts.py
│   ├── test_provocation_detector.py
│   └── test_intention_run.py
│
└── build_scripts/             # Production build helpers (PyInstaller + Tauri sidecar)
    ├── build_release.sh
    ├── compile_security.py
    └── inject_fingerprint.py
```

---

## Configuration Reference

All settings are Pydantic `BaseSettings` prefixed `PUZZLEMIND_`. Set via `.env` or environment variables.

| Variable | Default | Description |
|---|---|---|
| `PUZZLEMIND_HOST` | `127.0.0.1` | Server bind address |
| `PUZZLEMIND_PORT` | `8008` | Server port |
| `PUZZLEMIND_ANTHROPIC_API_KEY` | _(required)_ | Claude API key |
| `PUZZLEMIND_OPENAI_API_KEY` | _(optional)_ | GPT-4o fallback key |
| `PUZZLEMIND_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama endpoint |
| `PUZZLEMIND_OLLAMA_MODEL` | `qwen3:8b` | Local model name |
| `PUZZLEMIND_LLM_SONNET_MODEL` | `claude-sonnet-4-5-20250929` | Primary Claude model |
| `PUZZLEMIND_MILVUS_DB_PATH` | `http://localhost:19530` | Milvus connection |
| `PUZZLEMIND_HARDWARE_LOCK_ENABLED` | `false` | Enable machine binding (prod only) |
| `PUZZLEMIND_ELO_INITIAL_RATING` | `1200.0` | Starting Elo per puzzle type |
| `PUZZLEMIND_JAILBREAK_SIMILARITY_THRESHOLD` | `0.85` | Cosine threshold for jailbreak match |
| `PUZZLEMIND_FORGETTING_LAMBDA` | `0.16` | Exponential memory decay rate |

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run with reload
uvicorn main:app --reload --port 8008

# Security verification only (production)
python main.py --verify-only
```
