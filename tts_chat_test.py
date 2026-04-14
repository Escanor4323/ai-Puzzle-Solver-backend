"""Standalone TTS + LLM chat test server.

Run:
    cd ai-Puzzle-Solver-backend
    python3.11 tts_chat_test.py

Then open http://localhost:8001 in your browser.
Voices available via the dropdown:
  • Fast (edge-tts): Christopher, Brian, Ryan, Eric
  • Caine Clone (XTTS-v2): ~12 s — actual voice cloning from caine_reference.wav
"""

import asyncio
import io
import os
import tempfile
import uvicorn
import anthropic
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from config import Settings
_settings = Settings()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

SYSTEM_PROMPT = (
    "You are Caine, the theatrical and unpredictable host of The Amazing Digital Circus. "
    "You're charismatic, slightly unhinged, and genuinely delighted by everything. "
    "Keep every response to 1-2 sentences. Be witty and in character."
)

_REFERENCE_WAV = Path(__file__).parent / "tts" / "caine_reference.wav"

# Voice catalogue shown in the UI dropdown.
VOICES = {
    "christopher": ("edge", "en-US-ChristopherNeural",   "Christopher (edge-tts, fast)"),
    "brian":       ("edge", "en-US-BrianMultilingualNeural", "Brian (edge-tts, fast)"),
    "ryan":        ("edge", "en-GB-RyanNeural",           "Ryan UK (edge-tts, fast)"),
    "eric":        ("edge", "en-US-EricNeural",           "Eric (edge-tts, fast)"),
    "caine":       ("xtts", None,                         "Caine clone (XTTS-v2, ~12 s)"),
}

# ── Singletons ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _llm():
    return anthropic.Anthropic(api_key=_settings.ANTHROPIC_API_KEY)

@lru_cache(maxsize=1)
def _xtts_model():
    """Load XTTS-v2 once (heavy — called lazily on first Caine-voice request)."""
    import torch
    import torchaudio
    import soundfile as _sf

    # Patch torchaudio.load → soundfile so torchcodec is not required.
    def _sf_load(uri, frame_offset=0, num_frames=-1, **_kw):
        data, sr = _sf.read(str(uri), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)
        if frame_offset:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        return tensor, sr

    torchaudio.load = _sf_load

    # transformers 5.x removed LogitsWarper — coqui-tts still imports it via
    # 'from transformers import LogitsWarper' which goes through the lazy loader.
    # Setting on sys.modules entry (not just the module reference) is required
    # for the lazy loader to resolve the name.
    import sys, transformers
    if not hasattr(transformers, "LogitsWarper"):
        from transformers import LogitsProcessor
        sys.modules["transformers"].LogitsWarper = LogitsProcessor  # type: ignore[attr-defined]

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.api import TTS

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return model

# ── Synthesis helpers ─────────────────────────────────────────────────────────

import re as _re

def _normalize_for_tts(text: str) -> str:
    """Clean text before XTTS-v2 synthesis to prevent audio artifacts.

    XTTS-v2 produces noise / glitches on many punctuation characters because
    they are out-of-distribution for its text encoder.  This function maps
    them to spoken-language equivalents or removes them.
    """
    # Strip markdown / formatting noise
    text = _re.sub(r'\*+([^*]+)\*+', r'\1', text)   # *bold* / **bold**
    text = _re.sub(r'_+([^_]+)_+', r'\1', text)      # _italic_
    text = _re.sub(r'`[^`]+`', '', text)              # `code`
    text = _re.sub(r'#+\s*', '', text)                # ## headings

    # Stage directions like (laughs), [sighs dramatically] — remove entirely
    text = _re.sub(r'\([^)]*\)', '', text)
    text = _re.sub(r'\[[^\]]*\]', '', text)

    # Em/en dashes → natural pause via comma
    text = text.replace('—', ',').replace('–', ',')
    # Clean up any space-before-comma left behind
    text = _re.sub(r'\s+,', ',', text)

    # Ellipsis → single period
    text = text.replace('…', '.').replace('...', '.')

    # Repeated punctuation that triggers glitches
    text = _re.sub(r'[!]{2,}', '!', text)
    text = _re.sub(r'[?]{2,}', '?', text)
    text = _re.sub(r'[.]{2,}', '.', text)

    # Characters with no spoken equivalent — remove
    text = _re.sub(r'[~^|\\/<>{}@#$%^&*+=]', '', text)

    # Emojis and non-ASCII symbols
    text = _re.sub(r'[^\x00-\x7F]+', '', text)

    # Collapse multiple spaces / newlines → single space
    text = _re.sub(r'\s+', ' ', text).strip()

    return text


def _synthesize_edge(text: str, voice_id: str) -> bytes:
    """Synthesize with edge-tts. Runs a fresh event loop (called from thread)."""
    import edge_tts

    async def _go():
        communicate = edge_tts.Communicate(text, voice=voice_id)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_go())
    finally:
        loop.close()

def _synthesize_xtts(text: str) -> bytes:
    """Synthesize with XTTS-v2 using the Caine reference WAV."""
    text = _normalize_for_tts(text)
    model = _xtts_model()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="caine_tts_")
    try:
        os.close(tmp_fd)
        model.tts_to_file(
            text=text,
            speaker_wav=str(_REFERENCE_WAV),
            language="en",
            file_path=tmp_path,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "christopher"   # key from VOICES dict

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)

@app.get("/voices")
async def list_voices():
    return [{"id": k, "label": v[2]} for k, v in VOICES.items()]

@app.post("/chat")
async def chat(req: ChatRequest):
    msg = _llm().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": req.message}],
    )
    return {"response": msg.content[0].text}

@app.post("/tts")
async def tts(req: TTSRequest):
    voice_key = req.voice if req.voice in VOICES else "christopher"
    backend, voice_id, _ = VOICES[voice_key]

    loop = asyncio.get_event_loop()

    if backend == "edge":
        audio = await loop.run_in_executor(_executor, _synthesize_edge, req.text, voice_id)
        media_type = "audio/mpeg"
    else:  # xtts
        if not _REFERENCE_WAV.exists():
            raise HTTPException(status_code=503, detail="caine_reference.wav not found")
        audio = await loop.run_in_executor(_executor, _synthesize_xtts, req.text)
        media_type = "audio/wav"

    return Response(content=audio, media_type=media_type)

# ── Embedded UI ───────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Caine Chat — TTS Test</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #0d0d0d;
    color: #e8e8e8;
    display: flex;
    flex-direction: column;
    height: 100dvh;
    max-width: 700px;
    margin: 0 auto;
    padding: 16px;
    gap: 10px;
  }
  h1 { font-size: 1.1rem; color: #aaa; font-weight: 400; text-align: center; }
  #voice-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85rem;
    color: #888;
  }
  select {
    flex: 1;
    padding: 8px 12px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    color: #e8e8e8;
    font-size: 0.9rem;
    cursor: pointer;
    outline: none;
  }
  select:focus { border-color: #555; }
  #voice-warn {
    font-size: 0.75rem;
    color: #b45309;
    min-height: 16px;
  }
  #log {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 4px 0;
  }
  .msg {
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 85%;
    line-height: 1.5;
    font-size: 0.95rem;
    white-space: pre-wrap;
  }
  .msg.user  { background: #1e3a5f; align-self: flex-end; border-bottom-right-radius: 3px; }
  .msg.ai    { background: #1a1a1a; border: 1px solid #333; align-self: flex-start; border-bottom-left-radius: 3px; }
  .msg.status{ color: #666; font-size: 0.8rem; align-self: center; background: none; }
  #form { display: flex; gap: 8px; }
  #input {
    flex: 1;
    padding: 12px 16px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 24px;
    color: #e8e8e8;
    font-size: 1rem;
    outline: none;
  }
  #input:focus { border-color: #555; }
  button {
    padding: 12px 20px;
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 24px;
    cursor: pointer;
    font-size: 0.95rem;
    transition: background 0.15s;
  }
  button:hover  { background: #1d4ed8; }
  button:disabled { background: #333; cursor: default; }
  #tts-status { font-size: 0.75rem; color: #555; text-align: center; min-height: 16px; }
</style>
</head>
<body>
<h1>🎪 Caine Chat — TTS Test</h1>

<div id="voice-bar">
  <span>Voice:</span>
  <select id="voice-sel"></select>
</div>
<div id="voice-warn"></div>

<div id="log"></div>
<div id="tts-status"></div>

<form id="form" onsubmit="return false">
  <input id="input" type="text" placeholder="Say something…" autocomplete="off" autofocus>
  <button id="btn" type="submit">Send</button>
</form>

<script>
const log     = document.getElementById('log');
const input   = document.getElementById('input');
const btn     = document.getElementById('btn');
const status  = document.getElementById('tts-status');
const voiceSel = document.getElementById('voice-sel');
const voiceWarn = document.getElementById('voice-warn');

// Populate voice dropdown from server
fetch('/voices').then(r => r.json()).then(voices => {
  voices.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.label;
    voiceSel.appendChild(opt);
  });
});

voiceSel.addEventListener('change', () => {
  voiceWarn.textContent = voiceSel.value === 'caine'
    ? '⚠ XTTS-v2 takes ~12 s per response — model loads on first use'
    : '';
});

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

// Single shared AudioContext — created once and reused.
let _ctx = null;
function getCtx() {
  if (!_ctx || _ctx.state === 'closed') _ctx = new AudioContext();
  return _ctx;
}

async function playAudio(arrayBuffer) {
  const ctx = getCtx();
  // Context was already resumed synchronously in send() during the user
  // gesture, so this await is usually instant.
  if (ctx.state === 'suspended') await ctx.resume();
  // slice(0) copies the buffer so decodeAudioData doesn't detach the original.
  const audioBuf = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const src = ctx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(ctx.destination);
  await new Promise(resolve => { src.onended = resolve; src.start(0); });
}

async function send() {
  const text = input.value.trim();
  if (!text) return;

  // Unlock the AudioContext NOW, synchronously, while the user-gesture
  // (click / Enter) is still on the call stack. The browser grants autoplay
  // permission at this point; the actual resume can finish asynchronously.
  getCtx().resume();

  input.value = '';
  btn.disabled = true;
  input.disabled = true;

  addMsg('user', text);
  const thinking = addMsg('status', '…');

  try {
    // 1. LLM response
    const chatResp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!chatResp.ok) {
      const err = await chatResp.json().catch(() => ({ detail: chatResp.statusText }));
      throw new Error(err.detail || chatResp.statusText);
    }
    const { response } = await chatResp.json();
    thinking.remove();
    addMsg('ai', response);

    // 2. TTS
    const voice = voiceSel.value || 'christopher';
    const synthLabel = voice === 'caine' ? '⏳ cloning voice (~12 s)…' : '🔊 synthesizing…';
    status.textContent = synthLabel;
    const ttsResp = await fetch('/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: response, voice }),
    });
    if (!ttsResp.ok) throw new Error('TTS failed: ' + ttsResp.statusText);
    const ab = await ttsResp.arrayBuffer();
    status.textContent = '🔊 playing…';
    await playAudio(ab);
    status.textContent = '';

  } catch (err) {
    thinking.remove();
    addMsg('status', '⚠ ' + err.message);
    status.textContent = '';
  } finally {
    btn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

document.getElementById('form').addEventListener('submit', send);
</script>
</body>
</html>"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Open http://localhost:8001 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
