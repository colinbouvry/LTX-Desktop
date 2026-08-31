# LTX Desktop MCP server

Exposes LTX Desktop's local video generation to MCP clients. It wraps the app's Python
backend (a FastAPI service), so generation runs on your own GPU with the model weights
you already downloaded — no cloud calls, no API key.

## Tools

| Tool | Purpose |
| --- | --- |
| `ltx_generate_video` | Start a text-, image-, or audio-to-video render. Returns immediately. |
| `ltx_generation_progress` | Status, phase, step counter, and the output path once done. |
| `ltx_cancel_generation` | Cancel the render in progress. |
| `ltx_list_models` | The exact resolution / fps / duration combinations the active model accepts. |
| `ltx_list_loras` | LoRA adapters known to the backend. |
| `ltx_backend_status` | Reachability, GPU, and runtime policy. Start here when something fails. |

`ltx_generate_video` never blocks. A render takes minutes (1080p/10s measured ~5 min on
an RTX 5090), so the tool starts the work and returns; poll `ltx_generation_progress`
for the result. Errors from a failed render surface there too, rather than being lost.

## Requirements

A running LTX Desktop backend with at least one local checkpoint downloaded, and a GPU
the app considers capable of local generation — `ltx_backend_status` reports both. On an
API-only runtime policy, local generation and LoRAs are unavailable.

## Setup

```bash
cd mcp-server
uv sync
```

### Backend

The server talks to the backend over HTTP. Two ways to provide one:

**Headless (recommended).** Started without `LTX_AUTH_TOKEN`, the backend's auth
middleware is disabled, so no token juggling. Use a port other than the default 41954 if
the desktop app may also be open:

```bash
cd backend
LTX_APP_DATA_DIR="$LOCALAPPDATA/LTXDesktop" \
LTX_PORT=41955 \
uv run python ltx2_server.py
```

**Alongside the desktop app.** The app generates a random auth token per session and
passes it to its backend, so you must set `LTX_AUTH_TOKEN` to the same value — it is not
written anywhere readable. Headless is simpler.

Either way the two share one GPU slot: only one generation runs at a time, and this
server refuses to start one while the app is rendering.

### Client configuration

```json
{
  "mcpServers": {
    "ltx-desktop": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mcp-server", "run", "ltx-desktop-mcp"],
      "env": { "LTX_BACKEND_URL": "http://127.0.0.1:41955" }
    }
  }
}
```

## Tests

```bash
uv sync --extra test
uv run pytest
```

The suite drives the tools against a fake backend served over a real loopback socket,
so the same httpx client, headers, status handling and timeouts run as in production —
no patching. It covers the behaviours that are easy to get wrong: that starting a render
returns before it finishes, that a failed render surfaces its error instead of reporting
idle, and that a render started from the desktop app is not clobbered.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `LTX_BACKEND_URL` | `http://127.0.0.1:41954` | Where the backend listens. |
| `LTX_AUTH_TOKEN` | *(empty)* | Only needed when the backend was started with one. |

## Notes on settings

**Aspect ratio.** Anything beyond `16:9` and `9:16` is local-generation only; the cloud
path rejects it. Delivered dimensions snap to the /64 grid the two-stage pipeline
requires, so the real ratio can differ from the request by up to ~2.5%.

**Frame rate.** 24 and 48 land exactly on the VAE temporal grid. Other rates round the
frame count down slightly unless `fps × duration` is divisible by 8 — 30fps for 5s
delivers 4.83s.

**Cost.** Attention cost grows with pixels × frames, and nothing here enforces a VRAM
ceiling. High resolution combined with long duration can exhaust VRAM; on Windows the
driver then spills to system RAM and inference collapses in speed rather than failing.
Call `ltx_list_models` for what the active model advertises, and treat the far end of
that envelope as selectable rather than supported.
