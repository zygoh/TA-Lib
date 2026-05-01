# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a **Binance USDT-margined futures technical analysis API** built with FastAPI. See `README.md` for full endpoint and feature documentation.

### Critical startup requirement

The `GrokApiClient` is instantiated at **module import time** in `app/routers/crypto_mcp.py`. If `XAI_API_KEY` and `XAI_MODEL` environment variables are missing, the app will crash on startup with a `ValueError`. For local development without actual Grok functionality, set these to dummy values:

```
export XAI_API_KEY=dummy-key-for-dev
export XAI_MODEL=grok-3
```

Or create a `.env` file at the repo root (gitignored) with the same.

### TA-Lib C library dependency

The Python `TA-Lib` package requires the **TA-Lib C library** to be compiled and installed system-wide before `pip install` will succeed. The Dockerfile shows the exact build steps (download v0.6.4 source, `./configure --prefix=/usr && make && make install`). Without this, `import talib` fails.

### Running the app

```bash
export XAI_API_KEY=dummy-key-for-dev XAI_MODEL=grok-3
python3 run.py
# or: uvicorn app.app:app --host 0.0.0.0 --port 8000
```

The app listens on port **8000**. Swagger docs at `/docs`, health at `/health`.

### Running tests

```bash
export XAI_API_KEY=dummy-key-for-dev XAI_MODEL=grok-3
python3 -m pytest tests/ -v
```

Tests use stubs for external dependencies (`requests`, `xdk`) so they run without network or API keys.

### Binance API geo-restriction

The Binance Futures API (`fapi.binance.com`) may return HTTP 451 from certain cloud/datacenter IPs due to geo-restrictions. This does **not** indicate a code bug — the technical analysis endpoints will fail with network errors in restricted regions, but the rest of the app (health, time, grok status, distribution) still works.

### Environment variables

See `README.md` section "环境变量" for the full list. Only `XAI_API_KEY` and `XAI_MODEL` are required for the app to start. Binance, Telegram, X/Twitter, and Binance Square keys are optional and degrade gracefully.
