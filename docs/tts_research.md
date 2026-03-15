# TTS / Voice Model Research

**Date:** 2026-03-15
**Context:** CHAI uses Nebius Token Factory ($100 inference credit) as its primary AI backend. We investigated whether we could use Nebius for TTS as well.

---

## Nebius Token Factory — no audio models

As of March 2026, Nebius AI Studio / Token Factory offers **no TTS or STT models**. The catalog covers:

- LLMs (DeepSeek, Llama, Qwen, Mistral, Kimi, GLM, Gemma, …)
- Image generation (Flux Dev / Schnell)
- Embeddings (BAAI/bge, Qwen3-Embedding, …)

There is no `/v1/audio/` endpoint. The API is OpenAI-compatible but audio is out of scope.

---

## Options considered

| Option | Quality | Hindi support | Cost | Complexity |
|--------|---------|---------------|------|------------|
| **gTTS** (Google, free) | Adequate | Yes | Free (rate-limited) | Already in `requirements.txt` |
| macOS `say` | Basic | No | Free | Used in `sim_demo.py` only |
| OpenAI TTS (`tts-1`) | Good | Limited | $15 / 1M chars | One extra API key |
| ElevenLabs | Excellent | Yes (custom voices) | Paid tier | Extra SDK |
| Kokoro TTS (open-source) | Very good | Limited | Free (self-hosted) | Deploy on Nebius VM |

---

## Decision

**Use gTTS for the live demo.** Reasons:

1. Already wired up in `chai/voice/tts.py` — no changes needed.
2. Supports Hindi natively (`lang="hi"`), which is CHAI's primary speech output language.
3. Free with no additional API key.
4. Quality is sufficient for a robot assistant announcing short phrases.

`sim_demo.py` additionally tries the macOS `say` command (English only, no network) as a lightweight fallback during local development.

---

## Future upgrade path

If richer voice quality is needed post-hackathon:

- **Drop-in upgrade**: swap gTTS for the OpenAI TTS API — same call pattern, better prosody.
- **Self-hosted on Nebius VM**: run Kokoro TTS on the H100 instance for zero per-call cost and low latency.
- **Hindi-first**: ElevenLabs has the best multilingual quality if Hindi naturalness becomes a priority.
