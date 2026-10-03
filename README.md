# Local Model Buddy

Local Model Buddy is a friendly terminal guide for people who want private,
local AI without guessing whether their computer is powerful enough. It checks
CPU, RAM, free disk space, GPU and VRAM; matches those resources to a curated
model catalog; tunes conservative runtime settings; and verifies the setup with
a real generation before saving it.

It is designed to find useful options for low-memory computers, not just gaming
PCs.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e .
lmb
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Commands

- `lmb` — guided scan, task interview, recommendation, download and smoke test
- `lmb scan` — show detected hardware; never downloads anything
- `lmb recommend --task "help me write Python"` — rank safe choices
- `lmb doctor` — inspect runtime health and the saved setup
- `lmb run "Explain recursion simply"` — use the last verified setup

## Runtimes

Local Model Buddy can detect and drive:

- [Ollama](https://ollama.com/download), the simplest choice for most people
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases), which offers
  detailed control and direct GGUF downloads
- [vLLM](https://docs.vllm.ai/), offered only when a suitable GPU and an
  existing vLLM installation are detected

The tool does not silently install system software. If no runtime is present,
it prints the appropriate one-time installation instructions. Model downloads
always require confirmation.

## How recommendations work

The recommender reserves memory for the operating system, estimates model
weights, KV cache and runtime workspace, checks free disk space, then ranks only
the combinations that fit. The recommendation is an estimate, so the guided
flow performs a smoke test. If the first choice fails or is unreasonably slow,
it tries the next smaller safe candidate.

Catalog sizes and suitability scores are curated estimates, not benchmark
claims. Actual speed varies by memory bandwidth, CPU features, GPU drivers and
other running programs.
