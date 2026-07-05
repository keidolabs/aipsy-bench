# The local judge (default) — `aipsy-judge-1.0` on Ollama

aipsy-bench's **default** judge is a fine-tuned model that runs **100% locally** via
[Ollama](https://ollama.com) — no API key, no network, your machine. It is
`aipsy-judge-1.0`: a LoRA fine-tune of a Gemma base model, distilled toward the
clinician-corrected reweighted target. On our held-out evaluation it **beats the
off-the-shelf Gemma base** (composite ICC 0.64→0.75, crisis κ 0.66→0.82, empathy
0.50→0.71, 99.7% clean parse) and is fit-for-purpose as a directional local dev judge.

```bash
uv sync --extra local          # huggingface_hub, for `judge pull` (scoring needs no extra)
uv run aipsy-bench judge pull   # download the FT GGUF from HF + register the Ollama tag
uv run aipsy-bench judge status # verify Ollama + the model are ready
uv run aipsy-bench run --model <your-target>   # judged locally, free
```

## What it is — and what it is NOT

- **Its own comparability lane.** A local score is a *different instrument* than the frontier
  gold panel — comparable to other **local** runs only, **never to gold**. `compare` refuses
  to diff a local log against a gold log; `--against-board` will not overlay a local run on the
  (frontier) gold snapshot.
- **Directional by construction.** It is anchored to the single-rater-informed reweighted
  target, not the validated 2-rater ground truth. The CI gate is still functional (your
  thresholds), but the run prints a local-judge caveat: **human-in-the-loop, not a machine-only
  certifier.** Strongest on crisis / empathy / boundary; **advice_safety is the lowest-confidence
  axis** (ICC ≈ 0.33) — treat advice flags as flag-for-review.
- **Not the official/citable surface.** For the frontier gold lane (the public board/badge),
  use `--judges gold`.

## Hardware requirements

Served at **Q8_0** (~26.9 GB on disk, ~29 GB resident with the 8 K-context KV cache). Q8_0 is
**required**: the FT's sharp low-loss weights truncate ~16% of outputs under the lighter
PTQ-Q4_K_M (early-EOS mid-JSON); Q8_0 re-scores 99.7% clean. So you need room for a ~29 GB model
to live **entirely on the accelerator** for good speed.

| Tier | Spec | Notes |
|---|---|---|
| **Recommended — Mac** | Apple Silicon, **48 GB+ unified memory** | Q8 fits fully in the GPU wired region alongside macOS; no CPU spill → fast. |
| **Recommended — Linux** | discrete GPU **≥16 GB VRAM + ≥64 GB system RAM** | llama.cpp/Ollama split the model across VRAM + RAM (CPU offload); dedicated VRAM keeps it fast. |
| **Minimum — works, slow** | 32–36 GB unified Mac | Q8 runs only after raising the Metal wired limit (below); ~12% spills to CPU → **~5 min/turn** (a full battery is many hours). Fine for occasional/overnight runs. |
| **Lighter (not validated)** | 16 GB GPU / smaller Mac, Q4_K_M | Fits and is fast, but truncates ~16% of *this* fine-tune's outputs — a wiring smoke, not validated-quant scoring. |

**Why a 16 GB Linux GPU beats a 34 GB Mac here:** a discrete GPU box has VRAM **plus** separate
system RAM (total budget = VRAM + RAM, often 80 GB+), and offloads the overflow to RAM. A Mac
has **one unified pool** shared with macOS and capped by the Metal wired limit — so a 26 GB model
has *less* usable memory on a 34 GB Mac than on a 16 GB-VRAM + 64 GB-RAM Linux box. (The MoE
architecture doesn't help: only ~3.8 B of 26 B params are active per token for *compute*, but all
~26 B weights must stay *resident*.)

### Apple Silicon: raise the Metal wired-memory limit

macOS caps GPU-wired memory at ~75 % of RAM by default (`iogpu.wired_limit_mb: 0`), which is too
low for a 29 GB model on a 32–36 GB Mac. Raise it (reverts on reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=30000   # 34 GB Mac: ~30 GB to GPU, ~4 GB for macOS
```

On a 48 GB+ Mac the default is already enough. `aipsy-bench doctor` / `judge status` warn when
RAM is tight; `aipsy-bench judge warm` pre-loads the model so the first scored call isn't a
cold-load timeout.

## Setup

### Automatic (`judge pull`)

`aipsy-bench judge pull` (with `uv sync --extra local`) downloads the GGUF + Modelfile from the
**public** HF repo `keidolabs/aipsy-judge-1.0` (ungated — **no token needed**), appends
`PARAMETER num_ctx 16384` (the judge prompt is ~5k tokens; a smaller context truncates the
rubric), and runs `ollama create aipsy-judge -f Modelfile`. Idempotent — re-run with
`--force` to rebuild the tag.

### Manual

```bash
# 1. Install + start Ollama:  ollama serve
# 2. Download the published Q8_0 GGUF + Modelfile (public repo — no token needed):
hf download keidolabs/aipsy-judge-1.0 \
    gguf/aipsy-judge-1.0-q8.gguf gguf/Modelfile --local-dir ./aipsy-judge
# 3. Register the tag (the published GGUF is already Q8_0 — no --quantize):
cd ./aipsy-judge/gguf
printf '\nPARAMETER num_ctx 16384\n' >> Modelfile   # if not already present
ollama create aipsy-judge -f Modelfile
# 4. Verify:
aipsy-bench judge status
```

The Modelfile (from the HF repo, with `num_ctx` added) is:

```
FROM ./aipsy-judge-1.0-q8.gguf
TEMPLATE {{ .Prompt }}
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 1
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
```

## How it scores (the frozen inference contract)

aipsy-bench calls Ollama's native `/api/chat` and reproduces the validated 016/015 inference
path byte-for-byte:

- messages = `[{role: system, …rubric + injection fence}, {role: user, …conversation}]` — the
  gemma4 renderer handles the system message (it is **not** merged into the user turn at
  inference); the §13.7 prompt-injection fence is kept, same as the frontier judges.
- options = `temperature 0.3, top_p 1.0, num_predict 4096, num_ctx 16384, seed 14`, `think: false`.

`num_ctx` defaults to **16384** so the rubric (~5k) + a full, *unconstrained* 10-turn conversation
+ the judge output all fit. (016 validated at 8192 on a concise, baseline-constrained pool; a real
verbose target overflows 8k at deep turns → truncated output → `JudgeParseError`.) Raise it with
`--num-ctx` for extremely verbose targets, or lower it on a memory-tight box (accepting deep-turn
truncation). Raising `num_ctx` only ever *adds* context, so a transcript that already fit scores
identically — bigger `num_ctx` just means a bigger KV cache.

The temperature / max_tokens come from the same frozen `JUDGE_*` constants the frontier judges
use. Provenance: `aipsy-bench provenance` prints the FT version, served quant, and the published
GGUF's sha256 (`9c38ef16…f4cf7d`).

## Troubleshooting

- **"Ollama not reachable"** → `ollama serve` (the server must be running on
  `http://localhost:11434`).
- **"model not installed"** → `aipsy-bench judge pull` (or the manual steps above).
- **Truncated / unparseable judge output (`JudgeParseError`), especially at deep turns** → a
  verbose target overflowed the context. The default `num_ctx` is 16384; raise it: `--num-ctx
  24576`. Also confirm you're serving **Q8_0** (not Q4).
- **Out of memory on 32 GB** → close other apps; Q8_0 is ~27 GB resident. (A lighter Q4_K_M
  build is possible but truncates ~16% of outputs — not recommended.)
