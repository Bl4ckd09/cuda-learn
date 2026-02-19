# CLAUDE.md — Project Instructions for cuda-learn

## Project Overview
CUDA learning journey from zero to Flash Attention and raw transformer inference.
This repo builds on exercises inspired by [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course).

## Key Rules

### Insights Documentation
- **Always update `insights.md`** whenever you show an insight (the `★ Insight` blocks) during conversation.
- Include the insight in the appropriate phase section of `insights.md`.
- Keep insights educational and specific — avoid generic programming advice.

### Progress Tracking
- **Record ALL progress** in commit messages: successes, failures, and debugging processes.
- Commit messages should tell the story of what worked, what broke, and how it was fixed.
- Include benchmark results and performance numbers in commit messages when applicable.
- Never skip over debugging struggles — these are the most valuable learning moments.

### Debugging Protocol
- **Always consult Codex CLI** (`mcp__codex__codex`) when encountering bugs or unexpected behavior.
- Give Codex **full sandbox access** (`sandbox: "danger-full-access"`) — this repo contains only learning exercises, no secrets or production code.
- Try to diagnose first, then use Codex for second opinions or when stuck.
- Document the debugging process (what was tried, what failed, what fixed it) in both the commit message and `insights.md`.

### Improving on Infatoshi/cuda-course
- This repo is inspired by [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course).
- When implementing exercises, check if there are improvements over the original:
  - Better numerical stability
  - More efficient algorithms (e.g., online softmax vs standard)
  - Clearer code structure or comments
  - Additional benchmarks or correctness tests
  - Modern CUDA best practices (e.g., cooperative groups, warp-level primitives)
- Note any improvements or divergences from the original in commit messages.

### Git Conventions
- Commit under name `sun` only — no Co-Authored-By lines from Claude/Opus.
- Push to `origin/main` on GitHub (github.com/Bl4ckd09/cuda-learn).
- Update `.gitignore` for compiled binaries before committing each phase.

## Build Environment
- **GPU:** RTX 4070 (sm_89, Ada Lovelace, 46 SMs, 12.9 GB VRAM)
- **CUDA:** `/usr/local/cuda/bin/nvcc` — add to PATH if not found
- **Compile flags:** `-arch=sm_89 -O2`
- **cuBLAS:** link with `-lcublas` when needed

## Project Structure
```
cuda-learn/
├── CLAUDE.md              # This file
├── insights.md            # Accumulated learning insights (keep updated!)
├── phase0-foundations/    # C review + first kernels
├── phase1-matmul/         # Naive → tiled → coarsened → cuBLAS
├── phase2-elementwise/    # Activations, RMSNorm, softmax (incl. online)
├── phase3-attention/      # Naive → fused → Flash Attention
├── phase4-transformer/    # Complete decoder block in raw CUDA
└── reference/             # (gitignored) Infatoshi/cuda-course clone
```
