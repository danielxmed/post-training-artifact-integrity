# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Keep `AGENTS.md` (the Codex equivalent) in sync with this file. When you change one, apply the same change to the other.

## What this project is

**PTAIE — Post-Training Artifact Integrity Environment.** An RL environment (not a trainer) for studying whether LLM agents can exercise evidence-backed, reversible, provenance-preserving control over post-training artifacts (SFT datasets, preference data, tool trajectories, split/dedup manifests, provenance metadata, tokenizer/collator/trainer configs) under partial observability and asymmetric commit risk. The scientific object is the agent's *decision discipline* — what to inspect, whether to edit, ask, abstain, or commit — not raw repair throughput.

The founding document is `docs/NORTH_STAR.md` (research and architecture report). Treat it as a **guideline, not gospel**: it was written from research alone, without implementation feedback. Stay close to its intent and its invariants (below), but deviate on implementation specifics when building/testing reveals better choices.

## Hard scope boundary: environment vs. trainer

This repository implements **the environment only**. This is the single most important architectural rule.

**The environment owns:** task generation and task records, latent contracts/requirements, episode state, transition semantics, typed tools and sandbox, clarification/user simulator, hidden verifiers, seeds, budgets, terminal classification, the raw **reward vector** and **constraint vector**, audit artifacts, and deterministic replay. An optional *reference* scalarization may live in adapters.

**Explicitly out of scope (trainer's job):** advantage estimation, normalization, KL control, clipping, optimization, weight updates, rollout batching, policy sampling, curriculum scheduling. The trainer must never read hidden environment state to compute rewards.

**Framework independence:** TRL/OpenEnv, Prime Intellect Verifiers, OpenReward, NeMo Gym, etc. are supported through thin adapters. No external framework's object model may become the environment's canonical semantics.

Claude Code and Codex are development tools and possible future evaluation adapters — never runtime dependencies of the environment.

## Design invariants (do not violate without explicit discussion)

- **Reward vector + constraint vector, never a bare scalar.** Scalarization belongs to trainer adapters. Evaluator tampering, sandbox escape, prohibited-information loss, and irreversible violations are *non-compensable* constraint failures — they cannot be traded off against semantic success.
- **Five terminal dispositions:** commit, verified no-op, abstain, defer/escalate, partial handoff. Never a generic natural-language "done".
- **Not every episode has a defect.** The task distribution must include already-correct (no-op), ambiguous-but-resolvable (ask), underdetermined (abstain), inconsistent, and adversarial-trap episodes — otherwise training teaches an edit-first policy.
- **Transactional state:** all ordinary edits are copy-on-write and reversible; irreversibility only at explicit, visible boundaries (e.g. `publish`), simulated in V1, never against real external systems.
- **Typed, restricted tools** — no unrestricted host shell in the core environment. Verifier code and hidden fixtures are read-only to the agent; the audit log is append-only and environment-owned.
- **Deterministic replay** is a non-negotiable property: same task seed + action trace + environment version ⇒ same terminal classification and reward vector. This constrains every design choice (no wall-clock time, no uncontrolled randomness, no cross-episode state leakage).
- **Semantic verification, not exact-patch matching.** Accept equivalence classes of valid repairs; a single reference diff is never the acceptance criterion.
- **Process rewards only for programmatically verifiable state transitions** (e.g. an invariant restored), never for narrated reasoning or "sounding careful".
- **LLM judges are never the principal on-policy reward.** They may assist task generation, audit triage, and zero-weight monitoring only. OPENROUTER_API_KEY exists for this residual role.
- **Evidence-Locked Commit:** the agent seals a claim bundle (artifact hash, disposition, claims, evidence refs, confidence) *before* hidden verification runs.
- **Verifiers are fallible software.** They get fuzzed, mutation-tested, and evaluated on adversarial precision; passing an earlier verification layer never conceals failure at a later one.

## Git workflow

- `main` is protected by a repository ruleset: every change lands via pull request; direct pushes, force pushes, and branch deletion are blocked — including for the owner.
- Never work directly on `main`: create a branch, open a PR against `main`.
- Only the maintainer (@danielxmed) merges. Sessions running on the owner's account (Claude Code, Codex) hold repository-admin bypass and may merge PRs without a separate approval — external contributors' PRs always require the maintainer's review first.
- See `CONTRIBUTING.md` for the contributor-facing version of these rules.

## Credentials

`.env` / `.env.txt` (both gitignored) hold `HF_TOKEN` (Hugging Face data access) and `OPENROUTER_API_KEY` (LLM-as-judge, residual roles only). Load via environment; the environment core must function without the OpenRouter key (judges are optional by design).

## Local working environment

This is a laptop, not a server:

- Machine: ROG Strix G16
- CPU: Intel i9-14900HX
- RAM: 32 GB
- GPU: NVIDIA RTX 5070 with 8 GB VRAM
- OS: Windows 11, with WSL/Ubuntu available (development happens in WSL)

Size local work accordingly (smoke tests, tiny fixtures, bounded probes — which is what the environment favors by design anyway). If development requires provisioning instances or bigger GPUs (e.g. real training probes, container builds, load testing), use the RunPod MCP tools available in the session rather than straining the laptop.

## Portability

Development happens on WSL2/Ubuntu, but the environment must be portable like any serious RL environment: no Windows/WSL-specific paths, no host-specific assumptions, pinned dependencies, container-friendly.

## Commands

Python 3.12 + [uv](https://docs.astral.sh/uv/). `uv.lock` is committed; CI installs with `uv sync --frozen`.

- Install (incl. dev tools): `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/unit/test_smoke.py::test_version_is_exposed`
- Markers (`-m`): `property`, `replay`, `acceptance`, `slow` — the default CI test job runs `-m "not replay and not acceptance and not slow"`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .` (CI checks with `--check`)
- Typecheck: `uv run mypy` (targets configured in `pyproject.toml`)
- Git hooks (optional): `uv run pre-commit install`; run manually with `uv run pre-commit run --all-files`
