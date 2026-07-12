# PTAIE — Post-Training Artifact Integrity Environment

An RL **environment** (not a trainer) for studying whether LLM agents can exercise
evidence-backed, reversible, provenance-preserving control over post-training
artifacts (SFT datasets, preference data, tool trajectories, split/dedup manifests,
provenance metadata, tokenizer/collator/trainer configs) under partial observability
and asymmetric commit risk.

The scientific object is the agent's *decision discipline* — what to inspect,
whether to edit, ask, abstain, or commit — not raw repair throughput.

**Status: pre-alpha.** Milestone 1 — a vertical slice: the domain-agnostic
Artifact Integrity Kernel plus the SFT chat dataset artifact class, end to end
(six corruption operators, hidden layered verifiers, equivalence-class
acceptance, evidence-locked commit, deterministic replay, and scripted
oracle/baseline policies).

- Founding research/architecture report: [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md)
- Architecture overview: [`docs/architecture.md`](docs/architecture.md)
- Determinism contract: [`docs/determinism.md`](docs/determinism.md)
- Task format: [`docs/task-format.md`](docs/task-format.md)
- Contributor guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Developer/agent guidance and binding design invariants: [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md)

## Quickstart

Generate a task, run the reference oracle policy, and print the reward/constraint record:

```bash
uv run python scripts/quickstart.py 0
```

## Development

Python 3.12 + [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # install (creates .venv from the committed uv.lock)
uv run pytest                # run tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # typecheck
uv run pre-commit install    # optional: install git hooks
```

## License

Apache-2.0
