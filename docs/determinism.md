# Determinism

Deterministic replay is a non-negotiable property: the same task seed, action
trace, and `ENV_VERSION` must reproduce the same terminal classification,
reward vector, constraint vector, and **audit head hash**.

## What guarantees it

- **One RNG.** `ptaie.kernel.rng.DerivedRng` (sha256 seed derivation + labeled
  substreams) is the only randomness source in `kernel` and `plugins`. A
  guardrail test bans `time`, `datetime`, `uuid`, `secrets`, bare `random`,
  and `os.urandom` (dynamic-import and `os.urandom` forms included) everywhere
  but `rng.py`.
- **A logical clock, not a wall clock.** `AuditEvent.event_index` is the only
  clock. Nothing reads the system time.
- **Canonical serialization.** `ptaie.kernel.canonical` is the single hashing
  choke point: sorted keys, compact separators, `repr` floats (shortest
  round-trip), NaN/Inf and unpaired surrogates rejected, no Unicode
  normalization. Hash pydantic models only via
  `content_hash(model.model_dump(mode="json"))`.
- **Sorted, not iteration-ordered.** Manifest entries, tool listings, and
  constraint flags are explicitly sorted; nothing depends on `dict`/`set`
  iteration order in a serialized path.
- **No cross-episode state.** Each `PtaieEnv.reset` builds a fresh engine,
  blob store, and audit log; `EpisodeEngine` is single-use.

## How it is verified

- **`ReplayHarness`** records `(ActionTrace, ReplayResult)` and asserts
  `replay(trace) == recorded_result`, including the audit head hash (which
  transitively pins every event payload).
- **Golden traces** — `tests/fixtures/golden_traces/oracle.json`, one per
  latent state, are replayed for bit-identical outcomes
  (`tests/replay/test_golden_traces.py`). Regenerate with
  `python scripts/regenerate_golden.py`.
- **The `PYTHONHASHSEED` double-run** — CI runs
  `scripts/determinism_check.py --seeds 0:150` twice under *different*
  `PYTHONHASHSEED` values and byte-diffs the output, catching any
  hash-iteration-order dependence an identical double-run would hide. The
  local mirror is `tests/replay/test_determinism_hashseed.py`.

## The `ENV_VERSION` bump rule

`ENV_VERSION` (`src/ptaie/version.py`) is part of replay identity. **Any change
to environment semantics** — task generation, transition semantics,
verification, reward/constraint computation, or canonical serialization —
requires bumping it. A diff to the committed golden traces is the signal:
regenerate them only alongside an `ENV_VERSION` bump, and justify the change in
the PR. Replaying a trace under a mismatched version is a hard error, never a
silent best-effort.
