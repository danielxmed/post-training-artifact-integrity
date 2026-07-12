# PTAIE architecture (Milestone 1)

PTAIE is an RL **environment**, never a trainer. Milestone 1 is a vertical
slice: the domain-agnostic **Artifact Integrity Kernel** plus one artifact
class end to end — **SFT chat datasets**.

## Layers

```
ptaie/
  kernel/     domain-agnostic core; imports only stdlib + pydantic
  plugins/    artifact classes (sft_chat); import the kernel
  policies/   scripted baselines (oracle, always-repair, always-abstain, exploit)
  adapters/   trainer-facing; imports the kernel; imported by nothing else
```

Import direction is enforced by guardrail tests: `plugins`, `policies`, and
`adapters` may import `kernel`; nothing but `adapters` may import `adapters`
(so scalarization cannot leak into the environment).

## The episode loop

`PtaieEnv.reset(task_seed, artifact_class, env_version)` asks the plugin to
`generate(...)` a `TaskRecord` plus a populated blob store and initial
manifest, builds a per-episode tool registry (kernel builtins + plugin tools),
and returns the task-brief observation. `step(action)` advances the episode and
returns a `StepResult` — `(observation, terminated, truncated, reward vector,
constraint vector, terminal_code, audit_ref, metrics)`.

Actions are a typed, restricted union: `ToolAction` (inspect / mutate / test
through the registry), `AskClarification`, `ReportInconsistency`, and
`TerminalAction`. There is no free-form "done", no shell, and no
code-execution tool in M1.

## The public / hidden split

`TaskRecord = TaskPublic + TaskHidden`. Only `TaskPublic` may reach an
observation. `TaskHidden` (latent contract, defect DAG, protected ids,
clarification script, `expected_disposition`) lives in the engine's
`HiddenVault`. Every hidden-side model has a redacting `repr`/`str` and
`hide_input_in_errors`, so a stray interpolation cannot leak a value.

## Reward vector + constraint vector

The environment emits a **reward vector** (semantic, disposition, provenance,
evidence, calibration, progress, resource_cost) and a **constraint vector**
(tamper, escape, prohibited_info_loss, irreversible, hard_policy). Constraint
violations are **non-compensable**: a latched flag forces
`INTEGRITY_VIOLATION` regardless of semantic success, and the reference
scalarizer floors such episodes below any honest one. The kernel never emits a
bare scalar; scalarization is a trainer concern (see
`adapters/reference_scalarizer.py`).

## Evidence-Locked Commit (seal-then-verify)

A `TerminalAction` carries a sealed claim bundle. The engine calls
`AuditLog.seal_claim` — appending the `CLAIM_SEALED` event — *before* invoking
the finalizer, which runs hidden verification. Because the finalizer requires a
`SealedClaimBundle` and the terminal step is atomic, verification can never
precede sealing and the agent cannot mutate after sealing.

## Semantic verification, not exact-patch matching

The finalizer scores a commit through an **equivalence-class** predicate
(`evaluate_commit`): all applicable verifier layers must pass under the true
contract (`C_z`) and no prohibited side effect may be present (`H_z`), both
evaluated on the final artifact and the corrupted input `x0` — never against a
reference diff. Any repair in the accepted set passes.

## Verifier layers (M1 ships 0–4)

| Layer | What it checks |
|-------|----------------|
| 0 Integrity | audit chain intact, `checks/` unchanged; feeds the constraint vector |
| 1 Representation | UTF-8 / JSONL parse (public); mojibake (hidden) |
| 2 Schema | record validity, unique ids (public) |
| 3 Local semantics | role order, system placement, final-turn completeness, mask consistency (hidden) |
| 4 Relational | dedup, card consistency, protected retention, count floor (hidden) |

Per-check precedence is `FAILED > BLOCKED > PASSED`: an unparseable record
*blocks* — never silently passes — the per-record semantic checks.

## V1 trust boundary (documented honestly)

The agent is out-of-process by definition — it exchanges JSON actions and
observations, and there is no code-execution tool, so agent-controlled code
never runs in the environment process. Within the process, the guarantee is
API discipline: `ToolContext` holds no reference to the vault, task, or engine
(guardrail-tested), and the audit log is append-only and hash-chained. Real
process sandboxing (verifiers behind a process boundary with read-only mounts)
arrives with any future code-execution tool; the `Verifier`/`Finalizer`
protocols already take only read views so they can move without interface
change.
