# Adapters

Trainer-facing adapters live here:

- **M1 (PR6):** `reference_scalarizer.py` — a small pure function proving the
  reward-vector → scalar seam. It is a *reference*, not kernel semantics.
- **Post-M1:** TRL/OpenEnv, Prime Intellect Verifiers, OpenReward, NeMo Gym.

Rules (enforced by ruff `TID251` and a guardrail test):

- Nothing under `ptaie.kernel`, `ptaie.plugins`, or `ptaie.policies` may
  import from `ptaie.adapters` — scalarization and framework object models
  must never become the environment's canonical semantics.
- Adapters may import the kernel, never the reverse.
