"""Artifact Integrity Kernel — the domain-agnostic core of PTAIE.

Import rules (enforced by guardrail tests):
- ``ptaie.kernel`` imports only the standard library and pydantic;
- ``ptaie.plugins`` and ``ptaie.policies`` import the kernel, never the reverse;
- nothing outside tests imports ``ptaie.adapters``.
"""
