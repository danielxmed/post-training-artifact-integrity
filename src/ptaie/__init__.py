"""PTAIE — Post-Training Artifact Integrity Environment.

An RL environment (not a trainer) for studying evidence-backed, reversible,
provenance-preserving control of post-training artifacts under partial
observability and asymmetric commit risk.
"""

from ptaie.version import ENV_VERSION

__version__ = "0.1.0.dev0"

__all__ = ["ENV_VERSION", "__version__"]
