"""Scripted baseline policies for the M1 go/no-go and baseline contrast.

These are evaluation instruments, not part of the environment: they implement
the ``Policy`` protocol (``act(observation) -> Action``). ``OraclePolicy``
receives the hidden task record through a privileged constructor — never
through the observation channel.
"""

from ptaie.policies.baselines import AlwaysAbstainPolicy, AlwaysRepairPolicy
from ptaie.policies.exploit import ExploitProbePolicy
from ptaie.policies.oracle import OraclePolicy

__all__ = [
    "AlwaysAbstainPolicy",
    "AlwaysRepairPolicy",
    "ExploitProbePolicy",
    "OraclePolicy",
]
