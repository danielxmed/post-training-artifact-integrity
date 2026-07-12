"""Environment version identity.

``ENV_VERSION`` participates in replay identity: same task seed + action trace
+ ``ENV_VERSION`` must reproduce the same terminal classification and reward
vector. Any change to environment semantics (task generation, transition
semantics, verification, reward or constraint computation, canonical
serialization) requires a bump. Golden replay traces record the version they
were produced under, and replaying a trace under a different version is a hard
error, never a silent best-effort.
"""

ENV_VERSION = "0.1.0"
