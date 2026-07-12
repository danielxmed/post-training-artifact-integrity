"""The artifact-class plugin protocol and registry.

A plugin supplies everything domain-specific: task generation (including a
populated blob store + initial manifest), typed tools, visible validators, and
a clarification oracle. The hidden-verification + scoring hook (the
``Finalizer``) is injected into ``PtaieEnv`` separately and implemented in PR5.

The kernel owns the episode lifecycle, budgets, audit log, seal-then-verify
ordering, and reward/constraint emission; the plugin never computes rewards or
touches the audit log except through a tool's context.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ptaie.kernel.clarify import ClarificationOracle
from ptaie.kernel.errors import RegistryError
from ptaie.kernel.store.blob import BlobStore
from ptaie.kernel.store.manifest import Manifest
from ptaie.kernel.task import TaskRecord
from ptaie.kernel.tools.base import Tool
from ptaie.kernel.verification import Validator


@dataclass(frozen=True)
class GeneratedTask:
    """A generated task plus the materialized initial workspace."""

    task: TaskRecord
    store: BlobStore
    manifest: Manifest


@runtime_checkable
class ArtifactClassPlugin(Protocol):
    name: str
    plugin_version: str

    def generate(self, task_seed: int, env_version: str) -> GeneratedTask:
        """Pure function of (seed, env_version): the task record plus a fresh
        blob store and initial manifest (whose hash equals
        ``task.public.initial_manifest``)."""

    def tools(self) -> tuple[Tool, ...]:
        """Domain-typed inspect/mutate/test tools (registered with kernel builtins)."""

    def visible_validators(self) -> tuple[Validator, ...]:
        """Agent-runnable checks — a weak, public subset of hidden verification."""

    def clarification_oracle(self, task: TaskRecord) -> ClarificationOracle:
        """The deterministic oracle for this task's clarification script."""


class PluginRegistry:
    __slots__ = ("_plugins",)

    def __init__(self) -> None:
        self._plugins: dict[str, ArtifactClassPlugin] = {}

    def register(self, plugin: ArtifactClassPlugin) -> None:
        if plugin.name in self._plugins:
            raise RegistryError(f"duplicate plugin name {plugin.name!r}")
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> ArtifactClassPlugin:
        plugin = self._plugins.get(name)
        if plugin is None:
            raise RegistryError(f"unknown artifact class {name!r}")
        return plugin

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))
