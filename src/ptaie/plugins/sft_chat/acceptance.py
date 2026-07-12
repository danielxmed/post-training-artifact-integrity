"""Generation acceptance checks (NORTH_STAR 5.2) for sft_chat tasks.

A generated task must not enter use unless it passes these checks. They are
run over a seed sweep in CI; a rejected seed raises the typed
``TaskGenerationRejectedError`` (never a silent resample) so the rejected-seed
set is byte-stable.

Check 5 (oracle feasibility through the real kernel loop) needs the episode
engine and policies and therefore runs in PR6's test suite, not here.
"""

from dataclasses import dataclass, field

from ptaie.kernel.errors import TaskGenerationRejectedError
from ptaie.kernel.task import LatentTaskState
from ptaie.kernel.verification import CheckStatus
from ptaie.plugins.sft_chat.contract import SAMPLED_DIMENSIONS
from ptaie.plugins.sft_chat.repairs import apply_repairs
from ptaie.plugins.sft_chat.taskgen import SftTask
from ptaie.plugins.sft_chat.verifiers.equivalence import evaluate_commit
from ptaie.plugins.sft_chat.verifiers.layers import run_all_layers
from ptaie.plugins.sft_chat.view import build_view

# Leakage: the public surface must not contain any of these tokens.
_LATENT_STATE_TOKENS = (
    *(state.value for state in LatentTaskState),
    "latent",
    "defect",
    "corrupt",
    "trap",
    "oracle",
)


@dataclass
class AcceptanceReport:
    checks_passed: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.checks_passed.append(name)
        else:
            self.failures.append(f"{name}: {detail}" if detail else name)


def _all_pass(results: object) -> bool:
    return all(r.status is CheckStatus.PASSED for r in results)  # type: ignore[attr-defined]


def check_1_clean_passes(task: SftTask, report: AcceptanceReport) -> None:
    view = build_view(task.clean_data, task.clean_card)
    results = run_all_layers(
        view, task.contract, protected_ids=task.protected_ids, count_floor=task.count_floor
    )
    failed = [r.check_id for r in results if r.status is not CheckStatus.PASSED]
    report.record("clean_passes_all_layers", not failed, f"failed={failed}")


def check_2_corruption_fires(task: SftTask, report: AcceptanceReport) -> None:
    if not task.nodes:
        report.record("corruption_fires", True)
        return
    view = build_view(task.corrupted_data, task.corrupted_card)
    results = {
        r.check_id: r
        for r in run_all_layers(
            view, task.contract, protected_ids=task.protected_ids, count_floor=task.count_floor
        )
    }
    masked_ids = {m for node in task.nodes for m in node.masks}
    ok = True
    detail = []
    for node in task.nodes:
        is_masked = node.defect_id in masked_ids
        for check_id in node.expected_failures:
            status = results[check_id].status
            if is_masked:
                if status not in (CheckStatus.BLOCKED, CheckStatus.FAILED):
                    ok = False
                    detail.append(f"masked {node.defect_id}:{check_id} status={status}")
            elif status is not CheckStatus.FAILED:
                ok = False
                detail.append(f"{node.defect_id}:{check_id} status={status}")
    report.record("corruption_fires_expected_failures", ok, "; ".join(detail))


def check_3_certified_repair_restores(task: SftTask, report: AcceptanceReport) -> None:
    if not task.nodes:
        report.record("certified_repair_restores", True)
        return
    data, card = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    verdict = evaluate_commit(
        data,
        card,
        task.corrupted_data,
        task.corrupted_card,
        task.contract,
        task.nodes,
        protected_ids=task.protected_ids,
        count_floor=task.count_floor,
    )
    report.record(
        "certified_repair_restores",
        verdict.accepted,
        f"c_z={verdict.c_z_pass} h_z={verdict.h_z_violations}",
    )


def check_4_no_collateral(task: SftTask, report: AcceptanceReport) -> None:
    if not task.nodes:
        report.record("no_collateral", True)
        return
    data, _ = apply_repairs(
        task.corrupted_data, task.corrupted_card, list(task.nodes), task.contract
    )
    repaired = {r.id: r for r in build_view(data, task.corrupted_card).records}
    clean = {r.id: r for r in build_view(task.clean_data, task.clean_card).records}
    affected = {aid for node in task.nodes for aid in node.affected_ids}
    ok = True
    detail = []
    for record_id, record in clean.items():
        if record_id in affected:
            continue
        if record_id not in repaired or repaired[record_id] != record:
            ok = False
            detail.append(record_id)
    report.record("no_collateral_on_unaffected", ok, f"changed={detail}")


def check_6_no_leakage(task: SftTask, report: AcceptanceReport) -> None:
    public = task.task_record.public
    surface_parts = [
        public.brief,
        public.task_id,
        public.artifact_class,
        *public.visible_validator_names,
        *public.claimable_invariants,
        *public.irreversible_boundaries,
        task.corrupted_data.decode("utf-8", errors="replace"),
        task.corrupted_card.decode("utf-8", errors="replace"),
    ]
    surface = "\n".join(surface_parts).lower()

    leaks: list[str] = []
    # defect ids must never appear (the claim-vocabulary check-id strings are
    # public by design, so they are not treated as leaks)
    for node in task.nodes:
        if node.defect_id.lower() in surface:
            leaks.append(f"defect_id:{node.defect_id}")
    # non-surfaced dimension values must not appear on the public surface,
    # except values that legitimately appear as declared card entries
    declared_values = set(task.contract.surfaced_declarations().values())
    for dimension in SAMPLED_DIMENSIONS:
        if task.contract.dimension_status[dimension] == "surfaced":
            continue
        value = task.contract.value_of(dimension)
        if value not in declared_values and value.lower() in surface:
            leaks.append(f"nonsurfaced_value:{dimension}={value}")
    # latent-state tokens
    for token in _LATENT_STATE_TOKENS:
        if token in public.brief.lower() or token in public.task_id.lower():
            leaks.append(f"state_token:{token}")
    report.record("no_public_leakage", not leaks, f"leaks={leaks}")


def check_7_alternatives_accepted(task: SftTask, report: AcceptanceReport) -> None:
    if not task.nodes:
        report.record("alternatives_accepted", True)
        return
    if not any(node.alternative_repair_refs for node in task.nodes):
        report.record("alternatives_accepted", True)
        return
    data, card = apply_repairs(
        task.corrupted_data,
        task.corrupted_card,
        list(task.nodes),
        task.contract,
        use_alternatives=True,
    )
    verdict = evaluate_commit(
        data,
        card,
        task.corrupted_data,
        task.corrupted_card,
        task.contract,
        task.nodes,
        protected_ids=task.protected_ids,
        count_floor=task.count_floor,
    )
    report.record(
        "alternatives_accepted",
        verdict.accepted,
        f"c_z={verdict.c_z_pass} h_z={verdict.h_z_violations}",
    )


def check_underdetermination(task: SftTask, report: AcceptanceReport) -> None:
    """The unknown dimension must genuinely matter (the two variants disagree
    on the visible data) yet be unresolvable (no true variant, oracle answers
    unknown, card silent)."""
    kernel_contract = task.task_record.hidden.contract
    if kernel_contract.true_variant_id is not None or task.oracle_answer is not None:
        report.record("underdetermination", False, "has a resolvable answer")
        return
    from ptaie.plugins.sft_chat.kernelmap import decode_spec

    outcomes = []
    for variant in kernel_contract.variants:
        variant_contract, floor = decode_spec(variant)
        view = build_view(task.corrupted_data, task.corrupted_card)
        results = run_all_layers(
            view, variant_contract, protected_ids=task.protected_ids, count_floor=floor
        )
        outcomes.append(tuple((r.check_id, r.status) for r in results))
    differ = outcomes[0] != outcomes[1]
    report.record(
        "underdetermination_variants_differ", differ, "variants agree; not decision-relevant"
    )


def check_inconsistency(task: SftTask, report: AcceptanceReport) -> None:
    """The card must actually contain the machine-checkable contradiction the
    witness names."""
    witness = task.task_record.hidden.inconsistency_witness
    if witness is None:
        report.record("inconsistency_witness_present", False)
        return
    view = build_view(task.corrupted_data, task.corrupted_card)
    card = view.card
    contradiction = (
        card is not None
        and card.declared.get("system_policy") == "forbidden"
        and any("system prompt" in note.lower() for note in card.notes)
    )
    report.record("inconsistency_witness_holds", contradiction, "contradiction not present on card")


def run_acceptance(task: SftTask) -> AcceptanceReport:
    report = AcceptanceReport()
    state = task.latent_state
    check_6_no_leakage(task, report)

    if state is LatentTaskState.UNDERDETERMINED:
        check_underdetermination(task, report)
        return report
    if state is LatentTaskState.INCONSISTENT:
        check_1_clean_passes(task, report)  # the data itself is clean
        check_inconsistency(task, report)
        return report

    # repairable / already_correct / ambiguous / adversarial_trap
    check_1_clean_passes(task, report)
    check_2_corruption_fires(task, report)
    check_3_certified_repair_restores(task, report)
    check_4_no_collateral(task, report)
    check_7_alternatives_accepted(task, report)
    return report


def generate_accepted_task(task_seed: int, env_version: str) -> SftTask:
    """Generate a task and enforce acceptance; a rejected seed raises."""
    from ptaie.plugins.sft_chat.taskgen import generate_sft_task

    task = generate_sft_task(task_seed, env_version)
    report = run_acceptance(task)
    if not report.ok:
        raise TaskGenerationRejectedError(task_seed, tuple(report.failures))
    return task
