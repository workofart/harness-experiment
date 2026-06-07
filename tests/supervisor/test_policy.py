"""Unit tests for src/supervisor/policy.py -- all pure, hand-built records only.

Covers the Fisher statistics, the folded ``gate`` (promotion + regression-veto +
the still-solving floor), ``combine``, ``budget_from_baseline``,
``validate_candidate``, and the ``decide`` truth table (plan.md §6/§9). No git,
docker, or I/O -- every input is constructed in-process.
"""

from __future__ import annotations

import pytest

from src.experiment.record import ExperimentResult, TaskResult, TrialResult
from src.supervisor.policy import (
    BaselineComparison,
    CandidateDiff,
    Conclude,
    Decision,
    Diagnose,
    Halt,
    LoopResult,
    PendingRun,
    ProposeAndLaunch,
    RefreshBaseline,
    RunVeto,
    World,
    budget_from_baseline,
    combine,
    compare_candidate_against_baseline,
    compute_fisher_exact_p_value,
    decide,
    gate,
    validate_candidate,
)


# --- builders ---------------------------------------------------------------


def _trial(run_id: str, *, solved: bool) -> TrialResult:
    return TrialResult(
        run_id=run_id,
        solved=solved,
        failure_mode="solved" if solved else "verified_rejected",
        verifier_passed=solved,
    )


def _crash(run_id: str) -> TrialResult:
    # An infra crash: error set, excluded from valid_trials and all scoring.
    return TrialResult(run_id=run_id, solved=False, failure_mode="crash", error="boom")


def _task(*solveds: bool, budget: int | None = None) -> TaskResult:
    trials = [_trial(f"r{i}", solved=s) for i, s in enumerate(solveds)]
    return TaskResult(
        expected_trial_count=budget if budget is not None else len(trials),
        trials=trials,
    )


def _exp(
    *,
    experiment_id: str = "exp",
    commit: str = "head",
    run_status: str = "completed",
    tasks: dict[str, TaskResult] | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash=commit,
        run_status=run_status,  # type: ignore[arg-type]
        started_at="2026-01-01T00:00:00+00:00",
        finished_at=None if run_status == "running" else "2026-01-01T01:00:00+00:00",
        tasks=tasks or {},
    )


# --- Fisher exact statistics ------------------------------------------------


def test_fisher_perfect_separation_is_significant() -> None:
    # candidate 5/5 vs baseline 0/5: the cleanest separation -> tiny p.
    p_value = compute_fisher_exact_p_value(
        candidate_solved=5, candidate_total=5, baseline_solved=0, baseline_total=5
    )
    assert p_value < 0.05


def test_fisher_identical_arms_is_unremarkable() -> None:
    # Same rate in both arms -> p == 1.0 (every table is at least as likely;
    # the two-sided sum is 1.0 up to float rounding).
    assert compute_fisher_exact_p_value(
        candidate_solved=2, candidate_total=4, baseline_solved=2, baseline_total=4
    ) == pytest.approx(1.0)


def test_fisher_is_symmetric_in_the_two_arms() -> None:
    left = compute_fisher_exact_p_value(
        candidate_solved=4, candidate_total=5, baseline_solved=1, baseline_total=5
    )
    right = compute_fisher_exact_p_value(
        candidate_solved=1, candidate_total=5, baseline_solved=4, baseline_total=5
    )
    assert left == pytest.approx(right)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            candidate_solved=0, candidate_total=0, baseline_solved=0, baseline_total=3
        ),
        dict(
            candidate_solved=0, candidate_total=3, baseline_solved=0, baseline_total=0
        ),
        dict(
            candidate_solved=4, candidate_total=3, baseline_solved=0, baseline_total=3
        ),
    ],
)
def test_fisher_rejects_degenerate_tables(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        compute_fisher_exact_p_value(**kwargs)


# --- compare_candidate_against_baseline -------------------------------------


def test_compare_no_candidate_trials_is_uncompared() -> None:
    verdict = compare_candidate_against_baseline(
        candidate_solved=0, candidate_total=0, baseline_solved=2, baseline_total=3
    )
    assert verdict.kind == "uncompared"
    assert verdict.p_value is None


def test_compare_frontier_uses_majority_not_significance() -> None:
    # baseline never solved (baseline_solved == 0): no significance test, just a
    # candidate majority-solve requirement -- so a single noisy solve cannot read
    # as improvement on noise alone.
    improved = compare_candidate_against_baseline(
        candidate_solved=2, candidate_total=3, baseline_solved=0, baseline_total=0
    )
    assert improved.kind == "improvement" and improved.p_value is None

    not_majority = compare_candidate_against_baseline(
        candidate_solved=1, candidate_total=3, baseline_solved=0, baseline_total=4
    )
    assert not_majority.kind == "unchanged" and not_majority.p_value is None


def test_compare_significant_regression_and_improvement() -> None:
    regressed = compare_candidate_against_baseline(
        candidate_solved=1, candidate_total=10, baseline_solved=9, baseline_total=10
    )
    assert regressed.kind == "regression"
    improved = compare_candidate_against_baseline(
        candidate_solved=9, candidate_total=10, baseline_solved=1, baseline_total=10
    )
    assert improved.kind == "improvement"


def test_compare_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=4, candidate_total=3, baseline_solved=0, baseline_total=3
        )


# --- gate: promotion --------------------------------------------------------


def test_gate_promotion_keeps_a_significant_aggregate_gain() -> None:
    # baseline solves 0 of 5 tasks, candidate solves all 5 -> aggregate Fisher
    # well under alpha.
    tasks = {f"t{i}": _task(False, False) for i in range(5)}
    baseline = _exp(tasks=tasks)
    candidate = _exp(tasks={f"t{i}": _task(True, True) for i in range(5)})
    decision = gate(candidate, baseline, task_ids=frozenset(tasks), purpose="promotion")
    assert decision.kind == "keep"
    assert set(decision.verdicts) == set(tasks)


def test_gate_promotion_discards_an_insignificant_gain() -> None:
    # +1 task (1->2 of 2) is a real-direction gain but Fisher at this size is
    # not significant at alpha=0.20 -> discard.
    baseline = _exp(tasks={"a": _task(True, True), "b": _task(False, False)})
    candidate = _exp(tasks={"a": _task(True, True), "b": _task(True, True)})
    decision = gate(
        candidate, baseline, task_ids=frozenset({"a", "b"}), purpose="promotion"
    )
    assert decision.kind == "discard"
    assert "not significant" in decision.reason


def test_gate_promotion_discards_when_no_aggregate_improvement() -> None:
    baseline = _exp(tasks={"a": _task(True, True), "b": _task(True, True)})
    candidate = _exp(tasks={"a": _task(True, True), "b": _task(False, False)})
    decision = gate(
        candidate, baseline, task_ids=frozenset({"a", "b"}), purpose="promotion"
    )
    assert decision.kind == "discard"
    assert "did not improve" in decision.reason


def test_gate_promotion_keeps_a_pure_frontier_panel() -> None:
    # No baseline trials anywhere -> no Fisher test; a candidate majority-solve
    # is the bar.
    baseline = _exp(tasks={})
    candidate = _exp(tasks={"a": _task(True, True), "b": _task(True, True)})
    decision = gate(
        candidate, baseline, task_ids=frozenset({"a", "b"}), purpose="promotion"
    )
    assert decision.kind == "keep"
    assert "improved" in decision.reason


# --- gate: regression-veto (can only block) ---------------------------------


def test_gate_veto_keeps_when_test_does_not_regress() -> None:
    baseline = _exp(tasks={"x": _task(True, True), "y": _task(True, True)})
    candidate = _exp(tasks={"x": _task(True, True), "y": _task(True, True)})
    decision = gate(
        candidate, baseline, task_ids=frozenset({"x", "y"}), purpose="regression_veto"
    )
    assert decision.kind == "keep"


def test_gate_veto_blocks_a_lost_task() -> None:
    baseline = _exp(tasks={"x": _task(True, True), "y": _task(True, True)})
    candidate = _exp(tasks={"x": _task(True, True), "y": _task(False, False)})
    decision = gate(
        candidate, baseline, task_ids=frozenset({"x", "y"}), purpose="regression_veto"
    )
    assert decision.kind == "discard"
    assert "regressed" in decision.reason


# --- gate: the still-solving floor ------------------------------------------


def test_gate_floor_never_calls_a_still_solving_candidate_a_regression() -> None:
    # Degenerate high-rate baseline (100/100); the raw per-task test would flag
    # the candidate's 3/5 as a significant regression even though it still
    # majority-solves the task. The floor downgrades it to unchanged.
    raw = compare_candidate_against_baseline(
        candidate_solved=3, candidate_total=5, baseline_solved=100, baseline_total=100
    )
    assert raw.kind == "regression"  # without the floor

    baseline = _exp(
        tasks={
            "a": TaskResult(
                expected_trial_count=100,
                trials=[_trial(f"b{i}", solved=True) for i in range(100)],
            )
        }
    )
    candidate = _exp(tasks={"a": _task(True, True, True, False, False)})  # 3/5
    decision = gate(
        candidate, baseline, task_ids=frozenset({"a"}), purpose="regression_veto"
    )
    assert decision.verdicts["a"].kind == "unchanged"
    # Candidate still solves the one task it shares -> veto does not block.
    assert decision.kind == "keep"


# --- gate: crash trials are excluded ----------------------------------------


def test_gate_excludes_crash_trials_from_scoring() -> None:
    # Two recorded trials, one a crash: only the valid (solved) trial scores, so
    # the verdict reads 1/1 -- the crash is invisible to the gate. Against a true
    # frontier baseline (no prior trials) that lone solve majority-solves -> keep.
    baseline = _exp(tasks={})
    candidate_task = TaskResult(
        expected_trial_count=2, trials=[_crash("r0"), _trial("r1", solved=True)]
    )
    candidate = _exp(tasks={"a": candidate_task})
    decision = gate(candidate, baseline, task_ids=frozenset({"a"}), purpose="promotion")
    assert decision.verdicts["a"].candidate_total == 1
    assert decision.verdicts["a"].candidate_solved == 1
    assert decision.kind == "keep"


# --- combine ----------------------------------------------------------------


def _decision(
    kind: str, verdicts: dict[str, BaselineComparison] | None = None
) -> Decision:
    return Decision(kind=kind, reason=f"{kind} reason", verdicts=verdicts or {})  # type: ignore[arg-type]


def _verdict(task_id: str) -> dict[str, BaselineComparison]:
    return {
        task_id: BaselineComparison(
            kind="improvement",
            candidate_solved=1,
            candidate_total=1,
            baseline_solved=0,
            baseline_total=0,
            p_value=None,
        )
    }


def test_combine_keep_requires_train_keep_and_no_veto() -> None:
    train = _decision("keep", _verdict("train-a"))
    test = _decision("keep", _verdict("test-a"))
    merged = combine(train, test)
    assert merged.kind == "keep"
    # Merged verdicts carry both disjoint panels' evidence (§12).
    assert set(merged.verdicts) == {"train-a", "test-a"}


def test_combine_train_discard_short_circuits() -> None:
    merged = combine(_decision("discard", _verdict("train-a")), None)
    assert merged.kind == "discard"
    assert "train discarded" in merged.reason


def test_combine_no_test_panel_keeps_on_train_keep() -> None:
    merged = combine(_decision("keep", _verdict("train-a")), None)
    assert merged.kind == "keep"
    assert "no test panel" in merged.reason


def test_combine_test_veto_overrides_train_keep() -> None:
    merged = combine(
        _decision("keep", _verdict("train-a")), _decision("discard", _verdict("test-a"))
    )
    assert merged.kind == "discard"
    assert "test vetoed" in merged.reason
    assert set(merged.verdicts) == {"train-a", "test-a"}


# --- budget_from_baseline (#7-decision) -------------------------------------


def test_budget_economizes_only_deterministically_solved_tasks() -> None:
    baseline = _exp(
        tasks={
            "solid": _task(True, True, True),  # every valid trial solved
            "flaky": _task(True, False),  # not deterministic
        }
    )
    budget = budget_from_baseline(
        baseline, task_ids=frozenset({"solid", "flaky", "fresh"}), full=3
    )
    assert budget == {"solid": 1, "flaky": 3, "fresh": 3}


def test_budget_is_inert_when_full_is_one() -> None:
    baseline = _exp(tasks={"solid": _task(True)})
    assert budget_from_baseline(baseline, task_ids=frozenset({"solid"}), full=1) == {
        "solid": 1
    }


def test_budget_without_a_baseline_is_full_everywhere() -> None:
    assert budget_from_baseline(None, task_ids=frozenset({"a", "b"}), full=3) == {
        "a": 3,
        "b": 3,
    }


# --- validate_candidate (§7) ------------------------------------------------


def test_validate_candidate_accepts_editable_paths() -> None:
    diff = CandidateDiff(
        changed_paths=("src/harness/core.py", "tests/harness/test_core.py"),
        added_lines=("    threshold = 5",),
    )
    assert validate_candidate(diff, task_ids=frozenset({"mteb-retrieve"})) is None


def test_validate_candidate_rejects_paths_outside_the_allowlist() -> None:
    diff = CandidateDiff(
        changed_paths=("src/harness/core.py", "src/env/harbor.py"),
        added_lines=(),
    )
    message = validate_candidate(diff, task_ids=frozenset())
    assert message is not None and "src/env/harbor.py" in message


def test_validate_candidate_rejects_literal_task_ids() -> None:
    diff = CandidateDiff(
        changed_paths=("src/harness/core.py",),
        added_lines=('    if task_id == "mteb-retrieve":',),
    )
    message = validate_candidate(diff, task_ids=frozenset({"mteb-retrieve"}))
    assert message is not None and "mteb-retrieve" in message


def test_validate_candidate_checks_paths_before_task_ids() -> None:
    # A diff that fails both checks surfaces the path violation first.
    diff = CandidateDiff(
        changed_paths=("config/harness_config.json",),
        added_lines=('"mteb-retrieve"',),
    )
    message = validate_candidate(diff, task_ids=frozenset({"mteb-retrieve"}))
    assert message is not None and "editable allowlist" in message


def test_validate_candidate_task_id_match_is_word_bounded() -> None:
    # A task id appearing only as a substring of a larger identifier is not a leak.
    diff = CandidateDiff(
        changed_paths=("src/harness/core.py",),
        added_lines=("    mteb_retrieval_helper = 1",),
    )
    assert validate_candidate(diff, task_ids=frozenset({"mteb"})) is None


# --- decide: the outer-loop truth table (§6) --------------------------------


def _loop(
    *,
    kind: str = "candidate",
    experiment_id: str = "cand",
    decision: Decision | None = None,
) -> LoopResult:
    return LoopResult(
        experiment_id=experiment_id,
        kind=kind,  # type: ignore[arg-type]
        focus_name="focus",
        parent_baseline_experiment_id="base" if kind == "candidate" else None,
        decision=decision,
    )


def _world(**overrides: object) -> World:
    base: dict[str, object] = dict(
        head_commit="head",
        primary_dirty=False,
        train_tasks=frozenset({"a"}),
        test_tasks=frozenset({"b"}),
        active_baseline=_exp(experiment_id="base", commit="head"),
        pending=None,
        undiagnosed_candidate_id=None,
    )
    base.update(overrides)
    return World(**base)  # type: ignore[arg-type]


def test_decide_halts_on_a_run_that_died_mid_run() -> None:
    pending = PendingRun(loop=_loop(), result=_exp(run_status="crashed"))
    command = decide(_world(pending=pending))
    assert isinstance(command, Halt) and "died mid-run" in command.reason


def test_decide_halts_on_dirty_primary_worktree() -> None:
    assert isinstance(decide(_world(primary_dirty=True)), Halt)


def test_decide_running_takes_precedence_over_dirty() -> None:
    # Rule 1 (mid-run death) fires before rule 2 (dirty) -- order matters.
    pending = PendingRun(loop=_loop(), result=_exp(run_status="running"))
    command = decide(_world(pending=pending, primary_dirty=True))
    assert isinstance(command, Halt) and "died mid-run" in command.reason


def test_decide_halts_on_launch_incomplete() -> None:
    pending = PendingRun(loop=_loop(), result=None)
    command = decide(_world(pending=pending))
    assert isinstance(command, Halt) and "never recorded" in command.reason


def test_decide_concludes_a_completed_baseline_run() -> None:
    pending = PendingRun(
        loop=_loop(kind="baseline", experiment_id="seed"),
        result=_exp(experiment_id="seed"),
    )
    command = decide(_world(pending=pending))
    assert command == Conclude("seed")


def test_decide_concludes_a_candidate_that_failed_train() -> None:
    # Candidate did not improve over the baseline on the train panel -> Conclude
    # (discard); the test panel never runs.
    baseline = _exp(experiment_id="base", commit="head", tasks={"a": _task(True, True)})
    candidate = _exp(experiment_id="cand", tasks={"a": _task(False, False)})
    pending = PendingRun(loop=_loop(), result=candidate)
    command = decide(_world(pending=pending, active_baseline=baseline))
    assert command == Conclude("cand")


def test_decide_runs_veto_when_train_keeps_and_test_pending() -> None:
    # Candidate improves on train (frontier majority-solve) and the test task is
    # not yet recorded -> RunVeto.
    baseline = _exp(experiment_id="base", commit="head", tasks={})
    candidate = _exp(experiment_id="cand", tasks={"a": _task(True, True)})
    pending = PendingRun(loop=_loop(), result=candidate)
    command = decide(_world(pending=pending, active_baseline=baseline))
    assert command == RunVeto("cand")


def test_decide_concludes_a_candidate_with_both_panels_run() -> None:
    baseline = _exp(experiment_id="base", commit="head", tasks={})
    candidate = _exp(
        experiment_id="cand", tasks={"a": _task(True, True), "b": _task(True, True)}
    )
    pending = PendingRun(loop=_loop(), result=candidate)
    command = decide(_world(pending=pending, active_baseline=baseline))
    assert command == Conclude("cand")


def test_decide_diagnoses_a_concluded_undiagnosed_candidate() -> None:
    command = decide(_world(undiagnosed_candidate_id="cand"))
    assert command == Diagnose("cand")


def test_decide_refreshes_baseline_when_none_at_head() -> None:
    assert isinstance(decide(_world(active_baseline=None)), RefreshBaseline)


def test_decide_refreshes_baseline_when_commit_is_stale() -> None:
    stale = _exp(experiment_id="base", commit="old")
    assert isinstance(decide(_world(active_baseline=stale)), RefreshBaseline)


def test_decide_proposes_when_baseline_is_current() -> None:
    assert isinstance(decide(_world()), ProposeAndLaunch)
