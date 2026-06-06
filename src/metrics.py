"""Gate statistics: candidate-vs-baseline comparison primitives.

The Fisher exact test, the per-task and aggregate alphas, the
``BaselineComparison`` verdict, and ``compare_candidate_against_baseline`` --
the one function that answers "did the candidate beat the baseline on this
task?". Consumed by the promotion gate.

The foundation telemetry/majority helpers (``TaskMetrics``, ``FailureMode``,
``is_majority_solved``/``is_majority_decided``) live in ``src.contracts``; these
gate statistics move to ``supervisor/policy`` in the redesign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from src.contracts import is_majority_solved


def compute_fisher_exact_p_value(
    *,
    candidate_solved: int,
    candidate_total: int,
    baseline_solved: int,
    baseline_total: int,
) -> float:
    """Two-sided Fisher exact p-value for the 2x2 contingency table::

                  solved                       failed
        candidate candidate_solved             candidate_total - candidate_solved
        baseline  baseline_solved              baseline_total  - baseline_solved

    Unlike a one-sample binomial against the baseline *rate* (which treats that
    rate as a known point and so calls a candidate miss against a 3/3 baseline
    "impossible"), Fisher's exact test conditions on both margins and accounts
    for sampling uncertainty in BOTH arms. A small high-rate baseline is
    therefore correctly treated as weak evidence, which removes the
    small-sample false regressions that discarded solve-positive candidates.

    Two-sided p = sum of the probabilities of every table (with the margins
    fixed) that is no more likely than the observed one, clamped to [0, 1].
    """
    if candidate_total <= 0:
        raise ValueError("candidate_total must be positive")
    if baseline_total <= 0:
        raise ValueError("baseline_total must be positive")
    if not 0 <= candidate_solved <= candidate_total:
        raise ValueError("candidate_solved must be in [0, candidate_total]")
    if not 0 <= baseline_solved <= baseline_total:
        raise ValueError("baseline_solved must be in [0, baseline_total]")

    n = candidate_total + baseline_total
    row1 = candidate_total  # candidate-arm trials
    col1 = candidate_solved + baseline_solved  # total solved across both arms

    def _table_prob(solved_in_candidate: int) -> float:
        # Hypergeometric: P(candidate arm holds `solved_in_candidate` of the
        # `col1` solved trials) with both margins held fixed.
        return (
            math.comb(col1, solved_in_candidate)
            * math.comb(n - col1, row1 - solved_in_candidate)
            / math.comb(n, row1)
        )

    p_observed = _table_prob(candidate_solved)
    k_min = max(0, row1 - (n - col1))
    k_max = min(row1, col1)
    total = sum(
        prob
        for k in range(k_min, k_max + 1)
        if (prob := _table_prob(k)) <= p_observed * (1.0 + 1e-9)
    )
    return min(1.0, total)


# Two-sided alpha for the per-task Fisher exact test in `build_gate_verdicts`.
# These per-task verdicts are diagnostic evidence only -- they label each task's
# outcome; the promotion decision itself uses the aggregate alpha below. The
# strict 0.05 bar relies on Fisher exact being strongly conservative at the
# small per-task trial counts (n~3-5) so chance regressions stay rare per run.
PER_TASK_VERDICT_P_VALUE_ALPHA = 0.05

# Two-sided alpha for the gate's aggregate panel Fisher exact test, where each
# unit is a whole task rather than a trial. Relaxed relative to the per-task
# alpha: this is a single test over the panel (no multiplicity to guard) and a
# panel-wide solved-task gain is a weaker per-comparison signal than a per-task
# rate jump, so a more permissive bar is needed to detect real improvements.
AGGREGATE_PROMOTION_P_VALUE_ALPHA = 0.20


VerdictKind = Literal["improvement", "regression", "unchanged", "uncompared"]


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Single source of truth for "did the candidate beat the baseline on
    this task?". One comparison per task; the caller decides what counts
    as the baseline (typically pooled-control samples).

    Returned by `compare_candidate_against_baseline`. Consumed by the
    promotion gate and persisted experiment evidence.

    `kind` encodes the verdict:
    - "uncompared": candidate produced no trials.
    - "improvement": candidate beat the baseline by a margin significant at
      `alpha`. When the baseline has never been observed solving this task
      (`baseline_solved == 0`, covering both no-baseline frontier with empty
      baseline and tasks with trial history but no solves yet), the
      significance test is replaced by a majority-solve requirement on
      the candidate, since a single noisy solve against a never-solved
      baseline would otherwise read as an improvement on noise alone.
    - "regression": candidate underperformed the baseline at `alpha`.
      Never fires when `baseline_solved == 0` — there is no rate below 0%.
    - "unchanged": neither direction reached significance, or baseline is
      at the 0% floor and the candidate did not majority-solve.

    `p_value` is `None` when no statistical test was run (uncompared, or
    `baseline_solved == 0`).
    """

    kind: VerdictKind
    candidate_solved: int
    candidate_total: int
    baseline_solved: int
    baseline_total: int
    p_value: float | None

    @property
    def candidate_rate(self) -> float | None:
        if self.candidate_total <= 0:
            return None
        return self.candidate_solved / self.candidate_total

    @property
    def baseline_rate(self) -> float | None:
        if self.baseline_total <= 0:
            return None
        return self.baseline_solved / self.baseline_total


def compare_candidate_against_baseline(
    *,
    candidate_solved: int,
    candidate_total: int,
    baseline_solved: int,
    baseline_total: int,
    alpha: float = 0.05,
) -> BaselineComparison:
    """Compare a candidate's per-task trial counts against a baseline pool.

    Returns a `BaselineComparison` carrying the verdict and the numbers
    behind it. This is the only function in the codebase that answers
    "did the candidate beat the baseline?" — gate and evidence both route
    through it.

    Curriculum-frontier (no prior baseline samples for this task) is
    treated as the normal case: a candidate that majority-solves the task
    is an "improvement"; otherwise the comparison stays "unchanged".
    """
    if candidate_solved < 0 or candidate_total < 0:
        raise ValueError("candidate counts must be non-negative")
    if baseline_solved < 0 or baseline_total < 0:
        raise ValueError("baseline counts must be non-negative")
    if candidate_solved > candidate_total:
        raise ValueError("candidate_solved cannot exceed candidate_total")
    if baseline_solved > baseline_total:
        raise ValueError("baseline_solved cannot exceed baseline_total")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    if candidate_total == 0:
        return BaselineComparison(
            kind="uncompared",
            candidate_solved=candidate_solved,
            candidate_total=candidate_total,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            p_value=None,
        )

    if baseline_solved == 0:
        # Baseline has never solved this task. Covers two related cases:
        #   - baseline_total == 0: no-baseline frontier; no prior trials.
        #   - baseline_total >  0: task with trial history but no solves.
        # In both, a never-solved baseline cannot be regressed against and a
        # single candidate solve would otherwise read as improvement on noise
        # alone. Require a candidate majority-solve instead.
        kind: VerdictKind = (
            "improvement"
            if is_majority_solved(solved=candidate_solved, total=candidate_total)
            else "unchanged"
        )
        return BaselineComparison(
            kind=kind,
            candidate_solved=candidate_solved,
            candidate_total=candidate_total,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            p_value=None,
        )

    baseline_rate = baseline_solved / baseline_total
    candidate_rate = candidate_solved / candidate_total
    p_value = compute_fisher_exact_p_value(
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
    )
    if p_value >= alpha or candidate_rate == baseline_rate:
        kind = "unchanged"
    elif candidate_rate > baseline_rate:
        kind = "improvement"
    else:
        kind = "regression"
    return BaselineComparison(
        kind=kind,
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
        p_value=p_value,
    )
