"""The filter mechanism (Fletcher & Leyffer 2002; Def. 9 of the paper), with
the infeasibility guard the old ``utils/filter.py`` was missing.

Bug being fixed (paper's Remark 7 / Lemma 4): the old ``FilterSQP.add_to_filter``
would insert *any* accepted (f, vartheta) pair, including vartheta == 0 (a
feasible point, or one that only looks infeasible due to model error). Once a
(0, f) pair is in the filter, acceptance test (7) permanently blocks every
future point with f >= that f, since vartheta <= (1-gamma)*0 is never true
for any later infeasible point and the only way through is the objective
disjunct -- effectively capping the achievable objective forever and
deadlocking the method at a feasible, non-optimal point. Lemma 4 requires
every filter entry to have vartheta_i > 0; this is enforced structurally
here (add() silently drops vartheta <= 0 pairs) rather than left to caller
discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FilterEntry:
    vartheta: float
    f: float


class Filter:
    def __init__(self, gamma_vartheta: float):
        if not (0.0 < gamma_vartheta < 1.0):
            raise ValueError(f"gamma_vartheta must be in (0,1), got {gamma_vartheta}")
        self.gamma_vartheta = gamma_vartheta
        self.entries: List[FilterEntry] = []

    def is_acceptable(self, f: float, vartheta: float, extra=()) -> bool:
        """Acceptance test (7) against the stored filter plus any extra
        (vartheta_i, f_i) pairs supplied for this call only (e.g. Restore's
        F cup {(vartheta_k, f_k)}, which is not necessarily written back)."""
        g = self.gamma_vartheta
        for e in self.entries:
            if not (vartheta <= (1 - g) * e.vartheta or f <= e.f - g * vartheta):
                return False
        for vt, fv in extra:
            if vt <= 0:
                continue
            if not (vartheta <= (1 - g) * vt or f <= fv - g * vartheta):
                return False
        return True

    def add(self, f: float, vartheta: float) -> bool:
        """Add (f, vartheta) to the filter. Returns whether it was actually
        added (guard: vartheta <= 0 is never added -- Lemma 4)."""
        if vartheta <= 0:
            return False
        self.entries.append(FilterEntry(vartheta, f))
        # Housekeeping only (not required for the finite-area proof, Lemma
        # 15): drop entries the new pair component-wise dominates, to keep
        # the filter small.
        kept = [e for e in self.entries[:-1] if not (vartheta <= e.vartheta and f <= e.f)]
        kept.append(self.entries[-1])
        self.entries = kept
        return True
