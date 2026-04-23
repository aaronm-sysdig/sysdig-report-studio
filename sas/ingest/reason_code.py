"""Pure logic: given prior + current state, compute the reason_code for a CLOSED finding.

This is deliberately a pure function of a small named context so it can be
tested exhaustively without any DB dependency. The caller (finding_diff) is
responsible for assembling the context from DuckDB queries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ReasonCode = Literal["PATCHED", "RETIRED", "SCALED_TO_ZERO", "ACCEPTED",
                     "FEED_WITHDRAWN", "UNKNOWN"]


@dataclass(frozen=True)
class ReasonContext:
    risk_accepted_was: bool
    risk_accepted_is: bool
    newer_digest_exists_without_cve: bool
    image_still_runs_anywhere: bool
    cve_missing_from_feed: bool


def compute_reason_code(ctx: ReasonContext) -> ReasonCode:
    # Order matches spec §4.2.
    if not ctx.risk_accepted_was and ctx.risk_accepted_is:
        return "ACCEPTED"
    if ctx.newer_digest_exists_without_cve:
        return "PATCHED"
    if not ctx.image_still_runs_anywhere:
        return "RETIRED"
    if ctx.cve_missing_from_feed:
        return "FEED_WITHDRAWN"
    return "UNKNOWN"
