from sas.ingest.reason_code import compute_reason_code, ReasonContext


def test_risk_accepted_flip_is_accepted():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=True,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "ACCEPTED"


def test_newer_digest_without_cve_is_patched():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=True,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "PATCHED"


def test_image_not_running_anywhere_is_retired():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=False,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "RETIRED"


def test_cve_missing_from_feed_is_feed_withdrawn():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=True,
    )
    assert compute_reason_code(ctx) == "FEED_WITHDRAWN"


def test_none_of_the_above_is_unknown():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=False,
        newer_digest_exists_without_cve=False,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "UNKNOWN"


def test_accepted_takes_precedence_over_patched():
    ctx = ReasonContext(
        risk_accepted_was=False,
        risk_accepted_is=True,
        newer_digest_exists_without_cve=True,
        image_still_runs_anywhere=True,
        cve_missing_from_feed=False,
    )
    assert compute_reason_code(ctx) == "ACCEPTED"
