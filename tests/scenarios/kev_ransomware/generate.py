"""Generate two-day fixture for KEV ransomware denormalisation scenario.

Day 1: one ransomware-flagged CVE, one non-ransomware CVE, same image.
Day 2: both CVEs still open (reseen path exercises the UPDATE branch).
"""
from pathlib import Path

from tests.scenarios._builder import ScenarioBuilder

OUT = Path(__file__).parent


def main() -> None:
    b = ScenarioBuilder()
    # Ransomware-flagged finding
    b.add_finding(
        vulnerability_name="CVE-2024-RANSOM-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="true",
        vulnerability_severity="Critical",
        cvss_score=9.8,
    )
    # Non-ransomware finding on same image
    b.add_finding(
        vulnerability_name="CVE-2024-NORMAL-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="false",
        vulnerability_severity="High",
        cvss_score=7.5,
    )
    b.write_csv(OUT / "day1_2026-05-01.csv")

    # Day 2 — same findings reseen
    b.clear()
    b.add_finding(
        vulnerability_name="CVE-2024-RANSOM-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="true",
        vulnerability_severity="Critical",
        cvss_score=9.8,
    )
    b.add_finding(
        vulnerability_name="CVE-2024-NORMAL-1",
        image_id="sha256:kev001ransom",
        image_name="registry.example.com/app:v1.0",
        cisa_kev_known_ransomware="false",
        vulnerability_severity="High",
        cvss_score=7.5,
    )
    b.write_csv(OUT / "day2_2026-05-02.csv")
    print("Generated day1_2026-05-01.csv and day2_2026-05-02.csv")


if __name__ == "__main__":
    main()
