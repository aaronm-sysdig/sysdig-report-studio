# Adversarial Brainstorm — Historical Vuln Analytics

**Date:** 2026-04-23
**Purpose:** Stress-test the Historical Vuln Analytics 5-tuple query primitive against questions we haven't thought of, by dispatching four parallel Claude agents playing adversarial roles. Goal: surface unknown-unknowns *before* the spec is written.

**Context:** Phoenix HSL (Sysdig customer) is unhappy that Sysdig's vulnerability trend graph counts container instances rather than unique image-level findings. Competitors (Wiz, Upwind, CrowdStrike) are being evaluated. This sub-project of sysdig-report-studio is a single-customer proof-of-concept historical analytics tool.

**How to reuse this document:** If you need further adversarial input later ("you answered this last time, now we want X"), reference this file when dispatching a new agent so it doesn't repeat the same 15 questions. Personas, prompts, and raw outputs are all preserved verbatim below.

---

## Personas used

Four parallel agents (subagent_type: general-purpose), each given ~600-word adversarial prompts. All returned 15 questions each (60 total).

1. **Wiz SE** — pitches ~12mo history + exec dashboards + image lineage
2. **Upwind SE** — pitches ~90d retention + runtime correlation + sensor coverage
3. **CrowdStrike SE** — pitches EDR heritage + workload telemetry + KEV/ransomware
4. **Skeptical auditor / FAIR-risk practitioner** — catches gaming, regression, data quality issues

All personas instructed: focus on question *shape*, not vendor UI; don't fabricate competitor features; include adversarial specifics (numbers, dimensions, time windows).

---

## Synthesis: 12 themes (derived from the 60 questions)

| # | Theme | 5-tuple coverage | Notes |
|---|---|---|---|
| 1 | Trend of unique CVEs over time (not container counts) | Covered | Core case |
| 2 | Image lineage / tag chronology / digest-under-tag churn | Covered (needs Repository node) | Already planned |
| 3 | Team accountability / ownership | **Gap** — needs Team/Owner node | Design change |
| 4 | MTTR decomposition (patch vs retire vs accept) | **Gap** — needs reason codes on state log | Design change |
| 5 | Fast CVE blast-radius lookup | Covered | Indexing concern |
| 6 | KEV / exploit-in-the-wild prioritization | Covered | CSV already has KEV fields |
| 7 | Coverage assurance (what are we NOT seeing) | **Out of scope v1** | Customer-agreed — we can only chart what we have |
| 8 | Dev→Staging→Prod drift | Covered via namespace/cluster labels | Good enough for v1 |
| 9 | In-use vs on-disk | Covered | `Package In Use` column |
| 10 | Runtime/network/lateral-movement joins | **Out of scope v1** | Not in CSV; future |
| 11 | Gaming detection | Covered | Falls out of theme 4 reason codes |
| 12 | Data quality / scan freshness | Partial | `last_scan_seen_per_workload` is a heuristic |

---

## Three "tier-1 wow widgets" identified

These fell out of the synthesis as showstopper demos worth shipping on day one:

1. **CVE Blast Radius Timeline** — pick a CVE, see Gantt-style timeline across every image/workload/team. Kills Wiz-6, CS-6, CS-12.
2. **Team Accountability Leaderboard with Gaming Detection** — ranks teams by MTTR, drill-in shows decomposition (real patches vs retirements vs acceptances on dead images). Flags suspicious patterns. Kills the entire auditor persona.
3. **Tag Lineage View** — pick a repository, every tag with stacked-bar CVE counts, hover for per-tag diff (+3 new criticals, −7 fixed, 2 regressed). Kills Wiz-2, CS-3, Aud-9.

---

## Raw outputs — verbatim agent responses

### Wiz SE (15 questions)

1. Over the trailing 12 months, how many critical CVEs did Phoenix actually remediate versus how many simply aged out because the vulnerable image stopped running — and can you prove the difference on a single chart for Dan's board deck?
2. For each of your top 25 production images, what is the month-over-month trajectory of critical and high CVE counts across every tag lineage (latest, v1.x, v2.x), so Matt can see whether new builds are getting safer or worse?
3. Which 10 engineering teams introduced the most net-new critical vulnerabilities in Q1, ranked by unique CVE-image pairs, and what is each team's 90-day trend line?
4. What is your true image-level MTTR for KEV-listed CVEs over the last two quarters, measured from CVE disclosure date to the first pushed image digest that no longer contains it?
5. Can you show a board-ready slide that answers "are we more secure today than 6 months ago?" using a metric that is not distorted by replica count, autoscaling, or cluster churn?
6. When a critical CVE like a new OpenSSL or Log4j-class issue drops tomorrow morning, can you produce within 15 minutes a historical list of every image digest, tag, and cluster that has ever been exposed to it across the last year?
7. Which vulnerabilities that you previously reported as "fixed" have regressed back into a production image in the last 180 days, and which team owns each regression?
8. For Dan's quarterly board pack, can you produce a single time-series that shows KEV-exploitable criticals in runtime, trended weekly, with a clear remediation-vs-drift decomposition?
9. What percentage of your registry was actually scanned each week for the last 52 weeks, and can you prove no coverage gaps opened up during cluster migrations or pipeline changes?
10. For any given base image (say your golden Node 20 image), can you show the downstream blast radius — how many child images inherited it, which teams own them, and how CVE counts evolved tag-by-tag over the last year?
11. If the board asks "what is our critical-CVE backlog today versus 12 months ago, normalized per 100 running images," can you answer that from your current tooling in under 60 seconds?
12. Which images have the worst "time-to-patch" SLO compliance rate over the trailing 6 months, broken down by team, severity, and whether the CVE was KEV or EPSS>0.5?
13. When Matt needs to tell a product VP "your team's images have been carrying these 14 criticals for more than 90 days," can your current platform generate that artifact with historical evidence, or only a point-in-time snapshot?
14. For the last 4 quarters, what is the remediation velocity delta between teams — i.e., who is accelerating, who is regressing — and can you show it on the same dashboard Dan would present to the audit committee?
15. If a regulator or the board asks "prove that the critical CVE we flagged in the January review is no longer present anywhere in our estate, and has not reappeared since," can you produce a dated, auditable historical record image-by-image, or only tell them what is running right now?

### Upwind SE (15 questions)

1. Over the last 90 days, what percentage of critical CVEs observed at runtime in production were NOT flagged by your build-time or registry scans, and can you produce that list by workload and image digest?
2. For your top 20 most-deployed images, can Sysdig show a side-by-side comparison of CVEs present at build time versus CVEs actually loaded into memory at runtime on a live pod, over a rolling 30-day window?
3. Right now, across all your Kubernetes clusters and cloud accounts, which specific nodes, namespaces, or VPCs do NOT have a reporting Sysdig agent, and how long has each of those gaps existed?
4. When you had your last security incident, how quickly could you answer the question "was this host covered by our runtime sensor at the moment of compromise, yes or no, with a timestamp"?
5. Can you pull a single report showing every workload that ran in production in the last 60 days where the agent stopped reporting for more than 15 minutes, and what was running during that blind window?
6. For a given critical CVE disclosed this quarter, can you show the exact lineage of every image tag and digest that carried it, from first appearance in a dev pipeline through to the last production pod that executed it?
7. How many workloads started running a new image tag in the last 7 days that introduced critical CVEs not present in the previously deployed tag for that same workload?
8. What is your true MTTR for critical runtime-exposed vulnerabilities, measured from first runtime observation to last runtime observation, broken down by business unit over the last two quarters?
9. Can you show CVE drift between dev, staging, and prod for the same logical service, specifically which criticals exist in prod today that have already been fixed in dev but haven't rolled forward?
10. Over the last 90 days, which unique vulnerabilities were actually remediated versus which ones merely disappeared from the dashboard because replica counts dropped or pods were rescheduled?
11. For each internet-facing workload, can you list the runtime-loaded packages with known exploits in the wild and correlate that to which of those packages have observed network egress or inbound connections in the last 14 days?
12. What percentage of your cloud compute footprint, including serverless, Fargate tasks, and ephemeral CI runners, is covered by a runtime sensor today, and what percentage was covered 30, 60, and 90 days ago?
13. If an auditor asked you tomorrow to prove continuous runtime monitoring coverage for PCI-scoped workloads over the last 12 months, what is the longest coverage gap you would have to disclose, and can you produce that evidence in under an hour?
14. When a new image is promoted to production, can you automatically detect and alert within minutes if it regresses on vulnerability posture compared to the image it replaced, or do you only find out at the next scheduled scan?
15. For the last five production incidents involving a compromised workload, can Sysdig retrospectively show which other workloads in the same cluster shared the vulnerable image or library and had network reachability to the compromised pod during the incident window?

### CrowdStrike SE (15 questions)

1. Of every critical CVE currently running in your production clusters, how many carry a CISA KEV flag with a known-ransomware association, and can you pull that list — by image, by workload, by owning team — in under five minutes right now?
2. How many of those KEV-flagged, ransomware-associated criticals were already present in your environment 30, 60, and 90 days ago, and what is your mean-time-to-remediate specifically for that subset versus your overall CVE population?
3. For the vulnerabilities you patched last quarter, how many have since regressed — meaning the same CVE reappeared in a redeployed image or a new workload — and which teams are the repeat offenders?
4. Can you show me, as a single joined view, every internet-exposed workload that is also running a process with a KEV-listed exploitable CVE and also has an over-privileged service account attached? How long does assembling that answer take today?
5. What percentage of your production hosts and Kubernetes nodes are confirmed covered by your current runtime sensor, and how do you prove the negative — the unprotected 2 to 5 percent — to your auditors and your board?
6. When a new CVE like a Log4Shell-class issue drops tomorrow morning, what is the exact number of minutes from disclosure to you having a confirmed list of affected running workloads, their owners, and their external exposure status?
7. Your current trend line reportedly counts container instances rather than unique vulnerabilities — so when your exec team sees the graph go down, is that because you fixed vulnerabilities or because pods were rescheduled? How do you explain that distinction to your CISO?
8. Of the criticals flagged in the last 30 days, how many are on workloads that have east-west network reachability to your crown-jewel data stores, and how is that lateral-movement blast radius being tracked over time?
9. How do you currently correlate a vulnerability finding on a workload with the identity and secrets that workload can access — cloud IAM roles, Kubernetes service accounts, mounted secrets — to rank true business risk rather than raw CVSS?
10. For each of your top ten engineering teams, what is their individual remediation SLA compliance rate for KEV-flagged CVEs over the last two quarters, and can you produce that leaderboard for your next exec review without a manual CSV exercise?
11. Can you show me a historical view of your attack surface — how many externally exposed workloads with exploitable, known-ransomware CVEs you had on any given day over the past six months — and is that graph trending up or down?
12. When an active threat actor is reported exploiting a specific CVE in the wild, how do you go from that intel to a prioritized list of your affected workloads, their runtime behavior, and a response playbook — and how many tools does that workflow touch?
13. Of the vulnerabilities your scanner reports as critical today, what percentage are actually loaded into a running process versus merely present on disk in the image, and how does that change your true remediation backlog?
14. How many workloads in production are running images that have not been rescanned in the last 30 days, and what is your confidence interval that your current CVE counts reflect reality rather than stale scan data?
15. If your board asked tomorrow: "What is our residual ransomware exposure from unpatched workloads this quarter, trending against last quarter?" — can you answer with a single number and a defensible methodology, or does it take a week and three analysts?

### Skeptical auditor (15 questions)

1. Of all CVEs marked "fixed" in the last 90 days, what percentage were resolved by an actual image rebuild with a newer base layer digest versus the image simply disappearing from the inventory — and can you show me the two populations side-by-side by team?
2. Show me every team whose MTTR improved by more than 25% quarter-over-quarter, and decompose that improvement into: (a) patches applied, (b) images retired, (c) workloads scaled to zero, (d) CVEs reclassified or suppressed. If (b)+(c)+(d) exceeds (a), flag the team.
3. Which CVEs vanished from the trending dashboard in the last 30 days despite the underlying image digest, tag, and running workload count remaining unchanged — and who or what process caused the disappearance?
4. How many production images have not been rescanned in more than 14 days, broken down by team and by whether they carry known KEV-listed CVEs — i.e. what is our "quiet risk" surface?
5. Of CVEs currently in "risk accepted" state, how many exist in images that are no longer deployed to any running workload, meaning the acceptance is cosmetic and inflates the "managed" count?
6. What fraction of our declared image inventory has zero scan records in the last 30 days, and does that fraction correlate with teams whose remediation KPIs suddenly improved?
7. For each team, what is the ratio of "fixed via upgrade" to "fixed via deletion/retirement" over the trailing 90 days, and has that ratio shifted suspiciously right before quarterly review cycles?
8. Which CVEs have a public exploit (KEV, EPSS >0.7, or weaponized PoC) and have been open longer than 30 days — and of those, how many are in images owned by teams currently reporting green SLA compliance?
9. Show me images where the tag (e.g. `:v1.4.2`) is unchanged but the digest has changed more than 3 times in 30 days — are we silently rebuilding without version discipline, and are CVE counts oscillating as a result?
10. How many CVEs were "closed" because the scanner's vulnerability feed reclassified or withdrew them, versus closed by actual remediation action — and is any team disproportionately benefiting from feed churn?
11. For every workload in production, can you prove a scan exists within the last 7 days tied to the exact running digest — not a sibling tag, not a parent image, the actual running digest? What percentage fails this check?
12. Which teams show a pattern of CVEs being marked "will not fix" or "accepted" within 48 hours of SLA breach thresholds — i.e. acceptance used as an SLA escape hatch?
13. How many critical CVEs were reintroduced in the last 90 days by a new deployment of an image that had previously been remediated — and are any teams repeatedly regressing on the same CVE IDs?
14. What is the delta between CVEs the scanner reports for an image and CVEs reported by an independent second source (OS vendor advisories, GitHub Advisory DB) for the same package versions — where is our tool systematically under-reporting?
15. Of all "remediated" CVEs in the last quarter, how many reappeared within 30 days on the same image lineage, suggesting the fix was reverted, a base image rollback occurred, or the closure was premature?
