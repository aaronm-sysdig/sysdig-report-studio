# Phoenix HSL \- Pain point tracker \- April-2026 {#phoenix-hsl---pain-point-tracker---april-2026}

[Phoenix HSL \- Pain point tracker \- April-2026](#phoenix-hsl---pain-point-tracker---april-2026)

[Matt’s list of pain points:](#matt’s-list-of-pain-points:)

[Dan’s list](#dan’s-list)

[All notes](#all-notes)

[Appendix 1](#appendix-1)

[Problem / Context \- Today we store data for each finding in the table. We need a standard way so each product area can define the metrics they need, without reinventing the wheel each time.](#problem-/-context--)

[Core Concepts](#core-concepts)

[Standard (cannot be changed)](#standard-\(cannot-be-changed\))

[Configurable](#configurable)

[User Stories with Acceptance Criteria](#user-stories-with-acceptance-criteria)

[Example Queries](#example-queries)

[Out of Scope](#out-of-scope)

[User Roles / Permissions (if applicable)](#user-roles-/-permissions-\(if-applicable\))

[Design / UX](#design-/-ux)

[Pricing / Packaging](#pricing-/-packaging)

[Success Metrics](#success-metrics)

[Release Strategy](#release-strategy)

[Dependencies / Risks](#dependencies-/-risks)

[Supporting Docs](#supporting-docs)

## Matt’s list of pain points: {#matt’s-list-of-pain-points:}

Source: email 15-Apr-2026 \- [https://sysdigcloud.slack.com/archives/C06SXSBL61G/p1776257141815609](https://sysdigcloud.slack.com/archives/C06SXSBL61G/p1776257141815609)

| Headline | Details | Reference | Current Status | Notes | Competition? |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **(1) The Trend Graph Is Not a Measure of Remediation Progress** |  | Original Jira FR: [FR-1594](https://sysdig.atlassian.net/browse/FR-1594)[ProductBoard Note](https://sysdig.productboard.com/insights/all-notes?d=notes%2F53742937&insight=MTpOb3RlSW5zaWdodDphNTQyZWQ2NC1jYjRjLTRkNjctYWY0NS00ZWJmOTFjYzhhNTY%3D)  | Currently under investigation by product but no firm plans/commitment. | Side note \- Improvements to CSPM Exec reporting are planned for Q3 2026, reference \- [CSPM Executive Reports](https://sysdig.productboard.com/detail/MTpQbUVudGl0eTpmNzZlZjI1YS0wMWJiLTQ5YzgtOWIyZC05MzBmNGIzNjA1NTA=) What we ultimately need is something like this query but over a timeframe to show how an image is faring. `MATCH KubeWorkload AS k HAS Container RUNS Image AS i AFFECTED_BY Vulnerability AS v   WHERE v.hasFix = true AND v.severity IN ['High', 'Critical']   RETURN i.imageReference AS ImageRef, v.severity AS Severity, count(v) AS Vulnerabilities   ORDER BY ImageRef DESC, Severity DESC, Vulnerabilities DESC   LIMIT 10;`   ![][image1] | General consensus is Wiz and Upwind have better executive overview dashboards (but not clear on specifics around historical data retention \- will leave someone else to chime in). Wiz might be 12mo, Upwind might be 90d, unsure on Crowdstrike. |
| **2\. CSPM Cannot Be Trusted to Reflect Actual Coverage (Agent Health)** |  | Jira Product Item: [SSPROD-66078](https://sysdig.atlassian.net/browse/SSPROD-66078) Specific feedback on Karpenter scenario: [ProductBoard Feedback](https://sysdig.productboard.com/insights/all-notes?d=notes/54575430)Seems to relate to this existing product item on Agent Health / visibility:: [ProductBoard](https://sysdig.productboard.com/detail/0070e441-14a7-4d7e-b971-7de5c28e0da1)  | Looks like tentatively targeted for Q3 2026 but not committed (based on date in ProductBoard). *\[“Phoenix should have configured it correctly / but Sysdig should do a better job showing coverage”\]* |  | Wiz is more agentless focused,  Crowdstrike with EDR background seem to be stronger in this area. Upwind have “sensor health dashboards” but not clear how effective \- could not find much info on this. |
| **3\. No Seamless Path from a Runtime CVE to Its Source Image** |  | Potentially addressed by new ‘resource ownership’ functionality  [https://docs.sysdig.com/en/administration/resource-ownership/](https://docs.sysdig.com/en/administration/resource-ownership/)  | \[Next step \- Aaron to review further with Matt on next call if our new functionality addresses the concern / identify remaining gaps\] | Possibly related to recently released program owner flow / ownership information. | ? |
| **4\. The Jira Integration Is One-Way and Breaks Down with Ephemeral Workloads** |  | Nothing specific to two-way Jira sync yet (we could raise PB to verify if Product would entertain this).There are other Q1/Q2 roadmap items related to general improvements on Jira integrations however: Jira for Posture (planned Q1 2026\) [Jira for Posture](https://sysdig.productboard.com/detail/MTpQbUVudGl0eTo2MjYyOGU1My0yMzI1LTRiM2EtYjUyZi0yOGEzODEwMTU3NjU=) Jira for Runtime Events (planned for Q2 2026):[Jira Integration for Runtime Events](https://sysdig.productboard.com/detail/MTpQbUVudGl0eToxMGQ0YThlNC0zMzIwLTQzZGQtODMwZS1mYTU4N2I2MjM1ZGI=) | Nothing currently planned on two-way Jira sync. \[We don’t build this because every customer wants to do things differently, MCP could be explored for a more modern workflow perspective. “How do you explain the absence of something…”, should be managed on Jira side…\]. | PhoenixHSL hasn’t mentioned it before as they weren’t using it as it was broken. | For Wiz, Crowdstrike, Upwind \- looks like they all support one-way integration (support creation and updating of Jira tickets from issues), but doesn’t seem to be clear details (or existing features) on “two-way” sync. Gemini seems to point to a few ‘smoking guns’ for wiz to say that they do but its a little hazy from their public docs  |
| **5\. Alert Noise Is Eroding Team Confidence in the Platform** |  | Over the past 24 hours, they have had 19 high severity events.  14 of them are for malware detection.  I showed this to Matt and he agreed it was a known valid item.  I offered to filter it out…. Nothing. | Not product specific \- action would be to offer to work with them to tune alerts. \[Potentially good use case for MCP / triaging prior to direct integration. | Also a new concern… | Wiz/Crowdstrike/Upwind would have similar challenges on noise which rely on fine tuning. |
| **6\. No Correlation Between Pipeline and Runtime \- Shift-Left Is Broken in Practice** |  |  | Possibly not product specific \- need further clarity. \[This is probably a Aaron/Matt conversation.\]  | Not completely clear \- we should be able to do this but maybe they don’t have proper configuration in place. Probably needs more exploration. | ? |

Following our meeting today, please find below the structured feedback we committed to sharing with your management team. We want to be transparent: we are actively evaluating our options ahead of the October renewal, and the gaps outlined below are central to that decision. We have valued the relationship with your team, and that is precisely why we are being direct.

Our core concern is not any single feature. It is that after years of using Sysdig, we are still unable to produce meaningful reporting that demonstrates how the platform is working across our organisation, or whether our teams are genuinely improving their security posture over time. A security tool that cannot answer those questions is not delivering the value we are paying for.

*Comments:*

* *Matt asked about agents via dan, as he has not heard \- no definite answer,  ‘nothing actual and factual’*

**1\. The Trend Graph Is Not a Measure of Remediation Progress**

Sysdig's vulnerability trend chart tracks the count of vulnerability instances over time, but instance count is directly tied to how many container replicas are running at any given moment, not to whether vulnerabilities are actually being fixed. If a vulnerable image is deployed as 20 containers today, 200 tomorrow, and scaled back to 50 the day after, the graph moves dramatically in both directions without a single vulnerability being remediated. The underlying image and the vulnerability never changed.

This means the trend graph measures infrastructure scaling, not security improvement. It cannot tell us which specific images contain unresolved vulnerabilities, which teams are making progress, and which are not. There is no per-image or per-team longitudinal view that tracks whether a known vulnerability in a given image has been addressed across releases over time. We cannot use this data to hold teams accountable, to report to leadership, or to demonstrate that our security investment is having any effect. A meaningful remediation tracking capability would need to be anchored to the image, not the instance count.

*Refer appendix 1*

**2\. CSPM Cannot Be Trusted to Reflect Actual Coverage**

We recently discovered that our CSPM view was significantly understating the number of Managed Kubernetes nodes in our AWS environment. The Sysdig agent had not been deployed to a large portion of our nodes, and the platform gave us no indication that this was the case. We had no visibility into the gap until we identified it while investigating an incident, an Incident that should have been caught with the tool. If we cannot trust Sysdig to tell us what it is and is not covering, then the security posture it reports is unreliable. This is not an acceptable situation, as it is critical to its function.

*Comments:*

**3\. No Seamless Path from a Runtime CVE to Its Source Image**

When a CVE is identified at runtime, the critical question for remediation is: which image is this vulnerability coming from, and who owns it? Containers are ephemeral they are not the unit of remediation. The image is. Yet tracing a runtime CVE back to its source image is not a seamless workflow in Sysdig. The CVE detail view does not provide a direct, prominent link to the image or images responsible, which means engineers investigating an active vulnerability must navigate away, reconstruct context manually, and piece together the image-level picture themselves. In a time-sensitive situation, this is an operational failure. Remediation cannot begin until the correct image is identified and the owning team is notified and the platform makes that unnecessarily difficult.

*Comments:*

**4\. The Jira Integration Is One-Way and Breaks Down with Ephemeral Workloads**

The current Jira integration creates tickets from Sysdig findings, but this is where the integration ends. There is no two-way synchronisation, no lifecycle tracking, and no linkage between a ticket and the image-level vulnerability it was raised against. This creates a fundamental traceability problem, particularly for runtime findings.

When a Jira ticket is created for a runtime vulnerability, it reflects the state of a specific running container at that moment in time. As soon as that container is terminated and respawned which in a Kubernetes environment can happen continuously — the runtime context that generated the ticket no longer exists. The ticket becomes an orphan: there is no way to know from within Sysdig whether the underlying image was ever fixed, whether the ticket was resolved because the work was done or simply because the container was replaced, or whether the same vulnerability is still present in every new instance being deployed.

A meaningful integration would close this loop. Ticket status changes in Jira should be reflected back in Sysdig. Resolved tickets should be validated against whether the vulnerability still exists in the current version of the image. Open tickets should persist and remain linked to the image, not to the ephemeral runtime instance that first triggered them. Without this, the Jira integration provides the appearance of a remediation workflow while offering no actual traceability or assurance that anything has been fixed.

*Comments:*

**5\. Alert Noise Is Eroding Team Confidence in the Platform**

The volume of false positive alerts generated by default policies is causing alert fatigue which blocks us from integrating the Platform with our SOC MDR Service. When there are too many alerts the platform becomes unreliable. This is a well-documented issue among Sysdig customers and one that has persisted for us despite investment in tuning. The noise-to-signal ratio out of the box is not acceptable at this stage of the product's maturity.

*Comments:*

**6\. No Correlation Between Pipeline and Runtime \- Shift-Left Is Broken in Practice**

This is our most serious concern. Sysdig is positioned as a shift-left security platform, but we have no way to identify which of our pipelines have no Sysdig scanning configured at all, which have scanning enabled but without enforcement, or whether a vulnerability present at runtime was also visible at pipeline scan time and should have been blocked before deployment. Applications are reaching production with vulnerabilities that should have been caught and blocked during build or deploy stages, and Sysdig gives us no visibility into where we are failing. The shift-left model only works if the platform can close the loop between pipeline and runtime and currently it cannot.

Taken together, these gaps mean we cannot use Sysdig to produce the reporting and governance insights our security programme requires. We are not able to demonstrate to our leadership that the platform is working, that teams are improving, or that our investment is reducing real risk.

We look forward to understanding how your product team intends to address such concerns, and on what timeline. That response will be a significant factor in our renewal decision in October.

*Comments:*

## Dan’s list {#dan’s-list}

Source: Exec briefing here \- [https://docs.google.com/document/d/17qhDRVDg0maZSlwGYDCb5e0qPwyzp-7wLo1jk37cgPw/edit?tab=t.0](https://docs.google.com/document/d/17qhDRVDg0maZSlwGYDCb5e0qPwyzp-7wLo1jk37cgPw/edit?tab=t.0) Nov-2025):

| Headline | Details | Reference | Current Status | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Exec dashboards | Exec reports with historical time series on key metrics like vulns found, remediated, resurfaced. Visual dashboards on risks which are highest priority and why. | [FR-1594](https://sysdig.atlassian.net/browse/FR-1594) | Exec dashboards for CSPM planned for (Q2/Q3?) but not clear if includes historical data context (unlikely IMO): [CSPM Executive Reports](https://sysdig.productboard.com/detail/MTpQbUVudGl0eTpmNzZlZjI1YS0wMWJiLTQ5YzgtOWIyZC05MzBmNGIzNjA1NTA=) |  |
| Bulk risk acceptance | Bulk risk acceptance (currently restricted to single risk acceptance which is a usability issue when there are many risks to accept). | [FR-1704](https://sysdig.atlassian.net/browse/FR-1704) | Can’t see anything existing in ProductBoard so likely not planned, need to migrate FR to Customer Feedback to see if product will accept. |  |
| Jira integrations | JIRA integrations for not only VM but across VM, RTD and CSPM/Compliance/Risks with remediation capability if possible | There exists planned items in product board: Jira for Posture (planned Q1 2026\) [Jira for Posture](https://sysdig.productboard.com/detail/MTpQbUVudGl0eTo2MjYyOGU1My0yMzI1LTRiM2EtYjUyZi0yOGEzODEwMTU3NjU=) Jira for Runtime Events (planned for Q2 2026):[Jira Integration for Runtime Events](https://sysdig.productboard.com/detail/MTpQbUVudGl0eToxMGQ0YThlNC0zMzIwLTQzZGQtODMwZS1mYTU4N2I2MjM1ZGI=) |  |  |

**Insights**

# **All notes** {#all-notes}

Suraj Rajpal Christian Laffin this is the previous FR [https://sysdig.atlassian.net/browse/FR-1594](https://sysdig.atlassian.net/browse/FR-1594)

# 

# 

# Appendix 1 {#appendix-1}

Below is from [https://sysdig.productboard.com/insights/all-notes?d=MTpQbUVudGl0eTowZWU3NzE1Mi0yMTExLTQ0MGMtYTI5ZS0zNmUyNWQzNjA1ZTY%3D](https://sysdig.productboard.com/insights/all-notes?d=MTpQbUVudGl0eTowZWU3NzE1Mi0yMTExLTQ0MGMtYTI5ZS0zNmUyNWQzNjA1ZTY%3D)

# Problem / Context \-  {#problem-/-context--}

# Today we store data for each finding in the table. We need a standard way so each product area can define the metrics they need, without reinventing the wheel each time.

## Core Concepts {#core-concepts}

**Base Table (Raw Data):** A table designed to ingest high-volume, high-granularity raw event data. Data here is considered ephemeral and is kept for a short period. **This is already done**  
**Aggregation Tiers:** A series of progressively coarser-grained summary tables (e.g., daily, weekly, monthly) that are populated from the base table or the preceding tier. These tables are optimized for long-term storage and fast queries. Once an aggregation is done, it will not be updated, for example, if someone changes a zone, the output will not change for that zone scope.  
**Materialized Views:** The primary mechanism for automatically and incrementally populating the aggregation tiers as new data arrives in the base table.  
**Cardinality:** We must maintain some limited scope of cardinality within the database, it will not be possible to support a growing number of customers that have setup a large number of zones.

## Standard (cannot be changed) {#standard-(cannot-be-changed)}

**Max Retention per Interval (WIP)**

* Raw \- 32 Days  
* Daily Interval \- 90 days  
* Weekly Interval \- 53 weeks & WTD (Calculated Daily)  
* Monthly Interval \- 13 months & MTD (Calculated **Daily**)  
* Quarterly Interval \- 5 quarters & QTD (Calculated **Weekly**)  
* Yearly Interval \- Last Year & YTD (Calculated **Monthly(?)**)

**PreCalculate Metrics (WIP)**

* MOVA by supported metrics  
* MTTR by supported metrics

## Configurable {#configurable}

**Metric and Labels** (Examples)  
These are not set in stone, but just a starting point. Each of the product teams should help define which labels they need  
**Vulnerabilities**

* Vulns \- *Daily Interval* *Only* (Do we want to do this? this would mean nearly raw data, but removing certain columns)  
  * Resource  
  * Vuln id  
  * CVSS Score  
  * Package \+ Version  
  * Severity  
  * Zone (Limit to **10\***)  
  * In-use  
  * Exploitable  
  * Has Fix  
  * **Status**  
* Vuln Count by \- *All Intervals except for raw*  
  * Severity  
  * Zone (Limit to **10\***)  
  * In-use  
  * Exploitable  
  * Has Fix  
  * **Status**  
* Vuln MTTR/MOVA by \- *Weekly and longer intervals*  
  * Severity  
  * Zone (Limit to **10\***)  
  * In-use  
  * Exploitable  
  * Has Fix  
  * **Status**

﻿  
**Posture**

* Posture Findings \- *Daily Interval* *Only* (Do we want to do this? this would mean nearly raw data, but removing certain columns)  
  * Control Name  
  * Resource  
  * Severity  
  * Status  
  * Zone (Limit to **10\***)  
  * Platform (AWS, Azure)  
  * Resource Category (Compute, Storage, etc)  
* Posture Findings Count By \- All intervals except for raw  
  * Severity  
  * Status  
  * Zone (Limit to **10\***)  
  * Platform (AWS, Azure)  
  * Resource Category (Compute, Storage, etc)  
* Posture Finding MTTR/MOVA by \- *Weekly and longer intervals*  
  * Severity  
  * Status  
  * Zone (Limit to **10\***)  
  * Platform (AWS, Azure)  
  * Resource Category (Compute, Storage, etc)

﻿  
**Status** \- Does not exist for VM

\* zones \- 95% of customers have fewer than 20 zones, and 98% have fewer than 50 zones. Can we support 50 zones or more?

# **User Stories with Acceptance Criteria** {#user-stories-with-acceptance-criteria}

## Example Queries {#example-queries}

**Good Examples**  
**Vulns \- Daily Intervals Only**

* I want to know which **critical severities** that were **in-use** and **had an exploit** over the last 90 days  
* I want to know which **resources** had **CVE-2025-0001** over the last **90 days**

**Vulns Count By \- All Intervals except raw**

* I want to know the **count of each vuln severity** for my **zone production** over the last 1 year  
* I want to know which **zones** have the most open **critical vulns** over the last quarter and how we are doing so far this quarter  
  * Line chart with number of open critical vulns aggregated by monthly count, also support daily, weekly, quarterly or yearly

**Vuln MTTR/MOVA \- Intervals weekly or longer**

* I want to know if my **critical and high severity** vulns **MTTR** are lower or higher so far this year from last year  
  * MTTR (YTD): 10.5 Days  
    * Last Year: 15 Days  
* I want to know if my trending **critical and high** vulnerabilities **MOVA** is looking quarterly  
  * Line chart \- Critical and High vuln line showing last 3 quarters, and the current incomplete quarter

**Posture \- Daily Interval Only**

* I want to know all resources that failed control "Env Variable Exposing Secret" in the last 90 days

**Posture \- All Intervals except raw**

* I want to know the controls that had the most failing resources for the last 90 days

**Posture \- MTTR/MOVA \- Intervals weekly or longer**

* I want to know all

﻿

## **Out of Scope** {#out-of-scope}

Explicitly state exclusions to avoid scope creep.  
**Bad Example**

* I want to know the count of findings/resource on a single CVE there were for the entire year  
  * Instead, we can get this count only for the raw findings data, which is 32 days

## **User Roles / Permissions** (if applicable) {#user-roles-/-permissions-(if-applicable)}

How does this behave for different roles?

## **Design / UX** {#design-/-ux}

Link to prototype, mocks, or design doc.

## **Pricing / Packaging** {#pricing-/-packaging}

Which plan(s) does this belong to?

## **Success Metrics** {#success-metrics}

How will we measure success? New or updated telemetry/events?

## **Release Strategy** {#release-strategy}

preview release, or GA?

## **Dependencies / Risks** {#dependencies-/-risks}

Other initiatives, ingestion, or tech debt this relies on.
