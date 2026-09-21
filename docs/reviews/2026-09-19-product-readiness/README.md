# Product-readiness review — 2026-09-19

Two rounds of adversarial multi-agent review of `dsagent-v2` (branch `ui-harness`,
HEAD `8ac2758` + uncommitted docs), run on 2026-09-18/19.

- **Round 1** (5 agents, Opus 4.8) judged the code as a *local single-user dev tool*.
- **Round 2** (6 agents, Fable 5.1) re-judged everything against the **commercial bar**:
  deploy to the cloud and sell it as a hosted service (multi-tenant SaaS), and/or sell the
  harness + UI as a product customers self-host. Each agent was given an explicit method
  (tenancy isolation matrix, STRIDE per trust boundary, 12-factor + statefulness audit,
  durability inventory, "what must be true to charge money", 12-month premortem).

Nothing in the repo was modified by the reviewers. Every finding carries `file:line`
evidence; one finding (the sandbox) was **empirically verified** by a harmless probe.

## How to use this folder

| File | What it is | Use it for |
|---|---|---|
| [`00-consolidated-plan.md`](00-consolidated-plan.md) | The single deduplicated synthesis: verdict, strategic recommendation, blockers table with cross-agent convergence, what to keep, phased path. | Read first. Decide the three open decisions. |
| [`BACKLOG.md`](BACKLOG.md) | Every required change as a **task**, with id, phase, priority, effort, files, evidence, source agent(s) and dependencies. | Convert into ROADMAP items / tickets and work top-down. |
| `round2/*.md` | The six full commercial-bar reports, verbatim. | The evidence behind each backlog item (`file:line`, scenarios, target designs). |
| `round1/*.md` | The five full local-bar reports, verbatim, plus their P0–P3 synthesis. | Concrete bugs (several are regressions from the 2026-09-18 session) and UI/race root causes. |
| [`../../startup-readiness-review-2026-09-18.md`](../../startup-readiness-review-2026-09-18.md) | An earlier, independent stage-based assessment at the same commit. | Complementary framing (demo / pilot / paid local / SaaS go-no-go). |

## Status of the three decisions this review asks for

| Decision | Recommendation (round 2) | Status |
|---|---|---|
| Which business path first | **Self-host first, sold as a solution** ("DS/MMM team in a box", BYOK). SaaS only as Phase 2 after 2–3 paying self-host customers. Do not sell "domain-agnostic harness" until a second cartridge exists and the UI is driven from `/cartridges`. | **PENDING — owner** |
| License | **Open-core**: MIT harness on the existing public `datascience-agent` PyPI line; commercial license for `cartridges/ds`, the UI and enterprise features. Add `LICENSE` + generated `THIRD_PARTY_LICENSES`. All 258 deps are permissive (no AGPL/SSPL/BSL). | **PENDING — owner** |
| Start the "do now" batch (`N-*` in BACKLOG) | Yes — all are effort S, several are regressions from the 2026-09-18 session, and they close open wounds regardless of the path chosen. | **PENDING — owner** |

## Reading order for the impatient

1. `00-consolidated-plan.md` → "Veredicto" and "Bloqueantes innegociables".
2. `round2/03-saas-security.md` → boundary 3 (the sandbox) and the probe.
3. `round2/05-product-commercial.md` → part (b) "which path first" and part C (licenses).
4. `BACKLOG.md` → start at `N-01`.
