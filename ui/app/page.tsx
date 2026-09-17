"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import type { Catalogue, RunSummary, WorkflowShape } from "./lib/api";
import { getCatalogue } from "./lib/api";
import { duration, initial, money, when } from "./lib/format";
import { useRuns } from "./lib/use-run";

/**
 * Home — every run this server knows about, and what the team can do.
 *
 * Two tabs, because two people arrive here. An operator wants the ledger: a
 * run per row, newest first, with what it produced and what it cost, read down a
 * column. A stakeholder wants to know what this team *is* — which is the
 * Workflows tab, one card per workflow, with its personas, its steps and where
 * it will stop to ask them something.
 */
export default function Home() {
  const { runs, error, loading } = useRuns();
  const catalogue = useCatalogue();
  const [tab, setTab] = useState<"runs" | "workflows">("runs");

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Runs</h1>
          <p className="page-sub">
            Every run is a report the team writes in front of you. Open one to watch it fill,
            approve the decisions, and take the result away.
          </p>
        </div>
        <Link href="/new" className="btn btn-primary">
          New run
        </Link>
      </div>

      {error && (
        <div className="notice">
          Can’t reach the agent server: {error}. Start it with <code>dsagent serve</code>.
        </div>
      )}

      <nav className="tabs">
        <button
          type="button"
          className={tab === "runs" ? "is-on" : undefined}
          onClick={() => setTab("runs")}
        >
          Runs{runs.length ? ` · ${runs.length}` : ""}
        </button>
        <button
          type="button"
          className={tab === "workflows" ? "is-on" : undefined}
          onClick={() => setTab("workflows")}
        >
          What the team can do
        </button>
      </nav>

      {tab === "workflows" ? (
        <Workflows catalogue={catalogue} />
      ) : (
        <>
          {loading && <p className="dim">Reading the run directory…</p>}
          {!error && !loading && runs.length === 0 && <EmptyState />}
          {runs.length > 0 && (
            <table className="ledger">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Team</th>
                  <th className="num">Started</th>
                  <th className="num">Duration</th>
                  <th className="num">Cost</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <RunRow key={run.run_id} run={run} shape={shapeOf(catalogue, run.workflow)} />
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </main>
  );
}

function RunRow({ run, shape }: { run: RunSummary; shape: WorkflowShape | null }) {
  const done = run.steps_total ? run.steps_done / run.steps_total : 0;
  return (
    <tr>
      <td>
        <Link href={`/runs/${encodeURIComponent(run.run_id)}`} className="run-link">
          {run.workflow}
        </Link>
        <div className="run-dataset">{dataset(run) ?? run.run_id}</div>
      </td>
      <td>
        <StatusLabel run={run} />
      </td>
      <td>
        <div className={`bar is-${run.status}`} title={`${run.steps_done} of ${run.steps_total} steps`}>
          <i style={{ width: `${Math.round(done * 100)}%` }} />
        </div>
        <div className="run-dataset">
          {run.steps_done}/{run.steps_total} steps
        </div>
      </td>
      <td>
        <Avatars personas={shape?.personas ?? []} />
      </td>
      <td className="num">{when(run.started_at)}</td>
      <td className="num">{duration(run.duration)}</td>
      <td className="num">{money(run.cost_usd)}</td>
    </tr>
  );
}

/** Who worked on it, as initials — the team made visible at a glance. */
function Avatars({ personas }: { personas: string[] }) {
  if (personas.length === 0) return <span className="dim">—</span>;
  return (
    <span className="avatars">
      {personas.map((p) => (
        <span key={p} className="avatar" title={p}>
          {initial(p)}
        </span>
      ))}
    </span>
  );
}

/**
 * What the team can do, one card per workflow.
 *
 * Read off the loaded cartridge, so a second cartridge with other personas and
 * other steps renders without a line changing here — invariant 1, applied to a
 * screen. The gates are called out because they are the promise this product
 * makes that the others do not: it will stop and ask before it goes on.
 */
function Workflows({ catalogue }: { catalogue: Catalogue | null }) {
  if (!catalogue) return <p className="dim">Reading the cartridge…</p>;
  return (
    <>
      <div className="wf-grid">
        {catalogue.workflows.map((wf) => (
          <article key={wf.name} className="wf-card">
            <h2 className="wf-name">{wf.name}</h2>
            <p className="wf-desc">{wf.description}</p>
            <p className="wf-steps">
              {wf.steps.map((step) => (
                <span key={step.id} className="wf-chip">
                  <span className="avatar is-tiny">{initial(step.persona)}</span>
                  {step.id}
                  {step.gate && <span className="wf-chip-gate">asks you</span>}
                </span>
              ))}
            </p>
            <p className="wf-foot mono">
              delivers {wf.steps.at(-1)?.produces.join(", ") || "—"} · env {wf.env}
            </p>
          </article>
        ))}
      </div>
      <p className="wf-note">
        {catalogue.cartridges.map((c) => `${c.name} v${c.version}`).join(", ")} ·{" "}
        {[...new Set(catalogue.workflows.flatMap((w) => w.personas))].join(", ")}. Each persona
        sees only the skills the cartridge grants it.
      </p>
    </>
  );
}

/**
 * The status a person would say out loud.
 *
 * `run.json` records what a run was doing, not whether anyone is still doing it,
 * so a run left `running` by a server that was killed reads as "interrupted"
 * rather than spinning forever — the API's `live` is what separates the two.
 */
function StatusLabel({ run }: { run: RunSummary }) {
  if (run.status === "running" && !run.live) {
    return (
      <span className="status is-stale" title="No process is driving this run">
        interrupted
      </span>
    );
  }
  const label =
    run.status === "awaiting_gate"
      ? run.awaiting
        ? "waiting for you"
        : "paused"
      : run.status;
  return <span className={`status is-${run.status}`}>{label}</span>;
}

/**
 * The dataset a run was given, without the harness knowing what a dataset is.
 *
 * Any input whose value looks like a workspace path is the file this run worked
 * on; the name of that input belongs to the cartridge, not here (invariant 1).
 */
function dataset(run: RunSummary): string | null {
  const path = Object.values(run.inputs ?? {}).find(
    (value) => typeof value === "string" && value.startsWith("data/"),
  );
  return path ? path.slice("data/".length) : null;
}

function shapeOf(catalogue: Catalogue | null, workflow: string): WorkflowShape | null {
  return catalogue?.workflows.find((w) => w.name === workflow) ?? null;
}

function useCatalogue(): Catalogue | null {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  useEffect(() => {
    let live = true;
    getCatalogue()
      .then((c) => live && setCatalogue(c))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);
  return catalogue;
}

function EmptyState() {
  return (
    <div className="empty">
      <h2>Nothing has run yet</h2>
      <p>
        A run is one workflow from end to end: a team of personas working through declared
        steps, stopping at a gate when it needs a decision from you, and leaving a report and
        every file it wrote behind.
      </p>
      <Link href="/new" className="btn btn-primary">
        Start the first run
      </Link>
    </div>
  );
}
