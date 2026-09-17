"use client";

import Link from "next/link";

import type { RunSummary } from "./lib/api";
import { duration, money, when } from "./lib/format";
import { useRuns } from "./lib/use-run";

/**
 * Home — every run this server knows about, newest first.
 *
 * A ledger rather than a wall of cards: an operator comparing this run with the
 * last one reads down a column, and the numbers that matter (duration, cost) are
 * the ones a stakeholder asks about first. Runs started from the CLI or from the
 * chat are in here too — the list is the run directory, not a browser's memory.
 */
export default function Home() {
  const { runs, error, loading } = useRuns();

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Runs</h1>
          <p className="page-sub">
            Every workflow this team has run, with what it produced and what it cost.
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

      {loading && <p className="dim">Reading the run directory…</p>}

      {!error && !loading && runs.length === 0 && <EmptyState />}

      {runs.length > 0 && (
        <table className="ledger">
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Progress</th>
              <th className="num">Started</th>
              <th className="num">Duration</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <RunRow key={run.run_id} run={run} />
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}

function RunRow({ run }: { run: RunSummary }) {
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
      <td className="num">{when(run.started_at)}</td>
      <td className="num">{duration(run.duration)}</td>
      <td className="num">{money(run.cost_usd)}</td>
    </tr>
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
