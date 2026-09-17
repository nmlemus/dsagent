"use client";

import { useEffect, useState } from "react";

import { API, type RunDetail } from "../lib/api";
import { duration, money, percent, tokens } from "../lib/format";
import type { RunView } from "../lib/run-state";

/**
 * A finished run is a deliverable, not a log with a green tick on it.
 *
 * So the top of the document changes when the run ends: the numbers a
 * stakeholder asks about, a link anyone can open, the ways to take it away, and
 * the report's own opening paragraph lifted to the top — written by the persona
 * who wrote the report, not by this component.
 */
export function Finished({
  runId,
  detail,
  view,
  onReplay,
  replaying,
}: {
  runId: string;
  detail: RunDetail;
  view: RunView;
  onReplay: () => void;
  replaying: boolean;
}) {
  const usage = detail.usage ?? {};
  const cached = (usage.cache_read ?? 0) + (usage.cache_creation ?? 0);
  const figures = view.cards.filter((c) => c.kind === "chart").length;
  const files = view.files.filter((f) => f.kind === "deliverable").length;

  return (
    <div className="done">
      <dl className="done-figures">
        <Figure label="took" value={duration(detail.duration)} />
        <Figure
          label={usage.input_tokens ? `cost · ${percent(cached, usage.input_tokens)} cached` : "cost"}
          value={money(detail.cost_usd)}
        />
        <Figure label="deliverables + charts" value={`${files} + ${figures}`} />
        <Figure
          label={detail.gate_wait ? "waited for you" : "your decisions"}
          value={detail.gate_wait ? duration(detail.gate_wait) : "0"}
          title={`${tokens((usage.input_tokens ?? 0) + (usage.output_tokens ?? 0))} tokens`}
        />
      </dl>

      <p className="done-actions">
        <Share />
        <a className="btn" href={`${API}/runs/${encodeURIComponent(runId)}/export.html`}
           target="_blank" rel="noreferrer">
          Export report
        </a>
        <a className="btn" href={`${API}/runs/${encodeURIComponent(runId)}/download`}>
          Download artifacts
        </a>
        <a className="btn" href={`${API}/runs/${encodeURIComponent(runId)}/events.json`}
           target="_blank" rel="noreferrer">
          Run log
        </a>
        <button type="button" className="btn" onClick={onReplay} disabled={replaying}>
          {replaying ? "Replaying…" : "Replay this run"}
        </button>
      </p>

      <Summary runId={runId} view={view} />
      <Versions runId={runId} detail={detail} />
    </div>
  );
}

/**
 * The other times this question was asked of this file.
 *
 * A run is not a one-off: the same workflow on the same dataset, run again, is a
 * *version* of the same document, and knowing there are three of them — and
 * which one you are reading — is most of what "versioned" has to mean. Comparing
 * two of them is the part that is deferred; see the log.
 */
function Versions({ runId, detail }: { runId: string; detail: RunDetail }) {
  const [siblings, setSiblings] = useState<{ run_id: string; started_at: number | null }[]>([]);
  const data = detail.inputs?.data_path;

  useEffect(() => {
    let live = true;
    fetch(`${API}/runs`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((body: { runs: (RunDetail & { inputs: Record<string, string> })[] }) => {
        if (!live) return;
        setSiblings(
          body.runs
            .filter(
              (r) =>
                r.workflow === detail.workflow &&
                r.inputs?.data_path === data &&
                r.status === "done",
            )
            .sort((a, b) => (a.started_at ?? 0) - (b.started_at ?? 0))
            .map((r) => ({ run_id: r.run_id, started_at: r.started_at })),
        );
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [detail.workflow, data]);

  if (siblings.length < 2) return null;
  return (
    <p className="done-versions">
      <span className="done-eyebrow">Versions of this</span>
      {siblings.map((run, i) => (
        <a
          key={run.run_id}
          href={`/runs/${encodeURIComponent(run.run_id)}`}
          className={run.run_id === runId ? "is-here" : undefined}
          title={run.run_id}
        >
          v{i + 1}
        </a>
      ))}
      <span className="dim">same workflow, same dataset</span>
    </p>
  );
}

/**
 * Copy the link.
 *
 * The URL *is* the share: everything on this screen is rebuilt from the run's
 * own log, so anyone who can reach this server sees the same document. There is
 * no second, published copy to drift out of date — and no permission model
 * either, which is stated here rather than implied.
 */
function Share() {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="btn btn-primary"
      title="Anyone who can reach this server sees the same document"
      onClick={() => {
        void navigator.clipboard?.writeText(window.location.href).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        });
      }}
    >
      {copied ? "Link copied" : "Copy link"}
    </button>
  );
}

/**
 * The report's own opening, at the top of the document.
 *
 * Lifted, not written: the last step's markdown, up to its second heading. If a
 * report does not open with its point, this shows that rather than inventing
 * one — the fix belongs in the `reports` skill, not here.
 */
function Summary({ runId, view }: { runId: string; view: RunView }) {
  const last = view.steps.at(-1);
  const path = last ? Object.values(last.matched).flat().find((p) => p.endsWith(".md")) : null;
  const [text, setText] = useState<string | null>(null);

  useEffect(() => {
    if (!path) return;
    let live = true;
    fetch(`${API}/runs/${encodeURIComponent(runId)}/files/${segments(path)}`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((body) => live && setText(opening(body)))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [runId, path]);

  if (!text) return null;
  return (
    <div className="done-summary">
      <p className="done-eyebrow">In one paragraph</p>
      <p>{text}</p>
    </div>
  );
}

/** A workspace path, encoded segment by segment — the slashes are structure. */
function segments(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

/**
 * The report's first actual prose, wherever it starts.
 *
 * Not "everything under the first heading": a report that opens
 * `# Title` / `## Headline findings` has nothing under its first heading, and
 * the summary would be blank — which is what happened. So: skip headings and
 * blank lines, then take the block that follows.
 */
function opening(markdown: string): string {
  const lines = markdown.split("\n");
  const body: string[] = [];
  for (const line of lines) {
    const heading = /^#{1,6} /.test(line);
    if (body.length === 0 && (heading || line.trim() === "")) continue;
    if (heading) break;
    if (line.trim() === "" && body.length > 0) break;
    body.push(line);
  }
  const prose = body
    .join(" ")
    .replace(/[*_`>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (prose.length <= LIMIT) return prose;
  // On a word, with an ellipsis that says there is more — not mid-syllable.
  // Numbered items keep their numbers: "1." is how the report enumerates its
  // findings, and stripping them ran three of them into one sentence.
  const cut = prose.lastIndexOf(" ", LIMIT);
  return `${prose.slice(0, cut > 0 ? cut : LIMIT).trimEnd()}…`;
}

const LIMIT = 600;

/**
 * The run, scrubbed.
 *
 * The auditable-run story, and the best demo tool this product has: the same
 * screen, rebuilt from the same log, at any moment of the run. It is not a
 * second implementation of the document — the reducer that draws the live screen
 * is the one that draws this.
 */
export function Scrubber({
  span,
  at,
  onScrub,
  onLive,
}: {
  span: [number, number];
  at: number | null;
  onScrub: (at: number) => void;
  onLive: () => void;
}) {
  const [start, end] = span;
  const total = Math.max(end - start, 1);
  const position = at === null ? total : at - start;

  // 10× is the spec's number (§1.6) and a reasonable pace to watch: run 2's six
  // minutes take thirty-eight seconds.
  useEffect(() => {
    if (at === null || at >= end) return;
    const timer = setInterval(() => onScrub(Math.min(end, at + 1)), 100);
    return () => clearInterval(timer);
  }, [at, end, onScrub]);

  return (
    <div className="scrubber">
      <button type="button" onClick={at === null ? () => onScrub(start) : onLive}>
        {at === null ? "▶ Replay" : "Back to the end"}
      </button>
      <input
        type="range"
        min={0}
        max={total}
        value={position}
        onChange={(e) => onScrub(start + Number(e.target.value))}
        aria-label="Position in the run"
      />
      <span className="mono">
        {duration(position)} / {duration(total)}
      </span>
    </div>
  );
}

function Figure({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div title={title}>
      <dt>{label}</dt>
      <dd className="mono">{value}</dd>
    </div>
  );
}
