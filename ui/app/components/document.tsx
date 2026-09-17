"use client";

import { useEffect, useState } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { fileUrl, type RunDetail } from "../lib/api";
import { duration, initial } from "../lib/format";
import type { RunView, StepRow } from "../lib/run-state";
import { useClock } from "../lib/use-run";

import { useAsk } from "./ask";
import { ChartCard, TableCard } from "./cards";
import { GateCard, GateVerdict } from "./gate-card";

/**
 * The run as a document the team writes in front of you.
 *
 * Not a file list and not a log: a report with a heading per step, which starts
 * as the plan — every section greyed, in the order they will be written — and
 * fills in as the steps finish. That is the whole idea of the milestone
 * (`docs/ui-living-report.md` §0): when the run ends, the document *is* the
 * deliverable, and the reader has been watching it become one.
 *
 * A section that has not happened yet still says what it will hold, because a
 * greyed heading with a sentence under it is a plan, and an empty screen is a
 * product that looks broken.
 */
export function Document({
  runId,
  detail,
  view,
  onOpen,
  onDecide,
  deciding,
  onResume,
  resuming,
  header,
  children,
}: {
  runId: string;
  detail: RunDetail | null;
  view: RunView;
  /** Point the drawer at something: a file, a step, a chart's spec. */
  onOpen: (what: { kind: "file" | "step" | "spec"; id: string }) => void;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  deciding: boolean;
  onResume: () => void;
  resuming: boolean;
  /** What a finished run puts above its first section. */
  header?: React.ReactNode;
  /** Section bodies, keyed by step id — filled in as the milestone proceeds. */
  children?: (step: StepRow, state: SectionState) => React.ReactNode;
}) {
  const now = useClock(detail?.status === "running");
  const planned = plan(detail, view);

  return (
    <main className="doc">
      <article className="doc-body">
        <p className="doc-eyebrow">{finished(detail) ? "Report" : "Report in progress"}</p>
        <h1 className="doc-title">{heading(detail)}</h1>
        <p className="doc-meta">
          <span>Team: {team(view).join(", ") || "—"}</span>
          <span>
            Dataset: <span className="mono">{dataset(detail)}</span>
          </span>
          <span>{position(planned)}</span>
        </p>

        {header}

        {planned.map(({ step, state, section }, i) => (
          <section key={section + i} className={`sec is-${state}`} aria-label={section}>
            <h2 className="sec-head">
              {i + 1}. {section}
              <span className="sec-by">
                {step?.persona ?? ""}
                {state === "running" && " · writing"}
              </span>
            </h2>
            {state === "pending" && <p className="sec-plan">{promise(step, section)}</p>}
            {state === "running" && step && <AtWork step={step} now={now} />}
            {state !== "pending" && step && <Prose runId={runId} step={step} />}
            {/* The cards the step emitted, in the order it emitted them — the
                chart under the paragraph that describes it, which is where a
                figure belongs and where a file list can never put it. */}
            {step &&
              cardsFor(view, planned, i)
                .map((card) =>
                  card.kind === "chart" ? (
                    <ChartCard key={card.chartId} card={card} onOpen={onOpen} />
                  ) : (
                    <TableCard key={card.chartId} card={card} />
                  ),
                )}
            {children?.(step as StepRow, state)}
            {state === "done" && step && (
              <Sources step={step} onOpen={(id) => onOpen({ kind: "file", id })} />
            )}
            {state === "done" && step && <AskRow step={step} section={section} />}
            {/* The decision renders **at its own step**, under the evidence it is
                about. A gate in a side panel is a question about something the
                reader has to go and find; a gate here is a question about the
                paragraph above it. */}
            {step && step.status === "awaiting_gate" && !step.gate?.decision && (
              <GateCard
                runId={runId}
                step={step}
                detail={detail}
                next={nextOf(planned, i)}
                onDecide={onDecide}
                busy={deciding}
              />
            )}
            {step && <GateVerdict step={step} />}
            {step && step.status === "awaiting_gate" && step.gate?.decision === "reject" && (
              <p className="sec-resume">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={onResume}
                  disabled={resuming}
                >
                  {resuming ? "Resuming…" : "Resume the run"}
                </button>
                <span className="dim">
                  {step.persona} will look at your note and ask again.
                </span>
              </p>
            )}
            {/* A failed step, with eyes on it: what was promised, what is
                missing, the error in the words it arrived in, and the one
                button that does something about it. Nothing after it ran. */}
            {step && state === "failed" && (
              <div className="sec-failed">
                <p className="sec-failed-what">
                  <b>
                    {step.persona} could not finish {step.step}.
                  </b>{" "}
                  Nothing after it ran.
                </p>
                {step.produces
                  .filter((entry) => (step.matched[entry] ?? []).length === 0)
                  .map((entry) => (
                    <p key={entry} className="sec-failed-missing mono">
                      missing: {entry}
                    </p>
                  ))}
                {step.error && <p className="sec-failed-error">{step.error}</p>}
                <p className="sec-resume">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={onResume}
                    disabled={resuming}
                  >
                    {resuming ? "Retrying…" : `Retry from ${step.step}`}
                  </button>
                  <span className="dim">
                    The steps before it are kept; only this one runs again.
                  </span>
                </p>
              </div>
            )}
          </section>
        ))}
      </article>
    </main>
  );
}

export type SectionState = "pending" | "running" | "done" | "failed";

/**
 * The persona at work, in the place their section will be.
 *
 * `analyze` ran for three minutes before its first chart in run 003. A blank
 * space for three minutes is the difference between "working" and "broken", so
 * the section says who is in it and what they have called so far.
 */
function AtWork({ step, now }: { step: StepRow; now: number }) {
  const calls = Object.entries(step.tools).sort((a, b) => b[1] - a[1]);
  const said = calls.map(([tool, n]) => `${tool} × ${n}`).join(" · ");
  return (
    <div className="at-work">
      <span className="at-work-av">{initial(step.persona)}</span>
      <span className="at-work-who">
        <b>{step.persona} is working on this section</b>
        <span className="mono at-work-tools">
          {said || "reading the step’s instructions…"}
        </span>
      </span>
      <span className="pill is-running">
        <i className="pill-dot" />
        {duration(now - step.startedAt)}
      </span>
    </div>
  );
}

/**
 * The section's own words — the artifact the step wrote, rendered in place.
 *
 * The document is not a summary of the run written by the UI: it *is* the run's
 * artifacts, laid out as one document. `profile` wrote `data-profile.md`, the
 * gate wrote `data-gate.md`, `analyze` wrote `findings.md` — so the section
 * shows that file, and the "Sources" strip under it names the same path. Nothing
 * on this screen says anything a persona did not write.
 */
function Prose({ runId, step }: { runId: string; step: StepRow }) {
  const path = primary(step);
  const [text, setText] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!path) return;
    let live = true;
    fetch(fileUrl(runId, path))
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((body) => live && setText(body))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [runId, path]);

  if (!path || failed) return null;
  // No skeleton: the file is on the same host and the section around it is
  // already on screen. A placeholder that flashes for 40 ms is noise.
  if (text === null) return null;
  return (
    <div className="sec-prose">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{trim(text)}</ReactMarkdown>
    </div>
  );
}

/**
 * The markdown deliverable a section shows: the first one the step promised.
 *
 * `produces` is the step's contract in the order the cartridge declared it, and
 * a workflow that promises a report names it first — so "the first `.md`" is the
 * step's own answer to which file is the point, not a guess by the UI.
 */
function primary(step: StepRow): string | null {
  for (const entry of step.produces) {
    const hit = (step.matched[entry] ?? []).find((p) => p.endsWith(".md"));
    if (hit) return hit;
  }
  return null;
}

/**
 * Drop the artifact's own title.
 *
 * The section already has a heading, put there by the workflow; the file's `#`
 * line repeats it two lines further down. Everything below the first heading is
 * the persona's, untouched.
 */
function trim(markdown: string): string {
  const lines = markdown.split("\n");
  const first = lines.findIndex((line) => line.trim() !== "");
  if (first === -1 || !lines[first].startsWith("# ")) return markdown;
  return lines.slice(first + 1).join("\n").trimStart();
}

/**
 * Two questions worth asking about this section, and a way to ask your own.
 *
 * Follow-up chips are one of the patterns the research lists under "2026
 * delight", and they earn their place here for a duller reason: they teach what
 * this chat is for. An empty input next to a finished report is a box most
 * people never type in.
 */
function AskRow({ step, section }: { step: StepRow; section: string }) {
  const ask = useAsk();
  if (!ask) return null;
  const about = `In section "${section}" (step ${step.step})`;
  return (
    <p className="ask-row">
      <button
        type="button"
        onClick={() => ask(`${about}: what should I be most careful with?`)}
      >
        What should I be careful with?
      </button>
      <button
        type="button"
        onClick={() => ask(`${about}: what did ${step.persona} actually do, and from which files?`)}
      >
        What did {step.persona} actually do?
      </button>
    </p>
  );
}

/**
 * What a finding came from, under the finding.
 *
 * Clickable sources while the run is still going is one of the 2026 patterns the
 * research names; it is also the cheapest answer to the loudest complaint about
 * every product in this category, which is confident output nobody can check.
 */
function Sources({ step, onOpen }: { step: StepRow; onOpen: (path: string) => void }) {
  const files = Object.values(step.matched).flat();
  if (files.length === 0) return null;
  return (
    <p className="sec-sources">
      <span className="sec-sources-label">Sources</span>
      {files.map((path) => (
        <button key={path} type="button" className="mono sec-source" onClick={() => onOpen(path)}>
          {path}
        </button>
      ))}
    </p>
  );
}

type Planned = { step: StepRow | undefined; state: SectionState; section: string };

/**
 * The cards that belong in section `i` — and the ones that belong nowhere.
 *
 * A card names the step that emitted it. A chart amended from the conversation
 * was once emitted by the orchestrator, which is not a step, and the card
 * disappeared off the page rather than changing on it. That is fixed at the
 * source, but the *document* still owes the reader a rule: a card whose step it
 * does not recognise goes in the last section, because the one thing a document
 * may never do is silently drop something a run produced.
 */
function cardsFor(view: RunView, planned: Planned[], i: number) {
  const known = new Set(planned.map((p) => p.step?.step));
  const mine = view.cards.filter((card) => card.step === planned[i].step?.step);
  if (i !== planned.length - 1) return mine;
  return [...mine, ...view.cards.filter((card) => !known.has(card.step))];
}

/** The step a decision at `i` releases — what is actually being approved. */
function nextOf(planned: Planned[], i: number): { id: string; persona: string } | null {
  const after = planned[i + 1]?.step;
  return after ? { id: after.step, persona: after.persona } : null;
}

/**
 * The document's outline: one entry per step the workflow declares, in order.
 *
 * Built from the *workflow* rather than from the events, so the plan is complete
 * from the first second — a reader can see what is coming before anything has
 * run. The events then fill each entry in.
 */
function plan(detail: RunDetail | null, view: RunView): Planned[] {
  const declared = detail?.workflow_shape?.steps ?? [];
  const rows = new Map(view.steps.map((s) => [s.step, s]));
  const entries = declared.length
    ? declared.map((s) => ({ id: s.id, section: s.section || s.id, persona: s.persona }))
    : view.steps.map((s) => ({ id: s.step, section: s.section || s.step, persona: s.persona }));

  return entries.map(({ id, section, persona }, index) => {
    const step = rows.get(id);
    return {
      // A step that has not started yet still needs a row: the section it will
      // write is on the page from the first second, and everything downstream
      // reads a `StepRow`. A half-built object cast to one is how a pending
      // section crashed the screen — `gates` was undefined and the verdict list
      // read its length.
      step: step ?? placeholder(id, persona, index, entries.length),
      state: stateOf(step),
      section,
    };
  });
}

/** A step the run has not reached: real shape, nothing in it. */
function placeholder(step: string, persona: string, index: number, total: number): StepRow {
  return {
    step,
    persona,
    index,
    total,
    status: "done",
    section: "",
    produces: [],
    matched: {},
    startedAt: 0,
    endedAt: null,
    error: null,
    tools: {},
    running: 0,
    gate: null,
    gates: [],
    notes: [],
  };
}

function stateOf(step: StepRow | undefined): SectionState {
  if (!step) return "pending";
  if (step.status === "failed") return "failed";
  if (step.status === "started") return "running";
  // A step parked at its undecided gate has still written its section; the
  // decision is about what comes *next*, and hiding the evidence behind it would
  // be asking someone to approve something they cannot read.
  return "done";
}

function promise(step: StepRow | undefined, section: string): string {
  const who = step?.persona ?? "The team";
  return `${who} will write ${section.toLowerCase()} here.`;
}

function position(planned: Planned[]): string {
  const done = planned.filter((p) => p.state === "done").length;
  if (done === planned.length && planned.length > 0) return "Report complete";
  return `Section ${Math.min(done + 1, planned.length)} of ${planned.length}`;
}

function team(view: RunView): string[] {
  return [...new Set(view.steps.map((s) => s.persona))];
}

function finished(detail: RunDetail | null): boolean {
  return detail?.status === "done";
}

function heading(detail: RunDetail | null): string {
  return detail?.inputs?.question || detail?.workflow || "Run";
}

function dataset(detail: RunDetail | null): string {
  const path = detail?.inputs?.data_path;
  return path ? path.slice(path.lastIndexOf("/") + 1) : "—";
}
