"use client";

import { useEffect, useState } from "react";

import { API, type RunDetail, fileUrl } from "../lib/api";
import { duration, money } from "../lib/format";
import type { StepRow } from "../lib/run-state";
import { useClock } from "../lib/use-run";

/**
 * The decision the run is waiting for, at the step that is waiting.
 *
 * It renders from the event log rather than from a live interrupt, which is what
 * lets it appear in a second tab, survive a reload, and still be answerable after
 * the server has been restarted (§7.5, §7.6, §7.11). Answering posts to
 * `POST /runs/{id}/gate`; the run is what holds the question, not this browser.
 */
export function GateCard({
  runId,
  step,
  detail,
  next,
  onDecide,
  busy,
}: {
  runId: string;
  step: StepRow;
  /** The run, for what it has cost so far. */
  detail: RunDetail | null;
  /** The step this decision releases — what is actually being approved. */
  next: { id: string; persona: string } | null;
  onDecide: (decision: "approve" | "reject", note: string) => void;
  busy: boolean;
}) {
  const [note, setNote] = useState("");
  const now = useClock(true);
  const gate = step.gate;
  if (!gate) return null;

  // The wait is the number that says whether this gate earns its cost — run 003
  // spent 37 seconds here and nothing on the screen said so (M2.2.1, item 5).
  const waiting = gate.asked_at ? now - gate.asked_at : 0;

  return (
    <div className="gate">
      <div className="gate-head">
        <span className="gate-mark" aria-hidden="true" />
        <div>
          <div className="gate-title">{step.persona} needs a decision</div>
          <div className="gate-meta mono">
            {step.step} · waiting {duration(waiting)}
          </div>
        </div>
      </div>

      <p className="gate-message">{gate.prompt}</p>

      {/* Every approval says exactly what is approved — the third of the three
          principles, and the one thing the research found nobody doing well.
          Not "continue?", but: this person starts this work, on this, and it
          will cost about this much. */}
      <div className="gate-what">
        <div>
          <b>What you are approving</b>
          {next
            ? `That ${next.persona} starts ${next.id} on this data as it stands. `
            : "That the run continues on this data as it stands. "}
          {kept(step).length > 0
            ? "The artifacts above are what they will work from."
            : "The section above is what they will work from."}
        </div>
        <div>
          <b>What it costs</b>
          {money(detail?.cost_usd)} so far.{" "}
          <Estimate detail={detail} />
          <br />
          The run is paused until you answer; waiting costs nothing.
        </div>
      </div>

      <Diff runId={runId} step={step} />

      <div className="gate-actions">
        <button
          className="btn btn-primary"
          onClick={() => onDecide("approve", note.trim())}
          disabled={busy}
        >
          Approve and continue
        </button>
        <button className="btn" onClick={() => onDecide("reject", note.trim())} disabled={busy}>
          Send back
        </button>
        {/* One line, growing when it is written in: a note is optional on an
            approval and the reason for a rejection, and neither deserves to push
            the buttons off the bottom of the panel. */}
        <input
          className="gate-note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Add a note — kept in the run’s history"
          disabled={busy}
          aria-label="Note to send with the decision"
        />
      </div>
    </div>
  );
}

/**
 * What the rest of this workflow has cost in the runs before this one.
 *
 * An estimate from *this* run's own history would be circular, and one from a
 * price list would be a guess. Past runs of the same workflow are the only
 * honest source, and when there are none the card says so rather than inventing
 * a number — "cheap until it isn't" is the complaint the research records about
 * every platform in this category.
 */
function Estimate({ detail }: { detail: RunDetail | null }) {
  const [past, setPast] = useState<number[] | null>(null);
  const workflow = detail?.workflow;

  useEffect(() => {
    if (!workflow) return;
    let live = true;
    fetch(`${API}/runs`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((body: { runs: { workflow: string; status: string; cost_usd: number | null }[] }) => {
        if (!live) return;
        setPast(
          body.runs
            .filter((r) => r.workflow === workflow && r.status === "done" && r.cost_usd)
            .map((r) => r.cost_usd as number),
        );
      })
      .catch(() => live && setPast([]));
    return () => {
      live = false;
    };
  }, [workflow]);

  if (past === null) return <>Reading what past runs cost…</>;
  if (past.length === 0) return <>No finished run of this workflow yet, so no estimate.</>;
  const average = past.reduce((sum, c) => sum + c, 0) / past.length;
  return (
    <>
      {past.length} finished {past.length === 1 ? "run" : "runs"} of this workflow cost{" "}
      {money(average)} in total on average.
    </>
  );
}

/** What the step promised and delivered — the thing being approved, by name. */
function kept(step: StepRow): string[] {
  return Object.values(step.matched).flat();
}

/**
 * What changed since you sent this back.
 *
 * A gate asked a second time with nothing to show for it is the same question
 * again, and answering it is guesswork. The runner keeps the refused version at
 * the moment of rejection; this fetches both and shows the lines that differ.
 * Nothing is diffed on a first ask, because there is nothing to diff.
 */
function Diff({ runId, step }: { runId: string; step: StepRow }) {
  const previous = step.gates.filter((g) => g.decision === "reject").length;
  const path = kept(step).find((p) => p.endsWith(".md"));
  const [lines, setLines] = useState<{ sign: string; text: string }[] | null>(null);

  useEffect(() => {
    if (!previous || !path) return;
    let live = true;
    const was = `${API}/runs/${encodeURIComponent(runId)}/gate-version/` +
      `${encodeURIComponent(step.step)}/${previous}/` +
      `${path.split("/").map(encodeURIComponent).join("/")}`;
    Promise.all([
      fetch(was).then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status))))),
      fetch(fileUrl(runId, path)).then((r) => (r.ok ? r.text() : "")),
    ])
      .then(([before, after]) => live && setLines(changed(before, after)))
      .catch(() => live && setLines([]));
    return () => {
      live = false;
    };
  }, [runId, step.step, previous, path]);

  if (!previous || !path) return null;
  if (lines === null) return null;
  return (
    <div className="gate-diff">
      <b className="mono">{path}</b>
      {lines.length === 0 ? (
        <p className="dim">
          Nothing has changed in this file since you sent it back.
        </p>
      ) : (
        <pre>
          {lines.map((line, i) => (
            <span key={i} className={`diff-${line.sign === "+" ? "add" : "cut"}`}>
              {line.sign} {line.text}
              {"\n"}
            </span>
          ))}
        </pre>
      )}
    </div>
  );
}

/**
 * The lines one version has and the other does not.
 *
 * Set difference rather than a proper edit script: a reader wants to know what
 * the persona added and removed, and for a markdown artifact of thirty lines
 * that is the same answer an LCS would give, without the algorithm. Capped,
 * because a rewritten file is a rewritten file and printing all of it teaches
 * nobody anything.
 */
function changed(before: string, after: string): { sign: string; text: string }[] {
  const was = before.split("\n").map((l) => l.trimEnd());
  const now = after.split("\n").map((l) => l.trimEnd());
  const wasSet = new Set(was);
  const nowSet = new Set(now);
  const out = [
    ...was.filter((l) => l && !nowSet.has(l)).map((text) => ({ sign: "-", text })),
    ...now.filter((l) => l && !wasSet.has(l)).map((text) => ({ sign: "+", text })),
  ];
  return out.slice(0, 20);
}

/**
 * Every decision this gate has had, oldest first.
 *
 * A run that was sent back and later approved keeps both: the note explaining
 * the rejection is the reason the second answer was possible, and §7.10 asks for
 * it to still be there when the run has finished.
 */
export function GateVerdict({ step }: { step: StepRow }) {
  const history = step.gates.length > 0 ? step.gates : step.gate?.decision ? [step.gate] : [];
  if (history.length === 0) return null;

  return (
    <div className="verdicts">
      {history.map((gate, i) => {
        const waited =
          gate.decided_at && gate.asked_at ? duration(gate.decided_at - gate.asked_at) : null;
        return (
          <div key={`${gate.decided_at}-${i}`} className={`verdict is-${gate.decision}`}>
            <span className="verdict-label">
              {gate.decision === "approve" ? "Approved" : "Sent back"}
              {waited ? ` after ${waited}` : ""}
            </span>
            {gate.note && <p className="verdict-note">“{gate.note}”</p>}
          </div>
        );
      })}
    </div>
  );
}
