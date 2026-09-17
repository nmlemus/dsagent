"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import dynamic from "next/dynamic";

import { Canvas } from "../../components/canvas";
import { Progress } from "../../components/progress";
import { Resizer, useSplit } from "../../components/resizer";
import { getCatalogue } from "../../lib/api";
import { initial } from "../../lib/format";
import { useRun } from "../../lib/use-run";

/**
 * The chat is loaded on its own, after the rest of the screen.
 *
 * CopilotKit is the largest thing on this page by far, and the run — the reason
 * the page exists — must not wait for it to hydrate. Reloading mid-run is in the
 * demo script (§7.6): what has to be quick is the steps and the files, not the
 * message box.
 */
const Chat = dynamic(() => import("../../components/chat").then((m) => m.Chat), {
  loading: () => <div className="chat is-loading" />,
});

/**
 * The run screen: conversation left, progress top-right, files bottom-right.
 *
 * Everything on it is rebuilt from the run's event log, so this page is the same
 * whether the run started thirty seconds ago in this tab, is being watched from a
 * second one, was started by the CLI, or finished last week.
 */
export default function RunScreen() {
  const params = useParams<{ runId: string }>();
  const runId = decodeURIComponent(String(params.runId ?? ""));
  const run = useRun(runId);
  const replay = useReplayMode();
  const [reportFull, setReportFull] = useState(false);
  const [chatWidth, setChatWidth] = useSplit("chat", 380, CHAT_RANGE);
  const [progressHeight, setProgressHeight] = useSplit("progress", 380, PROGRESS_RANGE);

  if (run.error) {
    return (
      <main className="page">
        <div className="notice">
          {run.error}. <Link href="/">Back to runs</Link>
        </div>
      </main>
    );
  }

  return (
    <main
      className="run-screen"
      style={
        {
          "--chat-w": `${chatWidth}px`,
          // A waiting gate — or a failure — is the most important thing on the
          // screen, and neither the decision nor the reason may sit below a
          // fold. The region grows to fit and goes back to the operator's own
          // split once the run is moving again.
          "--progress-h": `${needsRoom(run) ? Math.max(progressHeight, STOPPED_MIN_H) : progressHeight}px`,
        } as React.CSSProperties
      }
    >
      <div className="run-pane run-pane-chat">
        <Chat runId={runId} detail={run.detail} replay={replay} />
      </div>
      <Resizer axis="x" label="Resize the conversation" onMove={setChatWidth} />

      <div className="run-pane run-pane-work">
        <Progress
          detail={run.detail}
          steps={run.view.steps}
          onOpen={run.pin}
          onDecide={(decision, note) => void run.decide(decision, note)}
          deciding={run.deciding}
          onResume={() => void run.resume()}
          resuming={run.resuming}
        />
        <Resizer
          axis="y"
          label="Resize the progress panel"
          onMove={(y) => setProgressHeight(y - HEADER_PX)}
        />
        <Canvas
          runId={runId}
          files={run.view.files}
          focused={run.focused}
          onFocus={run.pin}
          waiting={<Waiting run={run} />}
          finished={run.detail?.status === "done"}
          reportFull={reportFull}
          onReportFull={setReportFull}
        />
      </div>
    </main>
  );
}

/**
 * What fills the canvas before the first file exists.
 *
 * `analyze` ran for 187 seconds before its first figure in run 003. A blank pane
 * for three minutes is the difference between "working" and "broken", so this
 * says who is working and on what (§3, last paragraph).
 */
function Waiting({ run }: { run: ReturnType<typeof useRun> }) {
  const current = run.view.steps.find((s) => s.status === "started");
  if (!current) {
    return (
      <p className="view-note dim">
        {run.detail?.status === "pending"
          ? "The run is about to start."
          : "Files will appear here as the run writes them."}
      </p>
    );
  }
  const calls = Object.entries(current.tools).sort((a, b) => b[1] - a[1]);
  return (
    <div className="waiting">
      <span className="waiting-mark">{initial(current.persona)}</span>
      <p className="waiting-who">
        {current.persona} is working on <b>{current.step}</b>
      </p>
      <ul className="waiting-tools">
        {calls.map(([tool, n]) => (
          <li key={tool} className="mono">
            {tool} <b>{n}</b>
          </li>
        ))}
        {calls.length === 0 && <li className="dim">reading the step’s instructions…</li>}
      </ul>
    </div>
  );
}

/** Limits that keep a drag from making either region useless. */
const CHAT_RANGE: [number, number] = [280, 720];
const PROGRESS_RANGE: [number, number] = [120, 900];
const STOPPED_MIN_H = 560;

/** Whether the run is stopped and the panel has to show why, in full. */
function needsRoom(run: ReturnType<typeof useRun>): boolean {
  if (run.view.gateStep) return true;
  const status = run.detail?.status;
  return status === "failed" || (status === "awaiting_gate" && !run.detail?.live);
}
const HEADER_PX = 56;

/** Whether this server is serving a recording; the chat has nothing behind it. */
function useReplayMode(): boolean {
  const [replay, setReplay] = useState(false);
  useEffect(() => {
    let live = true;
    getCatalogue()
      .then((c) => live && setReplay(Boolean(c.replay)))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);
  return replay;
}
