"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { Canvas } from "../../components/canvas";
import { Chat } from "../../components/chat";
import { Progress } from "../../components/progress";
import { getCatalogue } from "../../lib/api";
import { initial } from "../../lib/format";
import { useRun } from "../../lib/use-run";

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
    <main className="run-screen">
      <div className="run-pane run-pane-chat">
        <Chat runId={runId} detail={run.detail} replay={replay} />
      </div>

      <div className="run-pane run-pane-work">
        <Progress
          runId={runId}
          detail={run.detail}
          steps={run.view.steps}
          onOpen={run.pin}
          onDecide={(decision, note) => void run.decide(decision, note)}
          deciding={run.deciding}
        />
        <Canvas
          runId={runId}
          files={run.view.files}
          focused={run.focused}
          onFocus={run.pin}
          waiting={<Waiting run={run} />}
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
