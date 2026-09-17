"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { AskProvider } from "../../components/ask";
import { Document } from "../../components/document";
import { Finished, Scrubber } from "../../components/finished";
import { Drawer, type Detail } from "../../components/drawer";
import { Rail } from "../../components/rail";
import { getCatalogue } from "../../lib/api";
import { useRun } from "../../lib/use-run";

/**
 * The chat is loaded on its own, after the rest of the screen.
 *
 * CopilotKit is the largest thing on this page by far, and the run — the reason
 * the page exists — must not wait for it to hydrate. Reloading mid-run is in the
 * demo script (§6.8): what has to be quick is the steps and the document, not the
 * message box.
 */
const Chat = dynamic(
  () => import("../../components/chat").then((m) => m.Chat),
  {
    loading: () => <div className="chat is-loading" />,
  },
);

/**
 * The run screen: team rail, living document, detail drawer.
 *
 * Three regions, in the order a person uses them. The rail is the *activity* —
 * who is working, on what, what it costs; the document is the *work* — the
 * report being written, section by section; the drawer is *evidence* — the raw
 * detail behind whatever was clicked, and it stays shut until it is asked for.
 *
 * Everything on all three is rebuilt from the run's event log, so this page is
 * the same whether the run started thirty seconds ago in this tab, is being
 * watched from a second one, was started by the CLI, or finished last week.
 */
export default function RunScreen() {
  const params = useParams<{ runId: string }>();
  const runId = decodeURIComponent(String(params.runId ?? ""));
  const run = useRun(runId);
  const replay = useReplayMode();
  const [detail, setDetail] = useState<Detail>(null);
  const [openStep, setOpenStep] = useState<string | null>(null);

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
    <AskProvider runId={runId} replay={replay}>
      <div className={`run-screen${detail ? " has-drawer" : ""}`}>
        <Rail
          detail={run.detail}
          steps={run.view.steps}
          openStep={openStep}
          onOpenStep={(step) => {
            setOpenStep(step);
            if (step) setDetail({ kind: "step", id: step });
          }}
          onChanged={run.refresh}
        >
          <Chat runId={runId} detail={run.detail} replay={replay} />
        </Rail>

        <Document
          runId={runId}
          detail={run.detail}
          view={run.view}
          onOpen={setDetail}
          header={
            run.detail?.status === "done" && (
              <Finished
                runId={runId}
                detail={run.detail}
                view={run.view}
                onReplay={() => run.span && run.scrub(run.span[0])}
                replaying={run.at !== null}
              />
            )
          }
          onDecide={(decision, note) => void run.decide(decision, note)}
          deciding={run.deciding}
          onResume={() => void run.resume()}
          resuming={run.resuming}
        />

        {run.span && run.detail?.status === "done" && (
          <Scrubber
            span={run.span}
            at={run.at}
            onScrub={run.scrub}
            onLive={() => run.scrub(null)}
          />
        )}

        <Drawer
          runId={runId}
          detail={detail}
          view={run.view}
          run={run.detail}
          onClose={() => setDetail(null)}
        />
      </div>
    </AskProvider>
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
