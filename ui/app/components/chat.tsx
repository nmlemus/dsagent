"use client";

import { CopilotChat, useAgentContext } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

import type { RunDetail } from "../lib/api";

import { AGENT, useSelection } from "./ask";

/**
 * The conversation, about this run.
 *
 * It is not what drives the run — the server does that (`docs/ui-product.md`
 * §4.1) — so the chat gets a thread of its own rather than the run's. A run and a
 * question asked while it works would otherwise be two writers on one checkpoint,
 * and a question is exactly the thing an operator asks *during* the long step.
 *
 * What ties them together is context: `useAgentContext` puts the run on screen
 * into the request, and the orchestrator has `list_run_files` / `read_run_file`,
 * so "which finding should I be most careful with?" is answered from the
 * artifacts rather than from memory (§6.10).
 *
 * The provider is not here: `AskProvider` wraps the whole run screen, because
 * the document and its cards ask questions too and every answer has to land in
 * this one conversation.
 */
export function Chat({ runId, detail, replay }: { runId: string; detail: RunDetail | null; replay: boolean }) {
  if (replay) {
    return (
      <section className="chat is-replay">
        <p className="dim">
          This run is a recording. Ask the team questions in a live session — nothing is
          listening behind a replay.
        </p>
      </section>
    );
  }
  return (
    <>
      <RunContext runId={runId} detail={detail} />
      <section className="chat">
        <div className="chat-invite">
          <h2>Ask the team</h2>
          <p>
            Which finding is fragile, what a step actually did, what the gate was checking —
            the orchestrator answers from this run’s own files.
          </p>
        </div>
        <CopilotChat
          agentId={AGENT}
          labels={{
            chatInputPlaceholder: "Ask about this run…",
            // The rail is 320px of navy with a step list above it; a two-line
            // disclaimer under every message box costs more of it than it is
            // worth, and the same sentence is on the home screen.
            chatDisclaimerText: "",
          }}
        />
      </section>
    </>
  );
}

/** Tells the orchestrator which run the person is looking at. */
function RunContext({ runId, detail }: { runId: string; detail: RunDetail | null }) {
  const { selection } = useSelection();
  useAgentContext({
    description:
      "The run currently open in the operator's browser. Use list_run_files and " +
      "read_run_file with this run_id to answer questions about it; do not start a " +
      "new run unless asked to.",
    value: {
      run_id: runId,
      workflow: detail?.workflow ?? "",
      status: detail?.status ?? "",
      inputs: detail?.inputs ?? {},
      steps: Object.values(detail?.steps ?? {}).map((s) => ({ id: s.id, status: s.status })),
      // What the reader has brushed on a chart, if anything. It rides with
      // every message while the selection stands, which is what the line under
      // the chart promises when it says "sent as context".
      selection: selection
        ? { chart: selection.chartId, of: selection.title, is: selection.text }
        : null,
    },
  });
  return null;
}
