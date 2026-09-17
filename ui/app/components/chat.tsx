"use client";

import { CopilotKit, CopilotChat, useAgentContext } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

import type { RunDetail } from "../lib/api";

const AGENT = "dsagent";

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
 * artifacts rather than from memory (§7.9).
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
    <CopilotKit runtimeUrl="/api/copilotkit" agent={AGENT} threadId={`chat:${runId}`}>
      <RunContext runId={runId} detail={detail} />
      <section className="chat">
        <CopilotChat
          agentId={AGENT}
          labels={{
            chatInputPlaceholder: "Ask about this run — findings, files, what a step did…",
          }}
        />
      </section>
    </CopilotKit>
  );
}

/** Tells the orchestrator which run the person is looking at. */
function RunContext({ runId, detail }: { runId: string; detail: RunDetail | null }) {
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
    },
  });
  return null;
}
