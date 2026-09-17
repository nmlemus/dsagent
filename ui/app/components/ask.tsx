"use client";

import { CopilotKit, useAgent } from "@copilotkit/react-core/v2";
import { createContext, useCallback, useContext, useMemo } from "react";

export const AGENT = "dsagent";

/**
 * One way to put a question to the team, from anywhere on the run screen.
 *
 * The chat is not the only view of a run — that is the first of the three
 * principles — but it *is* where an answer belongs, so a question asked at a
 * section, a chart or a table has to arrive in the same conversation rather than
 * opening a second one. This is that channel: `ask("…")` appends the message and
 * runs the agent, and the reply lands in the chat at the foot of the rail where
 * the person is already looking.
 *
 * `ask` is null when nothing is listening — a replay has no agent behind it —
 * and every caller uses that to hide the affordance rather than to offer a
 * button that does nothing.
 */
const AskContext = createContext<((question: string) => void) | null>(null);

export const useAsk = () => useContext(AskContext);

/**
 * The run screen's agent provider.
 *
 * It wraps the *whole* screen, not just the chat panel, because the document and
 * its cards ask questions too. The thread is the run's chat thread, deliberately
 * not the run's own: a run and a question asked while it works would be two
 * writers on one checkpoint, and a question is exactly the thing an operator
 * asks during the long step.
 */
export function AskProvider({
  runId,
  replay,
  children,
}: {
  runId: string;
  replay: boolean;
  children: React.ReactNode;
}) {
  if (replay) return <AskContext.Provider value={null}>{children}</AskContext.Provider>;
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent={AGENT} threadId={`chat:${runId}`}>
      <Channel>{children}</Channel>
    </CopilotKit>
  );
}

function Channel({ children }: { children: React.ReactNode }) {
  const { agent, isReady } = useAgent({ agentId: AGENT });

  const ask = useCallback(
    (question: string) => {
      const text = question.trim();
      if (!text) return;
      agent.addMessage({ id: messageId(), role: "user", content: text });
      void agent.runAgent();
    },
    [agent],
  );

  // Until the runtime has synced, `agent` is a stand-in; asking it something
  // would be a question nobody receives. Better to have no button than one that
  // silently drops what was typed.
  const value = useMemo(() => (isReady ? ask : null), [isReady, ask]);
  return <AskContext.Provider value={value}>{children}</AskContext.Provider>;
}

function messageId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `ask-${Date.now()}-${Math.random()}`;
}
