"use client";

import { CopilotChat, CopilotKit, useInterrupt } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

import { GateCard, gateOf, isGate } from "./gate-card";

const AGENT = "dsagent";

/**
 * The right pane. For now it holds the gate card and nothing else; the file
 * list and the DAG land here next (`docs/ui-slice.md` §4).
 *
 * `renderInChat: false` keeps the card out of the transcript and hands it back
 * for us to place. A gate is a decision about the run, not a message in a
 * conversation, and it has to stay reachable while the user reads the artifact
 * it is asking about.
 */
function Canvas() {
  const card = useInterrupt({
    enabled: (event) => isGate(event.value),
    renderInChat: false,
    render: ({ event, interrupt, resolve }) => {
      const gate = gateOf(event?.value, interrupt);
      if (!gate) return <></>;
      return <GateCard gate={gate} onDecision={(payload) => resolve(payload)} />;
    },
  });

  return (
    <section className="pane-canvas">
      <header>Canvas</header>
      {card ? (
        <div className="canvas-body">{card}</div>
      ) : (
        <div className="empty">Artifacts from a run will appear here.</div>
      )}
    </section>
  );
}

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent={AGENT}>
      <main className="shell">
        <section className="pane-chat">
          <CopilotChat
            agentId={AGENT}
            labels={{
              chatInputPlaceholder:
                "Ask what this team can do, or start a workflow…",
            }}
          />
        </section>
        <Canvas />
      </main>
    </CopilotKit>
  );
}
