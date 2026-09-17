"use client";

import { CopilotChat, CopilotKit } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

import { Canvas } from "./canvas";

const AGENT = "dsagent";

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
