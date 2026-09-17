"use client";

import { CopilotKit } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

import { Canvas } from "./canvas";
import { ChatPane } from "./chat-pane";
import { RunProvider } from "./run-context";

const AGENT = "dsagent";

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent={AGENT}>
      {/* One subscription to the run's events, shared: the canvas draws them,
          and the chat needs them to know which messages are a persona's. */}
      <RunProvider>
        <main className="shell">
          <ChatPane />
          <Canvas />
        </main>
      </RunProvider>
    </CopilotKit>
  );
}
