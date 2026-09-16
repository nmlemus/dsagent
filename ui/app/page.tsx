"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="dsagent">
      <main className="shell">
        <section className="pane-chat">
          <CopilotChat
            labels={{
              title: "DSAgent",
              initial:
                "Ask what this team can do, or start a workflow — try “which workflows can you run?”",
            }}
          />
        </section>
        {/* Placeholder. The canvas renders workspace files from `dsagent.file`
            events, the gate card from the interrupt, and the DAG from
            `dsagent.step` — see ../docs/ui-slice.md §4. */}
        <section className="pane-canvas">
          <header>Canvas</header>
          <div className="empty">
            Artifacts from a run will appear here.
          </div>
        </section>
      </main>
    </CopilotKit>
  );
}
