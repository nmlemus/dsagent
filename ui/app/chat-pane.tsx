"use client";

import {
  CopilotChat,
  CopilotChatMessageView,
} from "@copilotkit/react-core/v2";

import { useRun } from "./run-context";

const AGENT = "dsagent";

/**
 * The chat, minus the personas.
 *
 * While a workflow runs, each persona's own commentary arrives as an ordinary
 * assistant message on the same stream as the orchestrator's — the bridge
 * attributes only what comes through a tool named `task`, and the runner does
 * not use one. Left alone, the transcript reads as one voice changing
 * personality mid-conversation: marie narrating a profiling script, then noel
 * describing figures, with nothing marking the switch
 * (`docs/runs/ui-canvas-002.png`).
 *
 * So they are withheld here and shown in the DAG panel against the step they
 * belong to. This is a **rendering override**: `messageView` is a documented
 * slot, `CopilotChatMessageView` takes the messages to draw, and nothing about
 * the run, the transcript or the backend changes — the withheld messages are
 * still in the thread and still go back to the model as context.
 */
type MessageViewProps = React.ComponentProps<typeof CopilotChatMessageView>;

/**
 * Defined at module scope, not inline: a slot is a component type, so a new
 * function each render would remount the whole message list on every token.
 * It reads the run from context rather than from a closure for the same reason.
 */
function FilteredMessageView(props: MessageViewProps) {
  const { narrated } = useRun();
  return (
    <CopilotChatMessageView
      {...props}
      messages={(props.messages ?? []).filter((m) => !narrated.has(m.id))}
    />
  );
}
// The slot's type is the component *type*, statics included.
FilteredMessageView.Cursor = CopilotChatMessageView.Cursor;

export function ChatPane() {
  return (
    <section className="pane-chat">
      <CopilotChat
        agentId={AGENT}
        messageView={FilteredMessageView}
        labels={{
          chatInputPlaceholder: "Ask what this team can do, or start a workflow…",
        }}
      />
    </section>
  );
}
