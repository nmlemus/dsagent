import { useEffect, useReducer } from "react";
import { subscribeEvents } from "./api";
import type {
  EventEnvelope,
  FileEvent,
  NoteEvent,
  StepEvent,
  ToolEvent,
} from "./types";

// Folded live state of a run, rebuilt from the event log and then followed.
// Keyed views so the DAG, narration and canvas each read what they need.
export interface RunStream {
  steps: Record<string, StepEvent>; // latest step event per step id
  stepOrder: string[]; // first-seen order (index carries the true order)
  files: FileEvent[]; // append-only, deduped by path (latest change wins)
  notes: NoteEvent[]; // persona narration, in arrival order
  tools: Record<string, ToolEvent>; // latest per tool_call_id
  lastIndex: number;
  ended: boolean;
}

const empty: RunStream = {
  steps: {},
  stepOrder: [],
  files: [],
  notes: [],
  tools: {},
  lastIndex: 0,
  ended: false,
};

type Action = { type: "event"; e: EventEnvelope } | { type: "end" } | { type: "reset" };

function reduce(state: RunStream, action: Action): RunStream {
  if (action.type === "reset") return empty;
  if (action.type === "end") return { ...state, ended: true };

  const { index, name, value } = action.e;
  const next: RunStream = { ...state, lastIndex: Math.max(state.lastIndex, index) };

  switch (name) {
    case "dsagent.step": {
      const step = value as unknown as StepEvent;
      next.steps = { ...state.steps, [step.step]: step };
      next.stepOrder = state.stepOrder.includes(step.step)
        ? state.stepOrder
        : [...state.stepOrder, step.step];
      return next;
    }
    case "dsagent.file": {
      const file = value as unknown as FileEvent;
      const without = state.files.filter((f) => f.path !== file.path);
      next.files = [...without, file];
      return next;
    }
    case "dsagent.note": {
      next.notes = [...state.notes, value as unknown as NoteEvent];
      return next;
    }
    case "dsagent.tool": {
      const tool = value as unknown as ToolEvent;
      next.tools = { ...state.tools, [tool.tool_call_id]: tool };
      return next;
    }
    default:
      return next;
  }
}

// The last deliverable of the most recently completed step — what the canvas
// auto-focuses on (focus follows output, on step-done not on every file event).
export function focusDeliverable(steps: Record<string, StepEvent>): string | null {
  const done = Object.values(steps)
    .filter((s) => s.status === "done")
    .sort((a, b) => a.ts - b.ts);
  const last = done[done.length - 1];
  if (!last) return null;
  const paths = Object.values(last.produces_matched ?? {}).flat();
  return paths[paths.length - 1] ?? null;
}

// Subscribe to a run and expose its folded state. Re-subscribes when runId
// changes; resets so a switch never bleeds one run's steps into another's.
export function useRunStream(runId: string | null): RunStream {
  const [state, dispatch] = useReducer(reduce, empty);

  useEffect(() => {
    dispatch({ type: "reset" });
    if (!runId) return;
    const unsub = subscribeEvents(
      runId,
      (e) => dispatch({ type: "event", e }),
      { onEnd: () => dispatch({ type: "end" }) },
    );
    return unsub;
  }, [runId]);

  return state;
}
