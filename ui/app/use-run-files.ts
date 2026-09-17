"use client";

import { useAgent } from "@copilotkit/react-core/v2";
import { useEffect, useRef, useState } from "react";

/** `dsagent.file` — `docs/ui-slice.md` §3. */
export type RunFile = {
  run_id: string;
  step: string;
  path: string;
  kind: "deliverable" | "working";
  change: "created" | "modified";
  size: number;
  mtime: number;
};

/** `dsagent.step` — the subset the canvas uses. */
type StepEvent = {
  run_id: string;
  step: string;
  status: "started" | "done" | "failed" | "awaiting_gate";
  produces: string[];
};

export type RunFiles = {
  files: RunFile[];
  /** The path the viewer is showing, or null before anything has landed. */
  focused: string | null;
  focus: (path: string) => void;
};

/**
 * Workspace files as the run produces them.
 *
 * Two rules, both paid for by runs 001 and 002 (`docs/ui.md`):
 *
 * - **Append-only.** Run 002 wrote four figures in 14 seconds. The list grows in
 *   place; a file written twice updates its row rather than adding one.
 * - **Focus follows step completion, never a file event.** Run 001's last two
 *   files were 8 seconds apart and the second superseded the first
 *   (`report/findings.md`, then `report/findings.html`); run 002 did the same 3
 *   seconds apart. A canvas that auto-focused every file event would flash the
 *   markdown and replace it. So focus moves once, when a step reports `done`, to
 *   that step's last declared deliverable.
 */
export function useRunFiles(): RunFiles {
  const { agent } = useAgent();
  const [files, setFiles] = useState<RunFile[]>([]);
  const [focused, setFocused] = useState<string | null>(null);
  // The user's own click wins until the next step finishes, so a run in flight
  // cannot yank the pane away from something being read.
  const pinned = useRef(false);

  useEffect(() => {
    if (!agent) return;
    const sub = agent.subscribe({
      // `value` is optional on the protocol's CustomEvent, so it is narrowed
      // here rather than in the parameter type.
      onCustomEvent: ({ event }) => {
        const value = asObject(event.value);
        if (!value) return;

        if (event.name === "dsagent.file") {
          const file = value as RunFile;
          setFiles((prev) => {
            const i = prev.findIndex((f) => f.path === file.path);
            if (i === -1) return [...prev, file];
            const next = [...prev];
            next[i] = file;
            return next;
          });
          return;
        }

        if (event.name === "dsagent.step") {
          const step = value as StepEvent;
          if (step.status !== "done" || step.produces.length === 0) return;
          pinned.current = false;
          setFocused(step.produces[step.produces.length - 1]);
        }
      },
    });
    return () => sub.unsubscribe();
  }, [agent]);

  // Before any step finishes there is still something worth showing.
  useEffect(() => {
    if (focused === null && !pinned.current && files.length > 0) {
      setFocused(files[files.length - 1].path);
    }
  }, [files, focused]);

  return {
    files,
    focused,
    focus: (path: string) => {
      pinned.current = true;
      setFocused(path);
    },
  };
}

/** Custom-event values arrive as objects here, but a JSON string is cheap to allow. */
function asObject(value: unknown): Record<string, any> | null {
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" ? (value as Record<string, any>) : null;
}
