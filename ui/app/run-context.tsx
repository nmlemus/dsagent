"use client";

import { useAgent } from "@copilotkit/react-core/v2";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { RunFile } from "./types";
import { type StepRow, reduceStep, reduceTool } from "./types";

type Narration = { step: string; persona: string; text: string };

type RunState = {
  files: RunFile[];
  steps: StepRow[];
  focused: string | null;
  focus: (path: string) => void;
  /** Message ids the chat must not render — see `ChatPane`. */
  narrated: ReadonlySet<string>;
  narrationFor: (step: string) => Narration[];
};

const RunContext = createContext<RunState | null>(null);

export const useRun = (): RunState => {
  const value = useContext(RunContext);
  if (!value) throw new Error("useRun must be used inside <RunProvider>");
  return value;
};

/**
 * One subscription to the agent's event stream, shared by both panes.
 *
 * The chat and the canvas need the same three facts — which step is running,
 * what it has written, and which assistant messages belong to a persona rather
 * than to the orchestrator — and a second subscriber would mean a second copy
 * of the step bookkeeping that the third fact depends on.
 */
export function RunProvider({ children }: { children: React.ReactNode }) {
  const { agent } = useAgent();
  const [files, setFiles] = useState<RunFile[]>([]);
  const [steps, setSteps] = useState<StepRow[]>([]);
  const [narration, setNarration] = useState<Record<string, Narration>>({});
  const [focused, setFocused] = useState<string | null>(null);
  const pinned = useRef(false);

  // Which step is mid-flight, read synchronously inside the subscriber: a text
  // message's owner is decided the instant it starts, not on the next render.
  const running = useRef<{ step: string; persona: string } | null>(null);

  useEffect(() => {
    if (!agent) return;
    const sub = agent.subscribe({
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
        } else if (event.name === "dsagent.step") {
          running.current =
            value.status === "started"
              ? { step: value.step as string, persona: value.persona as string }
              : null;
          setSteps((prev) => reduceStep(prev, value));
          if (value.status === "done" || value.status === "awaiting_gate") {
            const target = lastMatched(value);
            if (target) {
              pinned.current = false;
              setFocused(target);
            }
          }
        } else if (event.name === "dsagent.tool") {
          setSteps((prev) => reduceTool(prev, value));
        }
      },

      // A persona's narration is an assistant message like any other on the
      // wire — the bridge only attributes messages from a tool literally named
      // `task`, and the runner does not use one. So ownership is decided by
      // *when* the message starts: inside a step's window it is that step's
      // persona talking, outside it the orchestrator.
      onTextMessageStartEvent: ({ event }) => {
        const owner = running.current;
        if (!owner) return;
        setNarration((prev) => ({
          ...prev,
          [event.messageId]: { ...owner, text: "" },
        }));
      },
      onTextMessageEndEvent: ({ event, textMessageBuffer }) => {
        setNarration((prev) =>
          prev[event.messageId]
            ? { ...prev, [event.messageId]: { ...prev[event.messageId], text: textMessageBuffer } }
            : prev,
        );
      },
    });
    return () => sub.unsubscribe();
  }, [agent]);

  useEffect(() => {
    if (focused === null && !pinned.current && files.length > 0) {
      setFocused(files[files.length - 1].path);
    }
  }, [files, focused]);

  const value = useMemo<RunState>(
    () => ({
      files,
      steps,
      focused,
      focus: (path: string) => {
        pinned.current = true;
        setFocused(path);
      },
      narrated: new Set(Object.keys(narration)),
      narrationFor: (step: string) =>
        Object.values(narration).filter((n) => n.step === step && n.text.trim()),
    }),
    [files, steps, focused, narration],
  );

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

/**
 * The step's last real file, for the focus move.
 *
 * `produces` is the promise and may be a glob, so it is not necessarily a path;
 * `produces_matched` maps each promise to what actually landed. Reading the
 * matched side is what lets a step whose contract is `artifacts/figures/*.png`
 * focus an actual figure.
 */
function lastMatched(step: Record<string, any>): string | null {
  const declared: string[] = step.produces ?? [];
  const matched: Record<string, string[]> = step.produces_matched ?? {};
  for (let i = declared.length - 1; i >= 0; i--) {
    const hits = matched[declared[i]] ?? [];
    if (hits.length > 0) return hits[hits.length - 1];
  }
  return null;
}

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
