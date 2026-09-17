"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type RunDetail, type RunSummary, answerGate, eventsUrl, getRun, listRuns } from "./api";
import { type RunEvent, type RunView, emptyRun, reduce } from "./run-state";

/**
 * The summary carries totals the event log does not (cost, tokens, gate wait).
 * Four seconds is slow enough to be free and fast enough that the header is not
 * visibly behind the steps beside it.
 */
const SUMMARY_POLL_MS = 4000;

export type Run = {
  detail: RunDetail | null;
  view: RunView;
  /** Null until the first load finishes; a string when the run cannot be read. */
  error: string | null;
  loading: boolean;
  /** The file the canvas is showing: the operator's pick, else the run's focus. */
  focused: string | null;
  pin: (path: string) => void;
  decide: (decision: "approve" | "reject", note?: string) => Promise<void>;
  deciding: boolean;
  refresh: () => void;
};

/**
 * One run, followed from its event log.
 *
 * The screen is rebuilt from `GET /runs/{id}/events` — the whole backlog, then
 * live — so opening a finished run, reloading mid-run and attaching to a run
 * somebody else started are the same code path. The summary is polled alongside
 * it for the numbers that are totals rather than events.
 */
export function useRun(runId: string): Run {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [view, setView] = useState<RunView>(emptyRun);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pinned, setPinned] = useState<{ path: string; at: number } | null>(null);
  const [deciding, setDeciding] = useState(false);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let live = true;
    const load = () =>
      getRun(runId)
        .then((d) => {
          if (!live) return;
          setDetail(d);
          setError(null);
        })
        .catch((e) => live && setError(e.message))
        .finally(() => live && setLoading(false));
    load();
    const timer = setInterval(load, SUMMARY_POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [runId, nonce]);

  useEffect(() => {
    // The accumulator lives here rather than in React state so that switching
    // runs starts from nothing without the effect having to reset state on its
    // way in: the connection opening is what clears the screen, and that is a
    // callback from an external system, which is exactly what effects are for.
    let current = emptyRun;
    const source = new EventSource(eventsUrl(runId));
    source.onopen = () => setView(current);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as RunEvent;
      // The end frame has no `name`: the run is over and so is the stream.
      if (!event.name) {
        source.close();
        setNonce((n) => n + 1);
        return;
      }
      current = reduce(current, { ...event, index: Number(message.lastEventId) });
      setView(current);
    };
    // A dropped connection reconnects on its own and resumes from `Last-Event-ID`,
    // which the server honours — so a flaky network costs a gap, not the screen.
    return () => source.close();
  }, [runId]);

  const decide = useCallback(
    async (decision: "approve" | "reject", note = "") => {
      setDeciding(true);
      try {
        await answerGate(runId, decision, note);
        setNonce((n) => n + 1);
      } finally {
        setDeciding(false);
      }
    },
    [runId],
  );

  // A pin holds the canvas on the operator's file until the next step finishes,
  // which is also the next moment the run has something better to show. Counting
  // finished steps rather than clearing the pin on a timer keeps that derivable.
  const settled = view.steps.filter((s) => s.status !== "started").length;
  const focused = pinned?.at === settled ? pinned.path : view.focus;
  const pin = useCallback((path: string) => setPinned({ path, at: settled }), [settled]);

  return { detail, view, error, loading, focused, pin, decide, deciding, refresh };
}

/** The home screen's list, refreshed while anything on it is moving. */
export function useRuns(): { runs: RunSummary[]; error: string | null; loading: boolean } {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const moving = useRef(false);

  useEffect(() => {
    let live = true;
    const load = () =>
      listRuns()
        .then((rows) => {
          if (!live) return;
          setRuns(rows);
          setError(null);
          moving.current = rows.some((r) => r.status === "running" || r.status === "pending");
        })
        .catch((e) => live && setError(e.message))
        .finally(() => live && setLoading(false));
    load();
    const timer = setInterval(() => moving.current && load(), 2000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  return { runs, error, loading };
}

/**
 * Seconds since the epoch, ticking once a second while `live`.
 *
 * Reading the clock during render is not allowed — and rightly: the same render
 * would produce a different number every time. Anything that counts up asks for
 * the time here instead.
 */
export function useClock(live: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(id);
  }, [live]);
  return now;
}
