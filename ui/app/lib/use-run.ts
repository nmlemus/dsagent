"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  type RunDetail,
  type RunSummary,
  answerGate,
  eventsUrl,
  getRun,
  listRuns,
  startRun,
} from "./api";
import { type RunEvent, type RunView, emptyRun, reduce, reduceAll } from "./run-state";

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
  /** Pick a stopped run up again — the runner's own resume. */
  resume: () => Promise<void>;
  resuming: boolean;
  refresh: () => void;
  /** Every event this run has emitted, in order — what a replay scrubs. */
  events: RunEvent[];
  /** Show the run as it was at this moment, or `null` for as it is now. */
  at: number | null;
  scrub: (at: number | null) => void;
  /** The first and last event's clock, for a scrubber to draw a track from. */
  span: [number, number] | null;
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
  const [resuming, setResuming] = useState(false);
  const [nonce, setNonce] = useState(0);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [at, setAt] = useState<number | null>(null);

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
    const seen: RunEvent[] = [];
    const source = new EventSource(eventsUrl(runId));
    source.onopen = () => {
      setView(current);
      setEvents([]);
    };

    // The server closes a finished run's stream with `event: end`, and a *named*
    // SSE event never reaches `onmessage` — so listening only there left the
    // stream unclosed on this side, and `EventSource` dutifully reconnected
    // every three seconds for as long as the tab stayed open, collecting a fresh
    // end frame each time. A finished run is finished: close, and refresh the
    // summary once so the header settles on its final numbers.
    const done = () => {
      source.close();
      setNonce((n) => n + 1);
    };
    source.addEventListener("end", done);

    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as RunEvent;
      // Belt and braces for the same frame arriving unnamed.
      if (!event.name) {
        done();
        return;
      }
      const indexed = { ...event, index: Number(message.lastEventId) };
      current = reduce(current, indexed);
      seen.push(indexed);
      setView(current);
      // Kept whole so the run can be replayed on the same screen. A run's log is
      // a few hundred objects; the alternative is fetching it a second time.
      setEvents([...seen]);
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

  const resume = useCallback(async () => {
    setResuming(true);
    try {
      await startRun(runId);
      setNonce((n) => n + 1);
    } finally {
      setResuming(false);
    }
  }, [runId]);

  // A pin holds the canvas on the operator's file until the next step finishes,
  // which is also the next moment the run has something better to show. Counting
  // finished steps rather than clearing the pin on a timer keeps that derivable.
  const settled = view.steps.filter((s) => s.status !== "started").length;
  const focused = pinned?.at === settled ? pinned.path : view.focus;
  const pin = useCallback((path: string) => setPinned({ path, at: settled }), [settled]);

  // Scrubbing rebuilds the screen from the log up to a moment, which is the same
  // reduction the live screen does — a replay is not a second implementation.
  const shown = useMemo(
    () => (at === null ? view : reduceAll(emptyRun, events.filter((e) => e.value.ts <= at))),
    [at, view, events],
  );
  const span: [number, number] | null =
    events.length > 1 ? [events[0].value.ts, events[events.length - 1].value.ts] : null;

  return {
    detail, view: shown, error, loading, focused, pin, decide, deciding, resume, resuming,
    refresh, events, at, scrub: setAt, span,
  };
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
