"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

/**
 * A draggable divider between two regions.
 *
 * The run screen is three panes on one screen and no two operators want the same
 * split: a stakeholder watching wants the progress tall, someone reading a report
 * wants the canvas tall. The size is a CSS variable on the container and is
 * remembered per axis, so the screen comes back the way it was left.
 *
 * Keyboard too — arrow keys move it, because a divider that only answers a mouse
 * is a control some people simply do not have.
 */
export function useSplit(key: string, initial: number, range: [number, number]) {
  const [dragged, setDragged] = useState<number | null>(null);
  // `localStorage` is an external store, so it is read as one: no effect reaches
  // in to overwrite state after the first paint, and the server renders the
  // default without a hydration mismatch.
  const saved = useSyncExternalStore(
    subscribeToStorage,
    () => readSplit(key),
    () => null,
  );

  const set = useCallback(
    (value: number) => {
      const clamped = Math.min(range[1], Math.max(range[0], value));
      setDragged(clamped);
      try {
        window.localStorage.setItem(`dsagent.split.${key}`, String(Math.round(clamped)));
      } catch {
        /* a browser with storage off still gets to drag it, just not to keep it */
      }
    },
    [key, range],
  );

  const stored = saved != null && saved >= range[0] && saved <= range[1] ? saved : null;
  return [dragged ?? stored ?? initial, set] as const;
}

function readSplit(key: string): number | null {
  try {
    const raw = window.localStorage.getItem(`dsagent.split.${key}`);
    const value = Number(raw);
    return raw != null && Number.isFinite(value) ? value : null;
  } catch {
    return null;
  }
}

/** A second tab moving the divider is a change worth hearing about. */
function subscribeToStorage(onChange: () => void): () => void {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

export function Resizer({
  axis,
  label,
  onMove,
}: {
  axis: "x" | "y";
  label: string;
  /** Called with the pointer's position along the axis, in px from the viewport. */
  onMove: (position: number) => void;
}) {
  const dragging = useRef(false);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!dragging.current) return;
      e.preventDefault();
      onMove(axis === "x" ? e.clientX : e.clientY);
    };
    const stop = () => {
      dragging.current = false;
      document.body.classList.remove("is-resizing");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
  }, [axis, onMove]);

  return (
    <div
      className={`resizer resizer-${axis}`}
      role="separator"
      aria-label={label}
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      tabIndex={0}
      onPointerDown={(e) => {
        dragging.current = true;
        document.body.classList.add("is-resizing");
        e.preventDefault();
      }}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 48 : 16;
        const box = (e.target as HTMLElement).getBoundingClientRect();
        const from = axis === "x" ? box.left : box.top;
        if (e.key === (axis === "x" ? "ArrowLeft" : "ArrowUp")) onMove(from - step);
        else if (e.key === (axis === "x" ? "ArrowRight" : "ArrowDown")) onMove(from + step);
        else return;
        e.preventDefault();
      }}
    />
  );
}
