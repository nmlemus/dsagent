import { useEffect, useRef } from "react";
import embed from "vega-embed";
import type { View } from "vega";

// The server owns the Vega config so a chart looks the same here and in an
// exported report (GET /chart-theme). Fetched once and shared.
let themePromise: Promise<Record<string, unknown>> | null = null;
function chartTheme(): Promise<Record<string, unknown>> {
  if (!themePromise) {
    themePromise = fetch("/chart-theme")
      .then((r) => (r.ok ? r.json() : {}))
      .catch(() => ({}));
  }
  return themePromise;
}

export function ChartView({ spec }: { spec: Record<string, unknown> }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let disposed = false;
    let view: View | null = null;
    let observer: ResizeObserver | null = null;
    chartTheme().then((config) => {
      if (disposed || !ref.current) return;
      const responsive = spec.width === "container";
      const initialWidth = ref.current.clientWidth;
      const chartSpec = responsive
        ? {
            ...spec,
            width: initialWidth,
            autosize: spec.autosize ?? { type: "fit-x", contains: "padding" },
          }
        : spec;
      embed(ref.current, chartSpec as never, {
        config: config as never,
        actions: false,
      })
        .then((res) => {
          if (disposed) res.view.finalize();
          else {
            view = res.view;
            // Vega listens to window.resize, but opening/collapsing a sidebar
            // also changes the available space without resizing the window.
            // Only container-sized charts follow it; fixed/faceted specs keep
            // their declared sizing and can scroll inside the viewer.
            if (responsive && ref.current) {
              let width = initialWidth;
              observer = new ResizeObserver(([entry]) => {
                const nextWidth = Math.floor(entry.contentRect.width);
                if (disposed || nextWidth <= 0 || nextWidth === width) return;
                width = nextWidth;
                res.view.width(width).resize().runAsync().catch(() => {
                  if (!disposed && ref.current)
                    ref.current.textContent = "No se pudo ajustar este gráfico.";
                });
              });
              observer.observe(ref.current);
            }
          }
        })
        .catch(() => {
          if (!disposed && ref.current)
            ref.current.textContent = "No se pudo mostrar este gráfico.";
        });
    });
    return () => {
      disposed = true;
      observer?.disconnect();
      view?.finalize();
    };
  }, [spec]);

  return (
    <div
      className="viewer"
      style={{ padding: 12, background: "#fff", minWidth: 0, overflowX: "auto" }}
    >
      {/* vega-embed defaults to inline-block. With width: "container" that
          shrink-wraps around a canvas whose initial width is zero, keeping the
          entire chart at zero pixels. Give the host its width before embed. */}
      <div ref={ref} style={{ display: "block", width: "100%" }} />
    </div>
  );
}

export default ChartView;
