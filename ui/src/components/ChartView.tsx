import { useEffect, useRef } from "react";
import embed from "vega-embed";

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
    let view: { finalize: () => void } | null = null;
    chartTheme().then((config) => {
      if (disposed || !ref.current) return;
      embed(ref.current, spec as never, {
        config: config as never,
        actions: false,
      })
        .then((res) => {
          if (disposed) res.view.finalize();
          else view = res.view;
        })
        .catch(() => {
          if (ref.current)
            ref.current.textContent = "No se pudo mostrar este gráfico.";
        });
    });
    return () => {
      disposed = true;
      view?.finalize();
    };
  }, [spec]);

  return (
    <div className="viewer" style={{ padding: 12, background: "#fff" }}>
      <div ref={ref} />
    </div>
  );
}

export default ChartView;
