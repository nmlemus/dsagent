import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { api, runSource } from "../lib/api";
import { basename, bytes, cost, duration } from "../lib/format";
import { focusDeliverable, useRunStream } from "../lib/useRunStream";
import type { RunChart } from "../lib/useRunStream";
import type { RunDetail } from "../lib/types";
import { FileViewer } from "./FileViewer";
import { FocusOverlay, type FocusTarget } from "./FocusOverlay";
import { GateCard } from "./GateCard";
import { StatusPill } from "./StatusPill";
import { Icon } from "./Icon";
import { runTitle, stepTitle, useExperience } from "../lib/experience";
const ChartView = lazy(() => import("./ChartView"));
function RunVisualization({
  chart,
  runId,
}: {
  chart: RunChart;
  runId: string;
}) {
  const spec = useMemo(
    () =>
      chart.spec
        ? { ...chart.spec, data: { url: api.fileUrl(runId, chart.data_ref) } }
        : null,
    [chart.spec, chart.data_ref, runId],
  );
  return (
    <section className="run-visualization">
      <h3>{chart.title}</h3>
      {chart.kind === "table" ? (
        <FileViewer source={runSource(runId)} path={chart.data_ref} />
      ) : spec ? (
        <Suspense fallback={<div className="empty">Preparando gráfico…</div>}>
          <ChartView spec={spec} />
        </Suspense>
      ) : (
        <div className="empty">Gráfico sin especificación disponible.</div>
      )}
    </section>
  );
}
export function RunDock({
  runId,
  onClose,
}: {
  runId: string;
  onClose: () => void;
}) {
  const { nouns } = useExperience();
  const stream = useRunStream(runId);
  const [detail, setDetail] = useState<RunDetail | null>(null),
    [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null),
    [pinned, setPinned] = useState(false),
    [focus, setFocus] = useState<FocusTarget | null>(null);
  const [tab, setTab] = useState("results"),
    [stopping, setStopping] = useState(false);
  const refresh = useCallback(() => {
    api
      .run(runId)
      .then((d) => {
        setDetail(d);
        setError("");
      })
      .catch(() => setError("No se pudo actualizar el estado."));
  }, [runId]);
  useEffect(() => {
    refresh();
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") refresh();
    }, 4000);
    return () => clearInterval(timer);
  }, [refresh]);
  useEffect(() => {
    if (stream.ended) refresh();
  }, [stream.ended, refresh]);
  const focusPath = useMemo(() => {
    const waiting = Object.values(stream.steps)
      .filter((s) => s.status === "awaiting_gate")
      .sort((a, b) => b.ts - a.ts)[0];
    return waiting
      ? (Object.values(waiting.produces_matched ?? {})
          .flat()
          .slice(-1)[0] ?? focusDeliverable(stream.steps))
      : focusDeliverable(stream.steps);
  }, [stream.steps]);
  useEffect(() => {
    if (!pinned && focusPath) setSelected(focusPath);
  }, [focusPath, pinned]);
  const files = [...stream.files].sort(
    (a, b) =>
      Number(b.kind === "deliverable") - Number(a.kind === "deliverable") ||
      a.path.localeCompare(b.path),
  );
  const charts = Object.values(stream.charts);
  const terminal =
    detail && ["done", "failed", "stopped"].includes(detail.status);
  async function stop() {
    setStopping(true);
    try {
      await api.stop(runId);
      refresh();
    } catch {
      setError("No se pudo detener. Vuelve a intentarlo.");
    } finally {
      setStopping(false);
    }
  }
  return (
    <div className="col rundock">
      <div className="col-head">
        <div className="run-heading">
          <span className="eyebrow">
            ESPACIO DE {nouns.run.singular.toUpperCase()}
          </span>
          <h2>
            {detail
              ? runTitle(detail, detail.workflow_shape)
              : `Cargando ${nouns.run.singular}…`}
          </h2>
          <div className="sub">{runId}</div>
        </div>
        <div className="row">
          {detail && <StatusPill status={detail.status} />}
          <button className="icon-btn" onClick={onClose} aria-label="Cerrar">
            <Icon name="close" />
          </button>
        </div>
      </div>
      <div className="col-body">
        {error && (
          <div className="err-banner" role="alert">
            {error}{" "}
            <button className="text-button" onClick={refresh}>
              Reintentar
            </button>
          </div>
        )}
        {detail?.driver_error && (
          <div className="err-banner" role="alert">
            {detail.driver_error}
          </div>
        )}
        {detail && (
          <div className="run-summary-strip">
            <div>
              <span>PROGRESO</span>
              <strong>
                {detail.steps_done} / {detail.steps_total}
                <small> pasos</small>
              </strong>
            </div>
            <div>
              <span>TIEMPO REGISTRADO</span>
              <strong>{duration(detail.duration)}</strong>
            </div>
            <div>
              <span>COSTO REGISTRADO</span>
              <strong>{cost(detail.cost_usd)}</strong>
            </div>
            <div className="run-summary-actions">
              {!terminal && (
                <button
                  className="btn danger sm"
                  disabled={stopping}
                  onClick={stop}
                >
                  {stopping ? "Deteniendo…" : "Detener"}
                </button>
              )}
              <a className="btn ghost sm" href={api.downloadUrl(runId)}>
                <Icon name="download" size={15} />
                Descargar archivos
              </a>
            </div>
          </div>
        )}
        {detail?.status === "awaiting_gate" && detail.awaiting && (
          <GateCard
            runId={runId}
            gate={detail.awaiting}
            stepTitle={(() => {
              const shape = detail.workflow_shape?.steps.find(
                (s) => s.id === detail.awaiting?.step,
              );
              return shape ? stepTitle(shape) : detail.awaiting.step;
            })()}
            onAnswered={() => {
              setPinned(false);
              refresh();
            }}
          />
        )}
        <div className="run-step-track">
          {(detail?.workflow_shape?.steps ?? []).map((shape, i) => {
            const status =
              stream.steps[shape.id]?.status ??
              detail?.steps[shape.id]?.status ??
              "pending";
            return (
              <div key={shape.id} className={`run-step ${status}`}>
                <span className="step-number">
                  {status === "done" ? (
                    <Icon name="check" size={13} />
                  ) : (
                    String(i + 1).padStart(2, "0")
                  )}
                </span>
                <div>
                  <strong>{stepTitle(shape)}</strong>
                  <small>
                    {shape.gate?.kind === "human"
                      ? "Incluye revisión humana"
                      : shape.persona}
                  </small>
                </div>
              </div>
            );
          })}
        </div>
        <div className="run-content-tabs">
          <div className="filter-tabs" aria-label="Contenido">
            <button
              aria-pressed={tab === "results"}
              className={tab === "results" ? "active" : ""}
              onClick={() => setTab("results")}
            >
              {nouns.deliverable.plural.charAt(0).toUpperCase() +
                nouns.deliverable.plural.slice(1)}{" "}
              y archivos <span>{files.length}</span>
            </button>
            <button
              aria-pressed={tab === "charts"}
              className={tab === "charts" ? "active" : ""}
              onClick={() => setTab("charts")}
            >
              Gráficos y tablas <span>{charts.length}</span>
            </button>
          </div>
          <a
            className="text-button"
            href={api.exportUrl(runId)}
            target="_blank"
            rel="noreferrer"
          >
            Informe completo
            <Icon name="external" size={14} />
          </a>
        </div>
        {tab === "charts" ? (
          <div className="run-charts">
            {charts.map((c) => (
              <RunVisualization key={c.chart_id} chart={c} runId={runId} />
            ))}
            {charts.length === 0 && (
              <div className="studio-empty">
                <Icon name="chart" size={28} />
                <h3>Aquí aparecerán tus visualizaciones</h3>
                <p>Todavía no se han registrado gráficos ni tablas.</p>
              </div>
            )}
          </div>
        ) : (
          <div className="run-results-layout">
            <div className="run-file-list">
              <span className="eyebrow">ARCHIVOS</span>
              {files.length === 0 ? (
                <p className="side-empty">
                  Los archivos aparecerán a medida que avance el trabajo.
                </p>
              ) : (
                files.map((f) => (
                  <button
                    key={f.path}
                    className={`run-file ${selected === f.path ? "selected" : ""}`}
                    onClick={() => {
                      setSelected(f.path);
                      setPinned(true);
                    }}
                    title={f.path}
                  >
                    <Icon name="file" size={17} />
                    <span>
                      <strong>{basename(f.path)}</strong>
                      <small>
                        {f.kind === "deliverable"
                          ? nouns.deliverable.singular
                          : "Archivo de trabajo"}{" "}
                        · {bytes(f.size)}
                      </small>
                    </span>
                  </button>
                ))
              )}
            </div>
            <div className="run-result-preview">
              {selected ? (
                <>
                  <div className="preview-heading">
                    <strong>{basename(selected)}</strong>
                    <button
                      className="icon-btn"
                      onClick={() => setFocus({ kind: "file", path: selected })}
                      aria-label="Ampliar"
                    >
                      <Icon name="external" size={16} />
                    </button>
                  </div>
                  <FileViewer source={runSource(runId)} path={selected} />
                </>
              ) : (
                <div className="studio-empty">
                  <Icon name="file" size={28} />
                  <h3>Un lugar para la evidencia</h3>
                  <p>Selecciona un archivo para explorar su contenido.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      {focus && (
        <FocusOverlay
          source={runSource(runId)}
          target={focus}
          onClose={() => setFocus(null)}
        />
      )}
    </div>
  );
}
