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
import { cap, runTitle, stepTitle, useExperience } from "../lib/experience";
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
  const [streamRevision, setStreamRevision] = useState(0);
  const stream = useRunStream(runId, streamRevision);
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
    detail && ["done", "failed", "stopped", "interrupted"].includes(detail.status);
  const retryable = detail && (
    ["failed", "stopped", "interrupted"].includes(detail.status) ||
    (detail.status === "awaiting_gate" && !detail.awaiting)
  );
  async function retry() {
    setStopping(true);
    try {
      await api.start(runId);
      setStreamRevision((version) => version + 1);
      refresh();
    } catch {
      setError("No se pudo reanudar. Vuelve a intentarlo.");
    } finally {
      setStopping(false);
    }
  }
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
        {detail?.status === "interrupted" && (
          <p role="alert" className="err-banner">
            La ejecución se interrumpió. Revisa los archivos y las acciones del
            último paso antes de reintentarlo: puede haber realizado cambios
            aunque no haya registrado su resultado.
          </p>
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
              {retryable && (
                <button className="btn sm" disabled={stopping} onClick={retry}>
                  {stopping ? "Reanudando…" : "Reintentar paso pendiente"}
                </button>
              )}
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
            key={detail.awaiting.gate_id ?? detail.awaiting.step}
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
              (detail?.steps[shape.id]?.status === "interrupted" ? "interrupted" : undefined) ??
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
                    {status === "interrupted"
                      ? "Interrumpido: requiere revisión"
                      : shape.gate?.kind === "human"
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
              {cap(nouns.deliverable.plural)} y archivos{" "}
              <span>{files.length}</span>
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
            {/* Reading surface: the index on the left names what came out of
                the run and separates what was asked for from what the work
                left behind; the right is one document at a time, on paper. */}
            <div className="run-file-list">
              {files.length === 0 ? (
                <p className="side-empty">
                  Los archivos aparecerán a medida que avance el trabajo.
                </p>
              ) : (
                [
                  {
                    key: "deliverable",
                    label: cap(nouns.deliverable.plural),
                    items: files.filter((f) => f.kind === "deliverable"),
                  },
                  {
                    key: "working",
                    label: "Archivos de trabajo",
                    items: files.filter((f) => f.kind !== "deliverable"),
                  },
                ]
                  .filter((group) => group.items.length > 0)
                  .map((group) => (
                    <div className="run-file-group" key={group.key}>
                      <h4>
                        {group.label} <span>{group.items.length}</span>
                      </h4>
                      {group.items.map((f) => (
                        <button
                          key={f.path}
                          className={`run-file ${selected === f.path ? "selected" : ""}`}
                          aria-current={
                            selected === f.path ? "true" : undefined
                          }
                          onClick={() => {
                            setSelected(f.path);
                            setPinned(true);
                          }}
                          title={f.path}
                        >
                          <Icon name="file" size={16} />
                          <span>
                            <strong>{basename(f.path)}</strong>
                            <small>{bytes(f.size)}</small>
                          </span>
                        </button>
                      ))}
                    </div>
                  ))
              )}
            </div>
            <div className="run-result-preview">
              {selected ? (
                <>
                  <div className="preview-heading">
                    <span className="preview-title">
                      <strong>{basename(selected)}</strong>
                      <small>
                        {files.find((f) => f.path === selected)?.kind ===
                        "deliverable"
                          ? cap(nouns.deliverable.singular)
                          : "Archivo de trabajo"}
                      </small>
                    </span>
                    <span className="preview-actions">
                      <a
                        className="icon-btn"
                        href={api.fileUrl(runId, selected)}
                        download
                        aria-label={`Descargar ${basename(selected)}`}
                      >
                        <Icon name="download" size={16} />
                      </a>
                      <button
                        className="icon-btn"
                        onClick={() =>
                          setFocus({ kind: "file", path: selected })
                        }
                        aria-label="Ampliar"
                      >
                        <Icon name="external" size={16} />
                      </button>
                    </span>
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
