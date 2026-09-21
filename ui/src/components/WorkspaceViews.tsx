import { useMemo, useRef, useState } from "react";
import { api, projectSource } from "../lib/api";
import { ago, basename, bytes, cost } from "../lib/format";
import { FocusOverlay, type FocusTarget } from "./FocusOverlay";
import type { Project, RunSummary, WorkflowShape } from "../lib/types";
import type { Artifact, Dataset } from "../lib/useWorkspace";
import {
  cap,
  runTitle,
  useExperience,
  useWorkspaceConfig,
  workflowTitle,
} from "../lib/experience";
import { Icon, iconName, type IconName } from "./Icon";
import { Dialog } from "./Dialog";
import { StatusPill } from "./StatusPill";

// Every word on these screens is either the harness's own structure (a status,
// a count, an action) or comes from the cartridge's `experience`. Nothing here
// recognises a workflow, a step or an input by name.

function useWorkflowMap() {
  const { workflows } = useWorkspaceConfig();
  return useMemo(
    () => new Map(workflows.map((w) => [w.name, w] as const)),
    [workflows],
  );
}

export function EmptyState({
  icon,
  title,
  text,
  action,
  onAction,
}: {
  icon: IconName;
  title: string;
  text: string;
  action?: string;
  onAction?: () => void;
}) {
  return (
    <div className="studio-empty">
      <span className="empty-symbol">
        <Icon name={icon} size={26} />
      </span>
      <h3>{title}</h3>
      <p>{text}</p>
      {action && (
        <button className="btn primary" onClick={onAction}>
          {action}
          <Icon name="arrow" size={16} />
        </button>
      )}
    </div>
  );
}
function PipelineArt() {
  const { title, home } = useExperience();
  const [input, work, output] = [
    home.stages[0] ?? "Entrada",
    home.stages[1] ?? "Trabajo",
    home.stages[2] ?? "Resultado",
  ];
  return (
    <div
      className="pipeline-art"
      aria-label={`Proceso: ${input}, ${work}, revisión y ${output}`}
      role="img"
    >
      <div className="pipeline-orbit orbit-one" />
      <div className="pipeline-orbit orbit-two" />
      <div className="pipeline-axis" />
      <span className="pipeline-caption">{home.eyebrow}</span>
      <div className="pipeline-node node-data">
        <Icon name="data" />
        <span>{input}</span>
        <i />
      </div>
      <div className="pipeline-core">
        <Icon name="spark" size={32} />
        <span>{title.toLowerCase()}</span>
      </div>
      <div className="pipeline-node node-analysis">
        <Icon name="chart" />
        <span>{work}</span>
        <i />
      </div>
      <div className="pipeline-node node-report">
        <Icon name="file" />
        <span>{output}</span>
        <i />
      </div>
      <span className="pipeline-review">
        <Icon name="check" size={13} /> Tu criterio, en cada decisión
      </span>
    </div>
  );
}
function RunRows({
  runs,
  onOpen,
}: {
  runs: RunSummary[];
  onOpen: (id: string) => void;
}) {
  const { nouns } = useExperience();
  const byName = useWorkflowMap();
  return (
    <div className="analysis-table">
      <div className="analysis-table-head">
        <span>{nouns.run.singular.toUpperCase()}</span>
        <span>ESTADO</span>
        <span>PROGRESO</span>
        <span>ACTIVIDAD</span>
        <span />
      </div>
      {runs.map((r) => {
        const wf = byName.get(r.workflow);
        return (
          <button
            className="analysis-table-row"
            key={r.run_id}
            onClick={() => onOpen(r.run_id)}
          >
            <span className="analysis-name">
              <span className={`analysis-icon ${r.status}`}>
                <Icon name={iconName(wf?.icon)} />
              </span>
              <span>
                <strong>{runTitle(r, wf)}</strong>
                <small>{wf ? workflowTitle(wf) : r.workflow}</small>
              </span>
            </span>
            <StatusPill status={r.status} />
            <span className="table-progress">
              <span className="progress-track">
                <i
                  style={{
                    width: `${r.steps_total ? Math.min(100, (r.steps_done / r.steps_total) * 100) : 0}%`,
                  }}
                />
              </span>
              <small>
                {r.steps_done}/{r.steps_total} pasos
              </small>
            </span>
            <span className="activity-time">
              {ago(r.finished_at ?? r.started_at) || "Por iniciar"}
            </span>
            <Icon name="chevron" size={17} />
          </button>
        );
      })}
    </div>
  );
}
export function Overview({
  project,
  runs,
  datasets,
  artifacts,
  loading,
  unavailable,
  onNewRun,
  onChat,
  onRuns,
  onData,
  onOpenRun,
}: {
  project?: Project;
  runs: RunSummary[];
  datasets: Dataset[];
  artifacts: Artifact[];
  loading: boolean;
  unavailable: boolean;
  onNewRun: () => void;
  onChat: () => void;
  onRuns: () => void;
  onData: () => void;
  onOpenRun: (id: string) => void;
}) {
  const { home, nouns, nav } = useExperience();
  const gates = runs.filter((r) => r.status === "awaiting_gate");
  const done = runs.filter((r) => r.status === "done").length;
  const completed = runs.filter((r) => r.finished_at).slice(0, 6);
  const [headline, ...headlineRest] = home.headline.split(/(?<=\.)\s+/);
  return (
    <div className="workspace-page overview-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">{home.eyebrow}</span>
          <h1>{nav.overview}</h1>
          <p>{project?.goal || home.intro}</p>
        </div>
        <span className="today-label">
          {new Intl.DateTimeFormat("es", {
            day: "numeric",
            month: "long",
            year: "numeric",
          }).format(new Date())}
        </span>
      </div>
      <section className="workspace-hero">
        <div className="hero-copy">
          <span className="hero-kicker">
            <span /> {nouns.run.plural.toUpperCase()} CON REVISIÓN HUMANA
          </span>
          <h2>
            {headline}
            {headlineRest.length > 0 && (
              <>
                <br />
                <em>{headlineRest.join(" ")}</em>
              </>
            )}
          </h2>
          <p>{home.intro}</p>
          <div className="hero-actions">
            <button className="btn hero-primary" onClick={onNewRun}>
              Crear {nouns.run.singular}
              <Icon name="arrow" size={17} />
            </button>
            <button className="hero-secondary" onClick={onChat}>
              <Icon name="spark" size={16} /> Abrir {nav.chat.toLowerCase()}
            </button>
          </div>
        </div>
        <PipelineArt />
      </section>
      <div className="workspace-stats">
        {[
          {
            icon: "chart" as const,
            label: `${cap(nouns.run.plural)} con resultado`,
            value: done,
            note: `Entre ${nouns.run.plural} recientes (últimos 100)`,
            action: onRuns,
          },
          {
            icon: "data" as const,
            label: cap(nouns.resource.plural),
            value: datasets.length,
            note: "Disponibles en este proyecto",
            action: onData,
          },
          {
            icon: "clock" as const,
            label: "Pendientes de revisión",
            value: gates.length,
            note: gates.length
              ? "Tu criterio marca el siguiente paso"
              : "Sin revisiones pendientes",
            action: () => (gates[0] ? onOpenRun(gates[0].run_id) : onRuns()),
          },
        ].map((s) => (
          <button className="stat-tile" key={s.label} onClick={s.action}>
            <span className="stat-icon">
              <Icon name={s.icon} />
            </span>
            <span className="stat-content">
              <span>{s.label}</span>
              <strong>
                {loading || unavailable
                  ? "—"
                  : s.value.toString().padStart(2, "0")}
              </strong>
              <small>
                {loading
                  ? "Cargando datos del proyecto"
                  : unavailable
                    ? "Datos pendientes de conexión"
                    : s.note}
              </small>
            </span>
            <Icon name="external" size={15} />
          </button>
        ))}
      </div>
      <div className="overview-lower">
        <section className="surface recent-analysis">
          <div className="section-heading">
            <div>
              <h2>Tu trabajo, en movimiento</h2>
              <p>Retoma {nouns.run.plural} en curso o revisa sus resultados.</p>
            </div>
            <button className="text-button" onClick={onRuns}>
              Ver {nav.runs.toLowerCase()}
              <Icon name="arrow" size={15} />
            </button>
          </div>
          {loading ? (
            <div className="skeleton-list" aria-label="Cargando">
              <i />
              <i />
              <i />
            </div>
          ) : runs.length ? (
            <RunRows runs={runs.slice(0, 4)} onOpen={onOpenRun} />
          ) : (
            <EmptyState
              icon="chart"
              title="Aquí empieza tu próximo resultado"
              text={`Sube ${nouns.resource.plural} y crea ${nouns.run.singular}. Sus avances y resultados aparecerán en este espacio.`}
              action={`Preparar ${nouns.resource.plural}`}
              onAction={onData}
            />
          )}
        </section>
        <section className="surface next-step">
          <div className="section-heading">
            <div>
              <span className="eyebrow">EL SIGUIENTE PASO</span>
              <h2>{gates.length ? "Tu revisión cuenta" : "Cómo funciona"}</h2>
            </div>
            <Icon name="spark" />
          </div>
          {gates.length ? (
            <>
              <span className="review-count">
                {gates.length.toString().padStart(2, "0")}
              </span>
              <p>
                Hay {nouns.run.plural} esperando tu revisión para continuar.
              </p>
              <button
                className="btn primary"
                onClick={() => onOpenRun(gates[0].run_id)}
              >
                Revisar ahora
                <Icon name="arrow" size={16} />
              </button>
            </>
          ) : (
            <>
              <ol className="getting-started">
                {home.steps.slice(0, 3).map((step, i) => (
                  <li key={step.title}>
                    <span>{String(i + 1).padStart(2, "0")}</span>
                    <div>
                      <strong>{step.title}</strong>
                      <small>{step.text}</small>
                    </div>
                    {i === 0 && datasets.length > 0 && (
                      <Icon name="check" size={16} />
                    )}
                    {i === 2 && done > 0 && <Icon name="check" size={16} />}
                  </li>
                ))}
              </ol>
              <button className="text-button" onClick={onChat}>
                Empezar con el {nav.chat.toLowerCase()}
                <Icon name="arrow" size={15} />
              </button>
            </>
          )}
        </section>
      </div>
      <div className="workspace-footer">
        <span>
          <span className="footer-dot" />{" "}
          {loading
            ? "Conectando con tu proyecto"
            : unavailable
              ? "Conexión pendiente"
              : "Datos de tu espacio de trabajo"}
        </span>
        <span>
          {completed.length
            ? `${completed.length} con actividad reciente`
            : "Diseñado para explorar. Construido para decidir."}
          {artifacts.length > 0
            ? ` · ${artifacts.length} archivos del ${nav.chat.toLowerCase()}`
            : ""}
        </span>
      </div>
    </div>
  );
}
export function Analyses({
  runs,
  loading,
  onOpen,
  onNewRun,
}: {
  runs: RunSummary[];
  loading: boolean;
  onOpen: (id: string) => void;
  onNewRun: () => void;
}) {
  const { nouns, nav } = useExperience();
  const byName = useWorkflowMap();
  const [filter, setFilter] = useState("all"),
    [query, setQuery] = useState("");
  const filtered = runs.filter(
    (r) =>
      (filter === "all" ||
        (filter === "active"
          ? ["running", "pending"].includes(r.status)
          : r.status === filter)) &&
      `${runTitle(r, byName.get(r.workflow))} ${r.run_id}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">DEL OBJETIVO AL RESULTADO</span>
          <h1>{nav.runs}</h1>
          <p>Avances, revisiones y resultados. Todo en un mismo lugar.</p>
        </div>
        <button className="btn primary" onClick={onNewRun}>
          <Icon name="plus" size={17} /> Crear {nouns.run.singular}
        </button>
      </div>
      <section className="surface">
        <div className="list-toolbar">
          <div className="filter-tabs" aria-label="Filtrar por estado">
            {[
              ["all", "Todos"],
              ["active", "En curso"],
              ["awaiting_gate", "Por revisar"],
              ["done", "Completados"],
              ["failed", "Con errores"],
            ].map(([id, label]) => (
              <button
                key={id}
                aria-pressed={filter === id}
                className={filter === id ? "active" : ""}
                onClick={() => setFilter(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="search-field">
            <Icon name="search" size={17} />
            <input
              aria-label={`Buscar ${nouns.run.plural}`}
              placeholder={`Buscar ${nouns.run.plural}…`}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
        </div>
        {loading ? (
          <div className="skeleton-list">
            <i />
            <i />
            <i />
          </div>
        ) : filtered.length ? (
          <RunRows runs={filtered} onOpen={onOpen} />
        ) : (
          <EmptyState
            icon="search"
            title={
              runs.length
                ? "No encontramos coincidencias"
                : `Todavía no hay ${nouns.run.plural}`
            }
            text={
              runs.length
                ? "Prueba otro término o cambia los filtros."
                : `Elige un proceso, conecta tus ${nouns.resource.plural} y sigue cada paso.`
            }
            action={
              runs.length
                ? "Restablecer filtros"
                : `Crear ${nouns.run.singular}`
            }
            onAction={
              runs.length
                ? () => {
                    setQuery("");
                    setFilter("all");
                  }
                : onNewRun
            }
          />
        )}
        <div className="table-footer">
          {filtered.length} de {runs.length} · últimos 100
        </div>
      </section>
    </div>
  );
}
export function DataLibrary({
  project,
  datasets,
  loading,
  onRefresh,
  onNewRun,
}: {
  project: string;
  datasets: Dataset[];
  loading: boolean;
  onRefresh: () => void;
  onNewRun: (dataset: string) => void;
}) {
  const { nouns, nav, resources } = useExperience();
  const ref = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [dragging, setDragging] = useState(false),
    [remove, setRemove] = useState<string | null>(null),
    [query, setQuery] = useState("");
  const accept = resources.formats.map((f) => `.${f}`).join(",");
  async function upload(files: FileList | null) {
    if (!files?.length || busy) return;
    setBusy(true);
    setError("");
    try {
      for (const file of Array.from(files))
        await api.uploadDataset(project, file);
      onRefresh();
    } catch {
      setError(
        "No se completó la carga. Revisa la conexión y vuelve a intentarlo.",
      );
    } finally {
      setBusy(false);
      if (ref.current) ref.current.value = "";
    }
  }
  async function deleteFile() {
    if (!remove) return;
    setBusy(true);
    setError("");
    try {
      await api.deleteDataset(project, remove);
      setRemove(null);
      onRefresh();
    } catch {
      setError("No se pudo eliminar el archivo.");
    } finally {
      setBusy(false);
    }
  }
  const filtered = datasets.filter((d) =>
    d.name.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">EL ORIGEN DE CADA RESULTADO</span>
          <h1>{nav.resources}</h1>
          <p>{resources.intro}</p>
        </div>
        <button
          className="btn primary"
          disabled={busy}
          onClick={() => ref.current?.click()}
        >
          <Icon name="upload" size={17} />
          {busy ? "Subiendo…" : "Subir archivos"}
        </button>
      </div>
      <input
        ref={ref}
        type="file"
        hidden
        multiple
        accept={accept || undefined}
        onChange={(e) => void upload(e.target.files)}
      />
      {error && (
        <div className="err-banner" role="alert">
          {error}
        </div>
      )}
      <div
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(e.dataTransfer.files);
        }}
      >
        <span className="upload-symbol">
          <Icon name="upload" size={28} />
        </span>
        <div>
          <h2>
            {busy
              ? "Preparando tus archivos…"
              : `Aquí empieza cada ${nouns.run.singular}.`}
          </h2>
          <p>
            Arrastra tus archivos o{" "}
            <button
              className="text-button"
              disabled={busy}
              onClick={() => ref.current?.click()}
            >
              explora tu equipo
            </button>
            .
          </p>
          <span className="file-format">
            {[
              ...resources.formats.map((f) => f.toUpperCase()),
              "ARCHIVOS DE TU PROYECTO",
            ].join(" · ")}
          </span>
        </div>
      </div>
      <section className="surface">
        <div className="section-heading">
          <h2>
            {cap(nouns.resource.plural)} disponibles{" "}
            <span className="count-label">{datasets.length}</span>
          </h2>
          <label className="search-field">
            <Icon name="search" size={17} />
            <input
              aria-label="Buscar archivo"
              placeholder="Buscar archivo…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
        </div>
        {loading ? (
          <div className="skeleton-list">
            <i />
            <i />
          </div>
        ) : filtered.length ? (
          <div className="dataset-list">
            {filtered.map((d) => (
              <div className="dataset-row" key={d.name}>
                <span className="file-type-icon">
                  <Icon name="data" size={22} />
                </span>
                <div>
                  <strong>{d.name}</strong>
                  <small>
                    {d.name.split(".").pop()?.toUpperCase()} · {bytes(d.bytes)}
                  </small>
                </div>
                <button
                  className="btn ghost sm"
                  onClick={() => onNewRun(d.name)}
                >
                  Usar en {nouns.run.singular}
                  <Icon name="arrow" size={15} />
                </button>
                <button
                  className="icon-btn danger"
                  onClick={() => setRemove(d.name)}
                  aria-label={`Eliminar ${d.name}`}
                >
                  <Icon name="trash" size={17} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            icon="data"
            title={
              query
                ? "Sin coincidencias"
                : `Un espacio listo para tus ${nouns.resource.plural}`
            }
            text={
              query
                ? "Prueba con otro nombre de archivo."
                : `Tus ${nouns.resource.plural} estarán disponibles aquí en cuanto los subas.`
            }
          />
        )}
      </section>
      {remove && (
        <Dialog
          title={`Eliminar ${nouns.resource.singular}`}
          onClose={() => {
            if (!busy) setRemove(null);
          }}
        >
          <div className="dialog-body stack">
            <p>
              Se eliminará <strong>{remove}</strong> de la biblioteca del
              proyecto. Las copias usadas en {nouns.run.plural} existentes se
              conservan.
            </p>
            <div className="dialog-actions">
              <button
                className="btn"
                onClick={() => setRemove(null)}
                disabled={busy}
              >
                Conservar archivo
              </button>
              <button
                className="btn danger"
                onClick={deleteFile}
                disabled={busy}
              >
                {busy ? "Eliminando…" : "Eliminar archivo"}
              </button>
            </div>
            {error && (
              <div role="alert" className="err-banner">
                {error}
              </div>
            )}
          </div>
        </Dialog>
      )}
    </div>
  );
}
export function Deliverables({
  project,
  runs,
  artifacts,
  onOpen,
  onNewRun,
}: {
  project: string;
  runs: RunSummary[];
  artifacts: Artifact[];
  onOpen: (id: string) => void;
  onNewRun: () => void;
}) {
  const { title, nouns, nav } = useExperience();
  const byName = useWorkflowMap();
  // A file a chat wrote opens in the app's viewer, like a run's deliverable.
  const [open, setOpen] = useState<FocusTarget | null>(null);
  const completed = runs.filter((r) => r.status === "done");
  const label = (wf?: WorkflowShape) =>
    (wf ? workflowTitle(wf) : nouns.deliverable.singular).toUpperCase();
  return (
    <div className="workspace-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">RESULTADOS QUE PUEDES EXPLORAR</span>
          <h1>{nav.deliverables}</h1>
          <p>
            Revisa {nouns.deliverable.plural} de cada {nouns.run.singular} y los
            archivos de tus conversaciones.
          </p>
        </div>
        <button className="btn primary" onClick={onNewRun}>
          <Icon name="plus" size={17} /> Crear {nouns.run.singular}
        </button>
      </div>
      {completed.length > 0 ? (
        <div className="deliverable-grid">
          {completed.map((r) => {
            const wf = byName.get(r.workflow);
            const name = runTitle(r, wf);
            return (
              <article className="report-card" key={r.run_id}>
                <button
                  className="report-preview"
                  onClick={() => onOpen(r.run_id)}
                  aria-label={`Ver resultados de ${name}`}
                >
                  <span className="report-paper">
                    <span className="report-paper-label">{label(wf)}</span>
                    <strong>{name}</strong>
                    <span className="paper-lines">
                      <i />
                      <i />
                      <i />
                    </span>
                    <span className="paper-process">
                      <Icon name="data" />
                      <Icon name="arrow" size={15} />
                      <Icon name={iconName(wf?.icon, "chart")} />
                      <Icon name="arrow" size={15} />
                      <Icon name="file" />
                    </span>
                    <span className="report-paper-footer">
                      {title.toUpperCase()} · {r.steps_done} PASOS COMPLETADOS
                    </span>
                  </span>
                </button>
                <div className="report-card-body">
                  <StatusPill status={r.status} />
                  <h2>{name}</h2>
                  <p>
                    {ago(r.finished_at)} ·{" "}
                    {r.cost_usd == null
                      ? "Costo no disponible"
                      : `${cost(r.cost_usd)} registrado`}
                  </p>
                  <div className="report-actions">
                    <button
                      className="text-button"
                      onClick={() => onOpen(r.run_id)}
                    >
                      Explorar resultados
                      <Icon name="arrow" size={15} />
                    </button>
                    <a
                      className="icon-btn"
                      href={api.downloadUrl(r.run_id)}
                      aria-label={`Descargar archivos de ${name}`}
                    >
                      <Icon name="download" size={18} />
                    </a>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <section className="surface">
          <EmptyState
            icon="file"
            title="Las respuestas merecen un buen lugar"
            text={`Al completar ${nouns.run.singular} podrás explorar aquí sus ${nouns.deliverable.plural}, archivos y evidencia.`}
            action={`Crear ${nouns.run.singular}`}
            onAction={onNewRun}
          />
        </section>
      )}
      {artifacts.length > 0 && (
        <section className="surface project-files">
          <div className="section-heading">
            <div>
              <h2>Archivos del {nav.chat.toLowerCase()}</h2>
              <p>Generados en las conversaciones de este proyecto.</p>
            </div>
          </div>
          {artifacts.map((f) => (
            <button
              className="dataset-row"
              key={f.path}
              onClick={() => setOpen({ kind: "file", path: f.path })}
              title={f.path}
            >
              <span className="file-type-icon">
                <Icon name="file" />
              </span>
              <div>
                <strong>{basename(f.path)}</strong>
                <small>
                  {bytes(f.bytes)} · {ago(f.mtime)}
                </small>
              </div>
              <Icon name="external" size={17} />
            </button>
          ))}
        </section>
      )}
      {open && (
        <FocusOverlay
          source={projectSource(project)}
          target={open}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}
