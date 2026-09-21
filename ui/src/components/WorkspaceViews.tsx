import { useMemo, useRef, useState } from "react";
import { api, projectSource } from "../lib/api";
import { ago, basename, bytes, cost } from "../lib/format";
import { FocusOverlay, type FocusTarget } from "./FocusOverlay";
import type { Project, RunSummary, WorkflowShape } from "../lib/types";
import type { Artifact, Dataset } from "../lib/useWorkspace";
import {
  cap,
  runTitle,
  stepTitle,
  useExperience,
  useWorkspaceConfig,
  workflowSummary,
  workflowTitle,
  workflowsFor,
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
// The home has two shapes. A project with nothing in it gets a compact start:
// the goal, the resources to bring, the workflows that can be run. A project
// with activity gets what needs attention, in that order — reviews first, then
// work in flight, then what came out of it, then the resources behind it. No
// number on this page is computed from anything but what the server returned,
// and no deliverable is drawn as a preview the file itself has not produced.
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
  onNewRun: (workflow?: string) => void;
  onChat: () => void;
  onRuns: () => void;
  onData: () => void;
  onOpenRun: (id: string) => void;
}) {
  const { home, nouns, nav, tagline, resources } = useExperience();
  const { workflows, cartridge } = useWorkspaceConfig();
  const byName = useWorkflowMap();
  const offered = workflowsFor(workflows, cartridge);
  const pending = runs.filter((r) => r.status === "awaiting_gate");
  const delivered = runs.filter((r) => r.status === "done");
  const quiet =
    !loading &&
    !unavailable &&
    runs.length === 0 &&
    datasets.length === 0 &&
    artifacts.length === 0;

  if (loading)
    return (
      <div className="workspace-page overview-page">
        <div className="page-heading">
          <div>
            <h1>{nav.overview}</h1>
            <p>{project?.goal || home.intro}</p>
          </div>
        </div>
        <div className="surface">
          <div className="skeleton-list" aria-label="Cargando">
            <i />
            <i />
            <i />
          </div>
        </div>
      </div>
    );

  if (quiet)
    return (
      <div className="workspace-page overview-page">
        <section className="start">
          <span className="eyebrow">{home.eyebrow}</span>
          <h1 className="start-headline">{home.headline}</h1>
          <p className="start-intro">{project?.goal || home.intro}</p>
          <p className="start-tagline">{tagline}</p>
          {/* The cartridge's declared stages, written out. They used to be the
              labels on an orbital illustration; the words were the only part of
              it carrying information. */}
          {home.stages.length > 0 && (
            <p className="start-stages">{home.stages.join(" → ")}</p>
          )}
          <div className="start-actions">
            <button className="btn primary" onClick={() => onNewRun()}>
              Crear {nouns.run.singular}
              <Icon name="arrow" size={16} />
            </button>
            <button className="text-button" onClick={onChat}>
              <Icon name="spark" size={15} />
              {nav.chat}
            </button>
          </div>
        </section>
        <div className="start-grid">
          <section className="start-block">
            <h2>{cap(nouns.resource.plural)}</h2>
            <p>{resources.intro}</p>
            {resources.formats.length > 0 && (
              <p className="start-formats">
                {resources.formats.map((f) => f.toUpperCase()).join(" · ")}
              </p>
            )}
            <button className="text-button" onClick={onData}>
              {nav.resources}
              <Icon name="arrow" size={15} />
            </button>
          </section>
          <section className="start-block">
            <h2>{nav.runs}</h2>
            {offered.length === 0 ? (
              <p>No hay procesos configurados para este proyecto.</p>
            ) : (
              <ul className="start-workflows">
                {offered.map((w) => (
                  <li key={w.name}>
                    <button onClick={() => onNewRun(w.name)}>
                      <span>
                        <strong>{workflowTitle(w)}</strong>
                        <small>
                          {workflowSummary(w) ||
                            `${w.steps.length} pasos · ${w.personas.length} ${
                              w.personas.length === 1
                                ? "especialista"
                                : "especialistas"
                            }`}
                        </small>
                      </span>
                      <Icon name="arrow" size={15} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
        {home.steps.length > 0 && (
          <ol className="start-steps">
            {home.steps.slice(0, 3).map((step, i) => (
              <li key={step.title}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                <strong>{step.title}</strong>
                <small>{step.text}</small>
              </li>
            ))}
          </ol>
        )}
      </div>
    );

  return (
    <div className="workspace-page overview-page">
      <div className="page-heading">
        <div>
          <h1>{nav.overview}</h1>
          <p>{project?.goal || home.intro}</p>
        </div>
        <button className="btn primary" onClick={() => onNewRun()}>
          <Icon name="plus" size={17} /> Crear {nouns.run.singular}
        </button>
      </div>

      {pending.length > 0 && (
        <section className="attention" aria-labelledby="pending-reviews">
          <div className="attention-heading">
            <h2 id="pending-reviews">
              <Icon name="clock" size={17} />
              Pendientes de revisión
            </h2>
            <small>
              {pending.length} de {runs.length} {nouns.run.plural}
            </small>
          </div>
          <ul className="review-list">
            {pending.map((r) => {
              const wf = byName.get(r.workflow);
              const shape = wf?.steps.find((s) => s.id === r.awaiting?.step);
              const produces = r.awaiting?.produces ?? [];
              return (
                <li key={r.run_id}>
                  <div className="review-what">
                    <strong>{runTitle(r, wf)}</strong>
                    <span className="review-stage">
                      {shape ? stepTitle(shape) : (r.awaiting?.step ?? "")}
                      {/* The two counts the server reports, restated. Which
                          ordinal the gated step holds is not one of them: a
                          declared DAG can finish steps out of reading order. */}
                      <small>
                        {" "}
                        · {r.steps_done} de {r.steps_total} pasos completados
                      </small>
                    </span>
                    {produces.length > 0 && (
                      <p className="review-produces">
                        Este paso produce{" "}
                        {produces.map((path) => (
                          <code key={path}>{basename(path)}</code>
                        ))}
                      </p>
                    )}
                    {r.awaiting?.prompt && (
                      <p className="review-prompt">{r.awaiting.prompt}</p>
                    )}
                  </div>
                  <button
                    className="btn primary sm"
                    onClick={() => onOpenRun(r.run_id)}
                  >
                    Revisar
                    <Icon name="arrow" size={15} />
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <section className="surface">
        <div className="section-heading">
          <div>
            <h2>{cap(nouns.run.plural)}</h2>
            <p>Lo más reciente de este proyecto.</p>
          </div>
          <button className="text-button" onClick={onRuns}>
            Ver {nav.runs.toLowerCase()}
            <Icon name="arrow" size={15} />
          </button>
        </div>
        {runs.length ? (
          <RunRows runs={runs.slice(0, 5)} onOpen={onOpenRun} />
        ) : (
          <EmptyState
            icon="chart"
            title={`Todavía no hay ${nouns.run.plural}`}
            text={`Elige un proceso y sigue cada paso desde aquí.`}
            action={`Crear ${nouns.run.singular}`}
            onAction={() => onNewRun()}
          />
        )}
      </section>

      <div className="overview-lower">
        <section className="surface">
          <div className="section-heading">
            <div>
              <h2>{cap(nouns.deliverable.plural)}</h2>
              <p>Lo que ya puedes leer y descargar.</p>
            </div>
          </div>
          {delivered.length === 0 && artifacts.length === 0 ? (
            <p className="section-note">
              Aquí aparecerán {nouns.deliverable.plural} cuando un{" "}
              {nouns.run.singular} termine.
            </p>
          ) : (
            <ul className="plain-list">
              {delivered.slice(0, 4).map((r) => {
                const wf = byName.get(r.workflow);
                return (
                  <li key={r.run_id}>
                    <button onClick={() => onOpenRun(r.run_id)}>
                      <Icon name="file" size={17} />
                      <span>
                        <strong>{runTitle(r, wf)}</strong>
                        <small>
                          {wf ? workflowTitle(wf) : r.workflow} ·{" "}
                          {ago(r.finished_at)}
                        </small>
                      </span>
                      <Icon name="chevron" size={15} />
                    </button>
                  </li>
                );
              })}
              {artifacts.slice(0, 3).map((f) => (
                <li key={f.path}>
                  <span className="plain-row">
                    <Icon name="file" size={17} />
                    <span>
                      <strong>{basename(f.path)}</strong>
                      <small>
                        {nav.chat} · {bytes(f.bytes)} · {ago(f.mtime)}
                      </small>
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
        <section className="surface">
          <div className="section-heading">
            <div>
              <h2>{cap(nouns.resource.plural)}</h2>
              <p>{resources.intro}</p>
            </div>
            <button className="text-button" onClick={onData}>
              {nav.resources}
              <Icon name="arrow" size={15} />
            </button>
          </div>
          {datasets.length === 0 ? (
            <p className="section-note">
              Todavía no has subido {nouns.resource.plural}.
            </p>
          ) : (
            <ul className="plain-list">
              {datasets.slice(0, 5).map((d) => (
                <li key={d.name}>
                  <span className="plain-row">
                    <Icon name="data" size={17} />
                    <span>
                      <strong>{d.name}</strong>
                      <small>{bytes(d.bytes)}</small>
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
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
  const { nouns, nav } = useExperience();
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
              // No drawn preview: a card shows what the run recorded, and the
              // file itself is read in the run's own reading surface.
              <article className="report-card" key={r.run_id}>
                <div className="report-card-body">
                  <div className="report-card-top">
                    <span className="report-kind">
                      <Icon name={iconName(wf?.icon, "file")} size={17} />
                      {label(wf)}
                    </span>
                    <StatusPill status={r.status} />
                  </div>
                  <h2>{name}</h2>
                  <dl className="report-facts">
                    <div>
                      <dt>Completado</dt>
                      <dd>{ago(r.finished_at) || "Sin fecha registrada"}</dd>
                    </div>
                    <div>
                      <dt>Pasos</dt>
                      <dd>
                        {r.steps_done} de {r.steps_total}
                      </dd>
                    </div>
                    <div>
                      <dt>Costo</dt>
                      <dd>
                        {r.cost_usd == null
                          ? "No disponible"
                          : cost(r.cost_usd)}
                      </dd>
                    </div>
                  </dl>
                  <div className="report-actions">
                    <button
                      className="btn sm"
                      onClick={() => onOpen(r.run_id)}
                      aria-label={`Abrir ${nouns.deliverable.plural} de ${name}`}
                    >
                      Abrir {nouns.deliverable.plural}
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
