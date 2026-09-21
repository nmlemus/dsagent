import { useEffect, useMemo, useRef, useState } from "react";
import { api, detailOf } from "../lib/api";
import { bytes } from "../lib/format";
import {
  cap,
  stepTitle,
  useWorkspaceConfig,
  workflowSummary,
  workflowTitle,
  workflowsFor,
} from "../lib/experience";
import { Icon, iconName } from "./Icon";
// The form is generated from what the workflow declares: its inputs, their
// labels and placeholders, its steps and gates. Nothing here knows an input or
// a workflow by name — a cartridge for another domain renders the same way.
const fieldTitle = (key: string, label: string | null) =>
  label || key.replace(/_/g, " ");
export function Launcher({
  project = "default",
  cartridge = "",
  session = "",
  initialDataset,
  initialWorkflow,
  onLaunched,
}: {
  project?: string;
  cartridge?: string;
  session?: string;
  initialDataset?: string;
  // Preselects one of this cartridge's workflows (the home offers them by
  // name). Ignored when the project's cartridge does not declare it.
  initialWorkflow?: string;
  onLaunched: (runId: string) => void;
}) {
  const config = useWorkspaceConfig();
  const { nouns, resources } = config.experience;
  const workflows = useMemo(
    () => workflowsFor(config.workflows, cartridge),
    [config.workflows, cartridge],
  );
  const [datasets, setDatasets] = useState<{ name: string; bytes: number }[]>(
    [],
  );
  const [error, setError] = useState<string | null>(null),
    [picked, setPicked] = useState(""),
    [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false),
    [uploading, setUploading] = useState(false),
    [review, setReview] = useState(false),
    [createdId, setCreatedId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    setPicked((current) =>
      workflows.some((w) => w.name === current)
        ? current
        : workflows.some((w) => w.name === initialWorkflow)
          ? (initialWorkflow ?? "")
          : (workflows[0]?.name ?? ""),
    );
  }, [workflows, initialWorkflow]);
  useEffect(() => {
    let live = true;
    api
      .datasets(project)
      .then((d) => {
        if (live) setDatasets(d);
      })
      .catch(() => {
        if (live) setError(`No se pudieron cargar ${nouns.resource.plural}.`);
      });
    return () => {
      live = false;
    };
  }, [project, nouns.resource.plural]);
  const wf = useMemo(
    () => workflows.find((w) => w.name === picked) ?? null,
    [workflows, picked],
  );
  useEffect(() => {
    if (!wf) return;
    const initial: Record<string, string> = {};
    for (const [key, spec] of Object.entries(wf.inputs))
      if (spec.default != null && spec.default !== "None")
        initial[key] = String(spec.default);
    if (initialDataset && wf.data_input)
      initial[wf.data_input] = `data/${initialDataset}`;
    setValues(initial);
    setReview(false);
  }, [wf, initialDataset]);
  const dataInput = wf?.data_input ?? null;
  const accept = resources.formats.map((f) => `.${f}`).join(",");
  async function upload(file: File) {
    setUploading(true);
    setError(null);
    try {
      const { name } = await api.uploadDataset(project, file);
      setDatasets(await api.datasets(project));
      if (dataInput) setValues((v) => ({ ...v, [dataInput]: `data/${name}` }));
    } catch {
      setError("No se pudo subir el archivo. Vuelve a intentarlo.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }
  async function launch() {
    if (!wf || busy) return;
    setBusy(true);
    setError(null);
    try {
      const inputs: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(values)) {
        if (value.trim() !== "") {
          const type = wf.inputs[key]?.type;
          inputs[key] =
            type === "int" ||
            type === "float" ||
            type === "number" ||
            type === "integer"
              ? Number(value)
              : type === "bool" || type === "boolean"
                ? value === "true"
                : value;
        }
      }
      const id =
        createdId ??
        (await api.create(wf.name, inputs, { project, session })).run_id;
      setCreatedId(id);
      await api.start(id);
      onLaunched(id);
    } catch (e) {
      // Show what the server refused and why. The generic sentence alone sent
      // operators looking for a fault in their own form when the server had
      // already explained itself (a replay that cannot run this workflow, a
      // workflow from another cartridge, a missing input). A refusal that
      // carries a reason is not worth retrying, so it does not offer to.
      const detail = detailOf(e);
      setError(
        detail
          ? `No se pudo iniciar: ${detail}`
          : "No se pudo iniciar. Puedes reintentar; conservamos tu configuración.",
      );
      setBusy(false);
    }
  }
  const missing =
    !wf ||
    Object.entries(wf.inputs).some(
      ([k, s]) => s.required && !values[k]?.trim(),
    );
  const humanGates = wf
    ? wf.steps.filter((s) => s.gate?.kind === "human").length
    : 0;
  return (
    <div className="col-body stack">
      {error && (
        <div className="err-banner" role="alert">
          {error}
        </div>
      )}
      {config.loading && (
        <div className="view-loading" role="status">
          Cargando procesos…
        </div>
      )}
      {!config.loading && workflows.length === 0 && (
        <p className="muted">
          No hay procesos configurados para este proyecto.
        </p>
      )}
      {workflows.length > 0 && !review && (
        <div>
          <span className="eyebrow">
            01 · ELIGE EL TIPO DE {nouns.run.singular.toUpperCase()}
          </span>
          <div className="workflow-options">
            {workflows.map((w) => (
              <button
                key={w.name}
                className={`workflow-option ${picked === w.name ? "selected" : ""}`}
                aria-pressed={picked === w.name}
                onClick={() => setPicked(w.name)}
              >
                <Icon name={iconName(w.icon)} />
                <strong>{workflowTitle(w)}</strong>
                <span>
                  {w.steps.length} pasos · {w.personas.length}{" "}
                  {w.personas.length === 1 ? "especialista" : "especialistas"}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
      {wf && (
        <>
          {review ? (
            <>
              <div className="workflow-summary">
                <span className="eyebrow">LISTO PARA COMENZAR</span>
                <h3>{workflowTitle(wf)}</h3>
                <dl className="launch-review">
                  {Object.entries(values)
                    .filter(([, v]) => v)
                    .map(([k, v]) => (
                      <div key={k}>
                        <dt>{fieldTitle(k, wf.inputs[k]?.label ?? null)}</dt>
                        <dd>{v}</dd>
                      </div>
                    ))}
                </dl>
              </div>
              <div>
                <span className="eyebrow">EL RECORRIDO</span>
                <ol className="launch-steps">
                  {wf.steps.map((s, i) => (
                    <li key={s.id}>
                      <span>{String(i + 1).padStart(2, "0")}</span>
                      <strong>{stepTitle(s)}</strong>
                      {s.gate && (
                        <small>
                          <Icon name="shield" size={12} />
                          {s.gate.kind === "human"
                            ? "Revisión humana"
                            : "Validación automática"}
                        </small>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
              <p className="launcher-note">
                Al iniciar se ejecutará el proceso con los{" "}
                {nouns.resource.plural} elegidos. Puede consumir recursos y
                llamadas a modelos; el costo final depende de la ejecución.
              </p>
              <div className="dialog-actions">
                <button
                  className="btn ghost"
                  disabled={busy || !!createdId}
                  onClick={() => setReview(false)}
                >
                  Editar configuración
                </button>
                <button
                  className="btn primary"
                  disabled={busy}
                  onClick={launch}
                >
                  {busy
                    ? `Iniciando ${nouns.run.singular}…`
                    : createdId
                      ? "Reintentar inicio"
                      : `Iniciar ${nouns.run.singular}`}
                  <Icon name="arrow" size={16} />
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="workflow-summary">
                <p>{workflowSummary(wf)}</p>
                <div className="row wrap">
                  <span>
                    {humanGates}{" "}
                    {humanGates === 1
                      ? "punto de revisión humana"
                      : "puntos de revisión humana"}
                  </span>
                  {wf.badge && <span className="chip">{wf.badge}</span>}
                </div>
              </div>
              <span className="eyebrow" style={{ marginBottom: 0 }}>
                02 · DEFINE EL PUNTO DE PARTIDA
              </span>
              <form
                className="stack"
                onSubmit={(e) => {
                  e.preventDefault();
                  setReview(true);
                }}
              >
                {Object.entries(wf.inputs).map(([key, spec]) => {
                  if (key === dataInput) return null;
                  const numeric = [
                    "int",
                    "float",
                    "number",
                    "integer",
                  ].includes(spec.type);
                  return (
                    <div className="field" key={key}>
                      <label htmlFor={`launch-${key}`}>
                        {fieldTitle(key, spec.label)}
                        {spec.required && <span className="req"> *</span>}
                      </label>
                      {spec.options ||
                      ["bool", "boolean"].includes(spec.type) ? (
                        <select
                          id={`launch-${key}`}
                          value={values[key] ?? ""}
                          required={spec.required}
                          onChange={(e) =>
                            setValues((v) => ({ ...v, [key]: e.target.value }))
                          }
                        >
                          <option value="">Seleccionar…</option>
                          {(spec.options ?? ["true", "false"]).map((o) => (
                            <option key={o} value={o}>
                              {o === "true" ? "Sí" : o === "false" ? "No" : o}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          id={`launch-${key}`}
                          type={numeric ? "number" : "text"}
                          step={
                            ["int", "integer"].includes(spec.type) ? 1 : "any"
                          }
                          value={values[key] ?? ""}
                          required={spec.required}
                          placeholder={
                            spec.guess != null
                              ? String(spec.guess)
                              : (spec.placeholder ?? "")
                          }
                          onChange={(e) =>
                            setValues((v) => ({ ...v, [key]: e.target.value }))
                          }
                        />
                      )}
                      {spec.help && (
                        <small className="faint">{spec.help}</small>
                      )}
                    </div>
                  );
                })}
                {dataInput && (
                  <div className="field">
                    <label htmlFor="launch-data">
                      {fieldTitle(
                        dataInput,
                        wf.inputs[dataInput]?.label ??
                          cap(nouns.resource.singular),
                      )}
                      {wf.inputs[dataInput]?.required && (
                        <span className="req"> *</span>
                      )}
                    </label>
                    <select
                      id="launch-data"
                      value={values[dataInput] ?? ""}
                      required={wf.inputs[dataInput]?.required}
                      onChange={(e) =>
                        setValues((v) => ({
                          ...v,
                          [dataInput]: e.target.value,
                        }))
                      }
                    >
                      <option value="">
                        Selecciona un archivo del proyecto
                      </option>
                      {datasets.map((d) => (
                        <option key={d.name} value={`data/${d.name}`}>
                          {d.name} · {bytes(d.bytes)}
                        </option>
                      ))}
                    </select>
                    {wf.inputs[dataInput]?.help && (
                      <small className="faint">
                        {wf.inputs[dataInput].help}
                      </small>
                    )}
                    <button
                      type="button"
                      className="text-button"
                      style={{ justifyContent: "flex-start" }}
                      disabled={uploading}
                      onClick={() => fileRef.current?.click()}
                    >
                      <Icon name="upload" size={15} />
                      {uploading
                        ? "Subiendo archivo…"
                        : "Subir un archivo nuevo"}
                    </button>
                    <input
                      ref={fileRef}
                      type="file"
                      hidden
                      accept={accept || undefined}
                      onChange={(e) =>
                        e.target.files?.[0] && void upload(e.target.files[0])
                      }
                    />
                  </div>
                )}
                <div className="dialog-actions">
                  <button
                    className="btn primary"
                    disabled={missing || uploading}
                  >
                    Revisar configuración
                    <Icon name="arrow" size={17} />
                  </button>
                </div>
              </form>
            </>
          )}
        </>
      )}
    </div>
  );
}
