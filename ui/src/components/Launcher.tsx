import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { bytes } from "../lib/format";
import type { WorkflowShape } from "../lib/types";

interface Props {
  project?: string;
  session?: string;
  onLaunched: (runId: string) => void;
}

export function Launcher({ project = "default", session = "", onLaunched }: Props) {
  const [workflows, setWorkflows] = useState<WorkflowShape[] | null>(null);
  const [datasets, setDatasets] = useState<{ name: string; bytes: number }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string>("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    api
      .cartridges()
      .then((r) => {
        setWorkflows(r.workflows);
        if (r.workflows[0]) setPicked(r.workflows[0].name);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const loadDatasets = () => api.datasets(project).then(setDatasets).catch(() => {});
  useEffect(() => {
    loadDatasets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project]);

  const wf = useMemo(() => workflows?.find((w) => w.name === picked) ?? null, [workflows, picked]);

  useEffect(() => {
    if (!wf) return;
    const init: Record<string, string> = {};
    for (const [key, spec] of Object.entries(wf.inputs)) {
      if (spec.default != null) init[key] = String(spec.default);
    }
    setValues(init);
  }, [wf]);

  const dataInput = wf?.data_input ?? null;

  async function upload(file: File) {
    try {
      const { name } = await api.uploadDataset(project, file);
      await loadDatasets();
      if (dataInput) setValues((v) => ({ ...v, [dataInput]: `data/${name}` }));
    } catch (e) {
      setError(String(e));
    }
  }

  async function launch() {
    if (!wf) return;
    setBusy(true);
    setError(null);
    try {
      // The data input is a project dataset path; seeding puts the file in the
      // run, so no per-run upload is needed.
      const inputs: Record<string, unknown> = {};
      for (const [key, val] of Object.entries(values)) {
        if (val !== "") inputs[key] = val;
      }
      const { run_id } = await api.create(wf.name, inputs, { project, session });
      await api.start(run_id);
      onLaunched(run_id);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  const missing =
    !wf ||
    Object.entries(wf.inputs).some(([k, s]) => s.required && !values[k]);

  return (
    <div className="col-body stack">
      {error && <div className="err-banner">{error}</div>}
      {workflows === null && !error && <div className="empty">Loading workflows…</div>}

      {workflows && (
        <div className="field">
          <label>Workflow</label>
          <select value={picked} onChange={(e) => setPicked(e.target.value)}>
            {workflows.map((w) => (
              <option key={w.name} value={w.name}>
                {w.name} — {w.cartridge}
              </option>
            ))}
          </select>
        </div>
      )}

      {wf && (
        <>
          <p className="muted" style={{ fontSize: 13, marginTop: -4 }}>{wf.description}</p>
          <div className="row wrap faint" style={{ fontSize: 12 }}>
            <span>env: {wf.env}</span>
            <span>·</span>
            <span>{wf.steps.length} steps</span>
            <span>·</span>
            <span>{wf.personas.join(", ")}</span>
          </div>

          {Object.entries(wf.inputs).map(([key, spec]) => {
            if (key === dataInput) return null;
            return (
              <div className="field" key={key}>
                <label>
                  {key} {spec.required && <span className="req">*</span>}
                  <span className="faint"> · {spec.type}</span>
                </label>
                {spec.options ? (
                  <select
                    value={values[key] ?? ""}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                  >
                    <option value="">—</option>
                    {spec.options.map((o) => (
                      <option key={o} value={o}>{o}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    value={values[key] ?? ""}
                    placeholder={spec.guess != null ? String(spec.guess) : ""}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                  />
                )}
              </div>
            );
          })}

          {dataInput && (
            <div className="field">
              <label>
                {dataInput} <span className="req">*</span>
                <span className="faint"> · dataset del proyecto</span>
              </label>
              {datasets.length > 0 && (
                <select
                  value={values[dataInput] ?? ""}
                  onChange={(e) => setValues((v) => ({ ...v, [dataInput]: e.target.value }))}
                >
                  <option value="">— elegir dataset —</option>
                  {datasets.map((d) => (
                    <option key={d.name} value={`data/${d.name}`}>
                      {d.name} ({bytes(d.bytes)})
                    </option>
                  ))}
                </select>
              )}
              <button
                className="btn ghost sm"
                style={{ marginTop: 6 }}
                onClick={() => fileRef.current?.click()}
              >
                ↥ Subir dataset al proyecto
              </button>
              <input
                ref={fileRef}
                type="file"
                hidden
                onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
              />
              {datasets.length === 0 && (
                <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
                  Aún no hay datasets en este proyecto — sube uno.
                </div>
              )}
            </div>
          )}

          <button className="btn primary" disabled={!!missing || busy} onClick={launch}>
            {busy ? "Launching…" : "Launch run"}
          </button>
        </>
      )}
    </div>
  );
}
