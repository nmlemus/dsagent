import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import type { WorkflowShape } from "../lib/types";

interface Props {
  project?: string;
  session?: string;
  onLaunched: (runId: string) => void;
}

export function Launcher({ project = "default", session = "", onLaunched }: Props) {
  const [workflows, setWorkflows] = useState<WorkflowShape[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string>("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .cartridges()
      .then((r) => {
        setWorkflows(r.workflows);
        if (r.workflows[0]) setPicked(r.workflows[0].name);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const wf = useMemo(
    () => workflows?.find((w) => w.name === picked) ?? null,
    [workflows, picked],
  );

  // Reset field values when the chosen workflow changes.
  useEffect(() => {
    if (!wf) return;
    const init: Record<string, string> = {};
    for (const [key, spec] of Object.entries(wf.inputs)) {
      if (spec.default != null) init[key] = String(spec.default);
    }
    setValues(init);
    setFile(null);
  }, [wf]);

  const dataInput = wf?.data_input ?? null;

  async function launch() {
    if (!wf) return;
    setBusy(true);
    setError(null);
    try {
      // Non-file inputs go in the create body; the data file is uploaded to the
      // run, which points its input at the stored path (api.py upload_input).
      const inputs: Record<string, unknown> = {};
      for (const [key, val] of Object.entries(values)) {
        if (key !== dataInput && val !== "") inputs[key] = val;
      }
      const { run_id } = await api.create(wf.name, inputs, { project, session });
      if (dataInput && file) await api.uploadData(run_id, dataInput, file);
      await api.start(run_id);
      onLaunched(run_id);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  const missing =
    !wf ||
    (dataInput && !file) ||
    Object.entries(wf.inputs).some(
      ([k, s]) => s.required && k !== dataInput && !values[k],
    );

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
          <p className="muted" style={{ fontSize: 13, marginTop: -4 }}>
            {wf.description}
          </p>
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
                      <option key={o} value={o}>
                        {o}
                      </option>
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
                <span className="faint"> · data file</span>
              </label>
              <label
                className={`dropzone ${over ? "over" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setOver(true);
                }}
                onDragLeave={() => setOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setOver(false);
                  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
                }}
              >
                {file ? (
                  <span className="mono">{file.name}</span>
                ) : (
                  <>Drop a file here, or click to choose</>
                )}
                <input
                  type="file"
                  hidden
                  onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
                />
              </label>
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
