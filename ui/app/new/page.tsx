"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import {
  type WorkflowInput,
  type WorkflowShape,
  createRun,
  getCatalogue,
  startRun,
  uploadInput,
} from "../lib/api";
import { fileSize, initial } from "../lib/format";

/**
 * New run — pick a workflow, fill in what it asks for, start it.
 *
 * The form is generated from the workflow's declared `inputs`, so a cartridge
 * with different inputs renders without a line changing here. The operator never
 * types `data_path=`: a file input is a drop zone, and the value the steps read
 * is whatever the server says the upload landed as.
 */
export default function NewRun() {
  const [workflows, setWorkflows] = useState<WorkflowShape[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chosen, setChosen] = useState<WorkflowShape | null>(null);

  useEffect(() => {
    let live = true;
    getCatalogue()
      .then((c) => {
        if (!live) return;
        setWorkflows(c.workflows);
        if (c.workflows.length === 1) setChosen(c.workflows[0]);
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);

  return (
    <main className="page page-narrow">
      <div className="page-head">
        <div>
          <h1 className="page-title">New run</h1>
          <p className="page-sub">
            Pick what the team should do and what it should work on. You can watch it from
            the moment it starts.
          </p>
        </div>
        <Link href="/" className="btn">
          Cancel
        </Link>
      </div>

      {error && <div className="notice">Can’t reach the agent server: {error}</div>}

      {workflows && workflows.length > 1 && (
        <section className="choices">
          {workflows.map((workflow) => (
            <WorkflowCard
              key={workflow.name}
              workflow={workflow}
              chosen={chosen?.name === workflow.name}
              onChoose={() => setChosen(workflow)}
            />
          ))}
        </section>
      )}

      {/* Keyed by workflow: switching one resets the form by remounting it,
          rather than by an effect that reaches in and clears the fields. */}
      {chosen && <RunForm key={chosen.name} workflow={chosen} />}
    </main>
  );
}

function WorkflowCard({
  workflow,
  chosen,
  onChoose,
}: {
  workflow: WorkflowShape;
  chosen: boolean;
  onChoose: () => void;
}) {
  const gates = workflow.steps.filter((s) => s.gate).length;
  return (
    <button className={chosen ? "choice is-chosen" : "choice"} onClick={onChoose}>
      <h2 className="choice-name">{workflow.name}</h2>
      <p className="choice-what">{workflow.description}</p>
      <div className="choice-team">
        {workflow.personas.map((persona) => (
          <span key={persona} className="avatar" title={persona}>
            {initial(persona)}
          </span>
        ))}
        <span className="dim">
          {workflow.steps.length} steps
          {gates > 0 && `, ${gates} stop${gates > 1 ? "s" : ""} for you`}
        </span>
      </div>
    </button>
  );
}

function RunForm({ workflow }: { workflow: WorkflowShape }) {
  const router = useRouter();
  const [values, setValues] = useState<Record<string, string>>(() => defaults(workflow));
  const [files, setFiles] = useState<Record<string, File>>({});
  const [starting, setStarting] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const missing = Object.entries(workflow.inputs).filter(
    ([name, spec]) => spec.required && !files[name] && !values[name]?.trim(),
  );

  async function start() {
    setStarting(true);
    setFailure(null);
    try {
      const { run_id } = await createRun(workflow.name, values);
      for (const [name, file] of Object.entries(files)) {
        await uploadInput(run_id, name, file);
      }
      await startRun(run_id);
      router.push(`/runs/${encodeURIComponent(run_id)}`);
    } catch (e) {
      setFailure(e instanceof Error ? e.message : String(e));
      setStarting(false);
    }
  }

  return (
    <section className="form">
      <div className="form-fields">
        {Object.entries(workflow.inputs).map(([name, spec]) => (
          <Field
            key={name}
            name={name}
            spec={spec}
            value={values[name] ?? ""}
            file={files[name] ?? null}
            onValue={(v) => setValues((current) => ({ ...current, [name]: v }))}
            onFile={(file) =>
              setFiles((current) => {
                const next = { ...current };
                if (file) next[name] = file;
                else delete next[name];
                return next;
              })
            }
          />
        ))}
      </div>

      <aside className="form-plan">
        <h2 className="plan-title">What will happen</h2>
        <ol className="plan-steps">
          {workflow.steps.map((step) => (
            <li key={step.id}>
              <span className="avatar">{initial(step.persona)}</span>
              <div>
                <b>{step.id}</b> <span className="dim">{step.persona}</span>
                {step.gate && <div className="plan-gate">stops here for your decision</div>}
              </div>
            </li>
          ))}
        </ol>
      </aside>

      {failure && <div className="notice">Could not start the run: {failure}</div>}

      <div className="form-actions">
        <button
          className="btn btn-primary"
          onClick={() => void start()}
          disabled={starting || missing.length > 0}
        >
          {starting ? "Starting…" : "Start run"}
        </button>
        {missing.length > 0 && (
          <span className="dim">
            Still needed: {missing.map(([name]) => label(name)).join(", ")}
          </span>
        )}
      </div>
    </section>
  );
}

function Field({
  name,
  spec,
  value,
  file,
  onValue,
  onFile,
}: {
  name: string;
  spec: WorkflowInput;
  value: string;
  file: File | null;
  onValue: (value: string) => void;
  onFile: (file: File | null) => void;
}) {
  const optional = !spec.required;
  return (
    <label className="field">
      <span className="field-label">
        {label(name)}
        {optional && <span className="dim"> · optional</span>}
      </span>
      {spec.type === "path" ? (
        <FileField file={file} onFile={onFile} />
      ) : spec.options ? (
        <select className="control" value={value} onChange={(e) => onValue(e.target.value)}>
          {spec.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : spec.type === "daterange" ? (
        <DateRange value={value} onValue={onValue} />
      ) : (
        <textarea
          className="control"
          rows={value.length > 90 ? 3 : 2}
          value={value}
          onChange={(e) => onValue(e.target.value)}
        />
      )}
      <span className="field-name mono dim">{name}</span>
    </label>
  );
}

/**
 * A drop zone that is also a file picker.
 *
 * The file goes up as the body of a `PUT`, and the input it fills is in the URL,
 * so the operator's side of this is one gesture and nothing about paths.
 */
function FileField({ file, onFile }: { file: File | null; onFile: (file: File | null) => void }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  return (
    <div
      className={over ? "drop is-over" : "drop"}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const dropped = e.dataTransfer.files?.[0];
        if (dropped) onFile(dropped);
      }}
    >
      <input
        ref={input}
        type="file"
        hidden
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {file ? (
        <div className="drop-file">
          <span className="mono">{file.name}</span>
          <span className="dim">{fileSize(file.size)}</span>
          <button className="btn btn-small" onClick={() => onFile(null)} type="button">
            Replace
          </button>
        </div>
      ) : (
        <button className="drop-invite" type="button" onClick={() => input.current?.click()}>
          Drop a file here, or choose one
        </button>
      )}
    </div>
  );
}

/** Two dates, kept as the `start..end` string the workflow declared. */
function DateRange({ value, onValue }: { value: string; onValue: (value: string) => void }) {
  const [from, to] = value.split("..");
  return (
    <div className="daterange">
      <input
        className="control"
        type="date"
        value={from ?? ""}
        onChange={(e) => onValue(`${e.target.value}..${to ?? ""}`)}
      />
      <span className="dim">to</span>
      <input
        className="control"
        type="date"
        value={to ?? ""}
        onChange={(e) => onValue(`${from ?? ""}..${e.target.value}`)}
      />
    </div>
  );
}

const defaults = (workflow: WorkflowShape): Record<string, string> =>
  Object.fromEntries(
    Object.entries(workflow.inputs).map(([name, spec]) => [name, spec.default ?? ""]),
  );

/** `data_path` is what the cartridge calls it; "Data path" is what a person reads. */
const label = (name: string): string =>
  name.replace(/[_-]+/g, " ").replace(/^./, (c) => c.toUpperCase());
