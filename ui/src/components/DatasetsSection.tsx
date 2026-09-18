import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { bytes } from "../lib/format";

// Project datasets in the sidebar: upload once, reused by every run of the
// project (the launcher and the chat both seed them into new runs).
export function DatasetsSection({ project }: { project: string }) {
  const [datasets, setDatasets] = useState<{ name: string; bytes: number }[]>([]);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const load = () => api.datasets(project).then(setDatasets).catch(() => {});
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project]);

  async function upload(file: File) {
    setBusy(true);
    try {
      await api.uploadDataset(project, file);
      await load();
    } catch {
      /* ignore */
    } finally {
      setBusy(false);
    }
  }

  async function remove(name: string) {
    try {
      await api.deleteDataset(project, name);
      await load();
    } catch {
      /* ignore */
    }
  }

  return (
    <>
      <div className="row spread" style={{ margin: "14px 8px 6px" }}>
        <span className="side-label" style={{ margin: 0 }}>Datasets</span>
        <button className="icon-btn" onClick={() => fileRef.current?.click()} title="Subir dataset">
          {busy ? "…" : "↥"}
        </button>
        <input
          ref={fileRef}
          type="file"
          hidden
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
      </div>
      {datasets.length === 0 ? (
        <div className="side-empty">Sube un CSV para correr análisis.</div>
      ) : (
        datasets.map((d) => (
          <div key={d.name} className="side-item" title={d.name}>
            <span className="side-item-icon">▤</span>
            <span className="side-item-title mono" style={{ fontSize: 12 }}>{d.name}</span>
            <span className="faint mono" style={{ fontSize: 10 }}>{bytes(d.bytes)}</span>
            <button className="side-item-x" title="Borrar" onClick={() => remove(d.name)}>✕</button>
          </div>
        ))
      )}
    </>
  );
}
