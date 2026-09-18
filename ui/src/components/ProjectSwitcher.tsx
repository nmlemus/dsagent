import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Project } from "../lib/types";

interface Props {
  projects: Project[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreated: (project: Project) => void;
}

export function ProjectSwitcher({ projects, activeId, onSelect, onCreated }: Props) {
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const active = projects.find((p) => p.id === activeId);

  async function create() {
    const n = name.trim();
    if (!n) return;
    try {
      const proj = await api.createProject(n);
      setName("");
      setCreating(false);
      setOpen(false);
      onCreated(proj);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="proj-switcher" ref={ref}>
      <button className="proj-current" onClick={() => setOpen((o) => !o)} title="Cambiar o crear proyecto">
        <span className="proj-icon">▣</span>
        <span className="proj-col">
          <span className="proj-kicker">Proyecto ▾</span>
          <span className="proj-name">{active?.name ?? "Default"}</span>
        </span>
      </button>

      {open && (
        <div className="proj-menu">
          <div className="proj-menu-head">Cambiar proyecto</div>
          {projects.map((p) => (
            <button
              key={p.id}
              className={`proj-item ${p.id === activeId ? "active" : ""}`}
              onClick={() => {
                onSelect(p.id);
                setOpen(false);
              }}
            >
              <span className="proj-item-icon">{p.id === activeId ? "●" : "○"}</span>
              <span className="proj-item-name">{p.name}</span>
              <span className="faint mono" style={{ fontSize: 11 }}>{p.runs} runs</span>
            </button>
          ))}
          <div className="proj-sep" />
          {creating ? (
            <div className="proj-create">
              <input
                autoFocus
                value={name}
                placeholder="Nombre del proyecto…"
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && create()}
              />
              <button className="btn primary sm" onClick={create}>Crear</button>
            </div>
          ) : (
            <button className="proj-new" onClick={() => setCreating(true)}>
              ＋ Nuevo proyecto
            </button>
          )}
        </div>
      )}
    </div>
  );
}
