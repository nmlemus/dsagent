import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Project } from "../lib/types";
import { useWorkspaceConfig } from "../lib/experience";
import { Icon } from "./Icon";
import { Dialog } from "./Dialog";
export function ProjectSwitcher({
  projects,
  activeId,
  onSelect,
  onCreated,
}: {
  projects: Project[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreated: (project: Project) => void;
}) {
  const [open, setOpen] = useState(false),
    [creating, setCreating] = useState(false);
  const [name, setName] = useState(""),
    [goal, setGoal] = useState(""),
    [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  // A project belongs to one cartridge — one team, one vocabulary. With a
  // single cartridge loaded there is nothing to choose and the field is hidden;
  // the binding is still recorded so the project keeps its domain later.
  const { cartridges } = useWorkspaceConfig();
  const [cartridge, setCartridge] = useState("");
  const chosen = cartridge || cartridges[0]?.name || "";
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  async function create(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const p = await api.createProject(name.trim(), goal.trim(), chosen);
      onCreated(p);
      setCreating(false);
      setName("");
      setGoal("");
    } catch {
      setError("No se pudo crear el proyecto. Inténtalo de nuevo.");
    } finally {
      setBusy(false);
    }
  }
  const active = projects.find((p) => p.id === activeId);
  return (
    <div
      className="proj-switcher"
      ref={ref}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          setOpen(false);
          ref.current?.querySelector("button")?.focus();
        }
      }}
    >
      <button
        className="proj-current"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-label="Cambiar proyecto"
      >
        <span className="proj-icon">
          <Icon name="folder" size={18} />
        </span>
        <span className="proj-col">
          <span className="proj-kicker">PROYECTO ACTIVO</span>
          <span className="proj-name">
            {active?.name === "Default"
              ? "Mi primer proyecto"
              : (active?.name ?? "Seleccionar proyecto")}
          </span>
        </span>
        <Icon name="down" size={15} />
      </button>
      {open && (
        <div className="proj-menu">
          <div className="proj-menu-head">Tus proyectos</div>
          {projects.map((p) => (
            <button
              key={p.id}
              className={`proj-item ${p.id === activeId ? "active" : ""}`}
              onClick={() => {
                onSelect(p.id);
                setOpen(false);
              }}
            >
              <Icon name="folder" size={16} />
              <span className="proj-item-name">
                {p.name === "Default" ? "Mi primer proyecto" : p.name}
              </span>
              {p.id === activeId && <Icon name="check" size={16} />}
            </button>
          ))}
          <button
            className="proj-new"
            onClick={() => {
              setCreating(true);
              setOpen(false);
            }}
          >
            <Icon name="plus" size={16} /> Crear proyecto
          </button>
        </div>
      )}
      {creating && (
        <Dialog
          title="Un nuevo espacio para tus ideas"
          onClose={() => setCreating(false)}
        >
          <form className="dialog-body stack" onSubmit={create}>
            <p className="muted">
              Reúne recursos, conversaciones y resultados alrededor de un
              objetivo.
            </p>
            {cartridges.length > 1 && (
              <div className="field">
                <label htmlFor="project-cartridge">Equipo</label>
                <select
                  id="project-cartridge"
                  value={chosen}
                  onChange={(e) => setCartridge(e.target.value)}
                >
                  {cartridges.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.experience.title} · {c.experience.tagline}
                    </option>
                  ))}
                </select>
                <small className="faint">
                  {cartridges.find((c) => c.name === chosen)?.description}
                </small>
              </div>
            )}
            <div className="field">
              <label htmlFor="project-name">Nombre del proyecto</label>
              <input
                id="project-name"
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Ej. Inteligencia comercial"
                required
                maxLength={120}
              />
            </div>
            <div className="field">
              <label htmlFor="project-goal">
                ¿Qué quieres descubrir? <span className="faint">Opcional</span>
              </label>
              <textarea
                id="project-goal"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="Entender qué impulsa nuestras ventas…"
                rows={3}
              />
            </div>
            {error && (
              <div className="err-banner" role="alert">
                {error}
              </div>
            )}
            <div className="dialog-actions">
              <button
                className="btn ghost"
                type="button"
                onClick={() => setCreating(false)}
              >
                Cancelar
              </button>
              <button className="btn primary" disabled={busy || !name.trim()}>
                {busy ? "Creando…" : "Crear proyecto"}
                <Icon name="arrow" size={17} />
              </button>
            </div>
          </form>
        </Dialog>
      )}
    </div>
  );
}
