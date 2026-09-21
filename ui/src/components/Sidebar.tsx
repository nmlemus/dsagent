import { useEffect, useRef } from "react";
import type { Conversation } from "../lib/conversations";
import type { Experience, Project } from "../lib/types";
import { useExperience } from "../lib/experience";
import { PRODUCT } from "../lib/brand";
import { Icon, BrandMark, CompanySignature, type IconName } from "./Icon";
import { ProjectSwitcher } from "./ProjectSwitcher";
export type WorkspaceView =
  "overview" | "chat" | "runs" | "data" | "deliverables";
// The five sections are the harness's structure; their names are the
// cartridge's (`experience.nav`).
export const NAV_IDS: {
  id: WorkspaceView;
  key: keyof Experience["nav"];
  icon: IconName;
}[] = [
  { id: "overview", key: "overview", icon: "grid" },
  { id: "chat", key: "chat", icon: "spark" },
  { id: "runs", key: "runs", icon: "chart" },
  { id: "data", key: "resources", icon: "data" },
  { id: "deliverables", key: "deliverables", icon: "file" },
];
export const navItems = (experience: Experience) =>
  NAV_IDS.map((n) => ({ ...n, label: experience.nav[n.key] }));
interface Props {
  projects: Project[];
  activeProjectId: string;
  onSelectProject: (id: string) => void;
  onCreatedProject: (p: Project) => void;
  conversations: Conversation[];
  activeConvId: string | null;
  onSelectConv: (id: string) => void;
  onDeleteConv: (conversation: Conversation) => void;
  onNewChat: () => void;
  view: WorkspaceView;
  onNavigate: (view: WorkspaceView) => void;
  collapsed: boolean;
  onCollapse: () => void;
  onOpenSettings: () => void;
  onNewRun: () => void;
  gateCount: number;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}
export function Sidebar(p: Props) {
  const experience = useExperience();
  const aside = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!p.mobileOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const main = document.querySelector<HTMLElement>(".studio-main");
    if (main) main.inert = true;
    aside.current?.querySelector<HTMLButtonElement>(".mobile-close")?.focus();
    return () => {
      if (main) main.inert = false;
      previous?.focus();
    };
  }, [p.mobileOpen]);
  return (
    <>
      {p.mobileOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Cerrar navegación"
          onClick={p.onCloseMobile}
        />
      )}
      <aside
        ref={aside}
        role={p.mobileOpen ? "dialog" : undefined}
        aria-modal={p.mobileOpen || undefined}
        onKeyDown={(event) => {
          if (!p.mobileOpen || event.key !== "Tab") return;
          const controls = Array.from(
            aside.current?.querySelectorAll<HTMLElement>(
              "button:not(:disabled), input, select, textarea, a[href]",
            ) ?? [],
          ).filter((el) => el.getClientRects().length > 0);
          const first = controls[0],
            last = controls[controls.length - 1];
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last?.focus();
          }
          if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first?.focus();
          }
        }}
        className={`studio-sidebar ${p.collapsed ? "is-collapsed" : ""} ${p.mobileOpen ? "mobile-open" : ""}`}
        aria-label="Navegación principal"
      >
        <div className="sidebar-brand">
          <BrandMark />
          {/* The product, not the cartridge: switching experience must not
              rename the application the user opened. */}
          <span className="brand-word">{PRODUCT}</span>
          <button
            className="icon-btn collapse-btn"
            onClick={p.onCollapse}
            aria-label={
              p.collapsed ? "Expandir navegación" : "Contraer navegación"
            }
          >
            <Icon name="panel" size={17} />
          </button>
          <button
            className="icon-btn mobile-close"
            onClick={p.onCloseMobile}
            aria-label="Cerrar navegación"
          >
            <Icon name="close" />
          </button>
        </div>
        <div className="sidebar-project">
          <ProjectSwitcher
            projects={p.projects}
            activeId={p.activeProjectId}
            onSelect={p.onSelectProject}
            onCreated={p.onCreatedProject}
          />
        </div>
        <div className="sidebar-section-label">ESPACIO DE TRABAJO</div>
        <nav className="workspace-nav">
          {navItems(experience).map((item) => (
            <button
              key={item.id}
              title={item.label}
              className={`nav-item ${p.view === item.id ? "selected" : ""}`}
              aria-current={p.view === item.id ? "page" : undefined}
              onClick={() => p.onNavigate(item.id)}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
              {item.id === "runs" && p.gateCount > 0 && (
                <b className="nav-count">{p.gateCount}</b>
              )}
              {item.id === "chat" && <span className="nav-ai">IA</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-conversations">
          <div className="sidebar-section-label">
            CONVERSACIONES
            <button
              className="icon-btn"
              onClick={p.onNewChat}
              aria-label="Nueva conversación"
            >
              <Icon name="plus" size={16} />
            </button>
          </div>
          {p.conversations.map((c) => (
            <div className="conversation-row" key={c.id}>
              <button
                className={`conversation-link ${p.view === "chat" && p.activeConvId === c.id ? "selected" : ""}`}
                onClick={() => p.onSelectConv(c.id)}
              >
                <Icon name="chat" size={15} />
                <span>{c.title || "Nueva conversación"}</span>
              </button>
              <button
                className="icon-btn conversation-delete"
                aria-label={`Eliminar conversación: ${c.title}`}
                onClick={() => p.onDeleteConv(c)}
              >
                <Icon name="trash" size={14} />
              </button>
            </div>
          ))}
          {p.conversations.length === 0 && (
            <p className="sidebar-empty">
              Tus ideas empiezan con una buena pregunta.
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <button className="btn primary sidebar-action" onClick={p.onNewRun}>
            <Icon name="plus" size={16} />
            Crear {experience.nouns.run.singular}
          </button>
          <button
            className="sidebar-user"
            onClick={p.onOpenSettings}
            title="Preferencias"
          >
            <span className="user-avatar">TU</span>
            <span className="user-name">
              Tu espacio<span>Preferencias</span>
            </span>
            <Icon name="settings" size={19} />
          </button>
          <p className="sidebar-signature">
            <CompanySignature />
          </p>
        </div>
      </aside>
    </>
  );
}
