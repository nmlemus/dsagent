import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { ago, cost } from "../lib/format";
import type { Conversation } from "../lib/conversations";
import type { Project, RunSummary } from "../lib/types";
import { DatasetsSection } from "./DatasetsSection";
import { ProjectSwitcher } from "./ProjectSwitcher";
import { StatusPill } from "./StatusPill";

interface Props {
  projects: Project[];
  activeProjectId: string;
  onSelectProject: (id: string) => void;
  onCreatedProject: (p: Project) => void;

  conversations: Conversation[];
  activeConvId: string | null;
  onSelectConv: (id: string) => void;
  onNewChat: () => void;
  onDeleteConv: (id: string) => void;

  activeRunId: string | null;
  onSelectRun: (id: string) => void;
  onNewRun: () => void;
  runsRefreshKey: number;

  onCollapse: () => void;
  onOpenSettings: () => void;
}

export function Sidebar({
  projects,
  activeProjectId,
  onSelectProject,
  onCreatedProject,
  conversations,
  activeConvId,
  onSelectConv,
  onNewChat,
  onDeleteConv,
  activeRunId,
  onSelectRun,
  onNewRun,
  runsRefreshKey,
  onCollapse,
  onOpenSettings,
}: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    let live = true;
    const load = () =>
      api.projectRuns(activeProjectId, { limit: 30 }).then((r) => live && setRuns(r)).catch(() => {});
    load();
    const t = setInterval(load, 4000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [runsRefreshKey, activeProjectId]);

  return (
    <div className="sidebar">
      <div className="sidebar-head">
        <ProjectSwitcher
          projects={projects}
          activeId={activeProjectId}
          onSelect={onSelectProject}
          onCreated={onCreatedProject}
        />
        <button className="icon-btn" onClick={onCollapse} title="Collapse" aria-label="Collapse sidebar">
          «
        </button>
      </div>

      <div className="sidebar-scroll">
        <button className="side-action" onClick={onNewChat}>
          <span>＋</span> New chat
        </button>

        <div className="side-label">Chats</div>
        {conversations.length === 0 && <div className="side-empty">No chats yet.</div>}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`side-item ${activeConvId === c.id ? "active" : ""}`}
            onClick={() => onSelectConv(c.id)}
          >
            <span className="side-item-icon">💬</span>
            <span className="side-item-title">{c.title || "New chat"}</span>
            <button
              className="side-item-x"
              title="Delete chat"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteConv(c.id);
              }}
            >
              ✕
            </button>
          </div>
        ))}

        <button className="side-action" style={{ marginTop: 14 }} onClick={onNewRun}>
          <span>＋</span> New run
        </button>

        <div className="side-label">Runs</div>
        {runs.length === 0 && <div className="side-empty">No runs yet.</div>}
        {runs.map((r) => (
          <div
            key={r.run_id}
            className={`side-item run ${activeRunId === r.run_id ? "active" : ""}`}
            onClick={() => onSelectRun(r.run_id)}
            title={r.run_id}
          >
            <span className="side-item-title mono" style={{ fontSize: 12 }}>
              {r.run_id.replace(/^eda-to-report-/, "eda·").replace(/^eda-charts-?/, "charts·")}
            </span>
            <div className="side-run-meta">
              <StatusPill status={r.status} />
              <span className="faint mono" style={{ fontSize: 10 }}>
                {r.steps_done}/{r.steps_total}
              </span>
              {r.cost_usd != null && (
                <span className="faint mono" style={{ fontSize: 10 }}>
                  {cost(r.cost_usd)}
                </span>
              )}
              <span className="faint" style={{ marginLeft: "auto", fontSize: 10 }}>
                {ago(r.finished_at ?? r.started_at)}
              </span>
            </div>
          </div>
        ))}

        <DatasetsSection project={activeProjectId} />
      </div>

      <button className="sidebar-user" onClick={onOpenSettings} title="Settings">
        <span className="user-avatar">N</span>
        <span className="user-name">You</span>
        <span className="user-gear">⚙</span>
      </button>
    </div>
  );
}
