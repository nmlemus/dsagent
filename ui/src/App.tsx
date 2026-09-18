import { type CSSProperties, useCallback, useEffect, useMemo, useState } from "react";
import { ChatCenter } from "./components/ChatCenter";
import { LauncherModal } from "./components/LauncherModal";
import { RunDock } from "./components/RunDock";
import { Settings } from "./components/Settings";
import { Sidebar } from "./components/Sidebar";
import { Spine } from "./components/Spine";
import { api } from "./lib/api";
import type { Project } from "./lib/types";
import {
  type Conversation,
  deleteConversation,
  listConversations,
  newConversation,
  saveConversations,
  upsertConversation,
} from "./lib/conversations";

const SIDEBAR_KEY = "ds.sidebarCollapsed";
const PROJECT_KEY = "ds.activeProject";

// A conversation is "real" (listed in the sidebar) once it has a message, i.e. a
// title. A fresh chat is a draft: active in the center, but not listed — so the
// sidebar never fills with empty "New chat" rows.
const isReal = (c: Conversation) => !!c.title && c.title !== "New chat";

export function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string>(() => {
    try {
      return localStorage.getItem(PROJECT_KEY) || "default";
    } catch {
      return "default";
    }
  });
  // Only real (titled) conversations live here and in the sidebar.
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    listConversations().filter(isReal),
  );
  const [active, setActive] = useState<Conversation | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [launcherOpen, setLauncherOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [runsRefreshKey, setRunsRefreshKey] = useState(0);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_KEY) === "1";
    } catch {
      return false;
    }
  });

  // One-time cleanup of legacy empty "New chat" rows, then load projects.
  useEffect(() => {
    const all = listConversations();
    if (all.some((c) => !isReal(c))) saveConversations(all.filter(isReal));
    api
      .projects()
      .then((ps) => {
        setProjects(ps);
        if (!ps.some((p) => p.id === activeProjectId)) setActiveProjectId(ps[0]?.id ?? "default");
      })
      .catch(() =>
        setProjects([{ id: "default", name: "Default", goal: "", cartridge: "", created_at: 0, runs: 0 }]),
      );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const projectConvs = useMemo(
    () =>
      conversations
        .filter((c) => c.projectId === activeProjectId)
        .sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations, activeProjectId],
  );

  // Keep `active` pointed at a conversation in the current project (a real one if
  // there is any, else a fresh draft). Runs on mount and on project switch.
  useEffect(() => {
    setActive((cur) => {
      if (cur && cur.projectId === activeProjectId) return cur;
      return projectConvs[0] ?? newConversation(activeProjectId);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProjectId]);

  const persist = (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* ignore */
    }
  };

  const setProject = (id: string) => {
    setActiveProjectId(id);
    persist(PROJECT_KEY, id);
    setActiveRunId(null);
  };

  const setCollapse = (v: boolean) => {
    setCollapsed(v);
    persist(SIDEBAR_KEY, v ? "1" : "0");
  };

  const newChat = useCallback(() => setActive(newConversation(activeProjectId)), [activeProjectId]);

  const selectConv = useCallback(
    (id: string) => setActive((prev) => conversations.find((c) => c.id === id) ?? prev),
    [conversations],
  );

  const removeChat = useCallback(
    (id: string) => {
      deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      setActive((cur) => {
        if (cur?.id !== id) return cur;
        const rest = conversations.filter((c) => c.projectId === activeProjectId && c.id !== id);
        return rest[0] ?? newConversation(activeProjectId);
      });
    },
    [conversations, activeProjectId],
  );

  // ChatCenter reports the derived title; a real title promotes the draft into
  // the sidebar (and updates an existing one).
  const onTitle = useCallback(
    (id: string, title: string) => {
      if (!title || title === "New chat") return;
      setConversations((prev) => {
        const base = prev.find((c) => c.id === id) ?? (active?.id === id ? active : null);
        if (!base) return prev;
        const updated: Conversation = { ...base, title, updatedAt: Date.now() };
        upsertConversation(updated);
        return [updated, ...prev.filter((c) => c.id !== id)];
      });
      setActive((a) => (a && a.id === id ? { ...a, title } : a));
    },
    [active],
  );

  const shellStyle = {
    "--sidebar-w": collapsed ? "48px" : "268px",
    "--dock-w": activeRunId ? "clamp(360px, 34vw, 520px)" : "0px",
  } as CSSProperties;

  return (
    <div className="shell" style={shellStyle}>
      {collapsed ? (
        <Spine label="DSAgent" icon="≡" side="left" onExpand={() => setCollapse(false)} />
      ) : (
        <Sidebar
          projects={projects}
          activeProjectId={activeProjectId}
          onSelectProject={setProject}
          onCreatedProject={(p) => {
            setProjects((prev) => [...prev, p]);
            setProject(p.id);
          }}
          conversations={projectConvs}
          activeConvId={active?.id ?? null}
          onSelectConv={selectConv}
          onNewChat={newChat}
          onDeleteConv={removeChat}
          activeRunId={activeRunId}
          onSelectRun={setActiveRunId}
          onNewRun={() => setLauncherOpen(true)}
          runsRefreshKey={runsRefreshKey}
          onCollapse={() => setCollapse(true)}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      )}

      {active ? (
        <ChatCenter
          key={active.id}
          conversation={active}
          project={activeProjectId}
          onTitle={(t) => onTitle(active.id, t)}
          onOpenRun={setActiveRunId}
        />
      ) : (
        <div className="chat-center" />
      )}

      {activeRunId && <RunDock key={activeRunId} runId={activeRunId} onClose={() => setActiveRunId(null)} />}

      {launcherOpen && (
        <LauncherModal
          project={activeProjectId}
          session={active?.threadId ?? ""}
          onClose={() => setLauncherOpen(false)}
          onLaunched={(runId) => {
            setLauncherOpen(false);
            setRunsRefreshKey((k) => k + 1);
            setActiveRunId(runId);
          }}
        />
      )}

      {settingsOpen && <Settings onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
