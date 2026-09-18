import { type CSSProperties, useCallback, useEffect, useState } from "react";
import { ChatCenter } from "./components/ChatCenter";
import { LauncherModal } from "./components/LauncherModal";
import { RunDock } from "./components/RunDock";
import { Settings } from "./components/Settings";
import { Sidebar } from "./components/Sidebar";
import { Spine } from "./components/Spine";
import { api } from "./lib/api";
import type { Project } from "./lib/types";
import { type Conversation, fromHeader, newConversation } from "./lib/conversations";

const SIDEBAR_KEY = "ds.sidebarCollapsed";
const PROJECT_KEY = "ds.activeProject";

export function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string>(() => {
    try {
      return localStorage.getItem(PROJECT_KEY) || "default";
    } catch {
      return "default";
    }
  });
  const [conversations, setConversations] = useState<Conversation[]>([]);
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

  useEffect(() => {
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

  const refreshSessions = useCallback((projectId: string) => {
    api
      .sessions(projectId)
      .then((headers) => setConversations(headers.map(fromHeader)))
      .catch(() => setConversations([]));
  }, []);

  // Load the project's chats from the server; pick or draft an active one.
  useEffect(() => {
    api
      .sessions(activeProjectId)
      .then((headers) => {
        const convs = headers.map(fromHeader);
        setConversations(convs);
        setActive((cur) =>
          cur && cur.projectId === activeProjectId ? cur : convs[0] ?? newConversation(activeProjectId),
        );
      })
      .catch(() => {
        setConversations([]);
        setActive((cur) =>
          cur && cur.projectId === activeProjectId ? cur : newConversation(activeProjectId),
        );
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
      api.deleteSession(id).catch(() => {});
      setConversations((prev) => prev.filter((c) => c.id !== id));
      setActive((cur) => {
        if (cur?.id !== id) return cur;
        const rest = conversations.filter((c) => c.projectId === activeProjectId && c.id !== id);
        return rest[0] ?? newConversation(activeProjectId);
      });
    },
    [conversations, activeProjectId],
  );

  // ChatCenter reports the derived title as the chat grows (and saves it to the
  // server). Reflect it, and pull a freshly-created chat into the list.
  const onTitle = useCallback(
    (id: string, title: string) => {
      if (!title || title === "New chat") return;
      setActive((a) => (a && a.id === id ? { ...a, title } : a));
      setConversations((prev) => {
        if (prev.some((c) => c.id === id)) return prev.map((c) => (c.id === id ? { ...c, title } : c));
        setTimeout(() => refreshSessions(activeProjectId), 900);
        return prev;
      });
    },
    [activeProjectId, refreshSessions],
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
          conversations={conversations}
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
