import {
  Component,
  type ReactNode,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { LauncherModal } from "./components/LauncherModal";
import { Settings } from "./components/Settings";
import {
  Sidebar,
  NAV_IDS,
  navItems,
  type WorkspaceView,
} from "./components/Sidebar";
import { Dialog } from "./components/Dialog";
import { Icon } from "./components/Icon";
import {
  Analyses,
  DataLibrary,
  Deliverables,
  Overview,
} from "./components/WorkspaceViews";
import { api } from "./lib/api";
import { useWorkspace } from "./lib/useWorkspace";
import { useCartridges } from "./lib/useCartridges";
import {
  DEFAULT_EXPERIENCE,
  WorkspaceConfigContext,
  type WorkspaceConfig,
  cap,
  cartridgeFor,
  runTitle,
} from "./lib/experience";
import { documentTitle, experienceLabel } from "./lib/brand";
import type { Project } from "./lib/types";
import {
  type Conversation,
  fromHeader,
  newConversation,
} from "./lib/conversations";
const ChatCenter = lazy(() => import("./components/ChatCenter"));
const RunDock = lazy(() =>
  import("./components/RunDock").then((m) => ({ default: m.RunDock })),
);
const storage = {
  get(key: string, fallback: string) {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key: string, value: string) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* preference is optional */
    }
  },
};
function initialView(): WorkspaceView {
  const value = location.hash.slice(1);
  return NAV_IDS.some((v) => v.id === value)
    ? (value as WorkspaceView)
    : "overview";
}
class WorkspaceBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <div className="workspace-page">
        <div className="surface studio-empty">
          <Icon name="alert" size={30} />
          <h1>No se pudo mostrar esta vista.</h1>
          <p>Recarga la página para volver a intentarlo.</p>
          <button className="btn primary" onClick={() => location.reload()}>
            Recargar
          </button>
        </div>
      </div>
    ) : (
      this.props.children
    );
  }
}
export function App() {
  const [projects, setProjects] = useState<Project[]>([]),
    [projectError, setProjectError] = useState("");
  const [activeProjectId, setActiveProjectId] = useState(() =>
    storage.get("ds.activeProject", "default"),
  );
  const [view, setView] = useState<WorkspaceView>(initialView);
  const [conversations, setConversations] = useState<Conversation[]>([]),
    [active, setActive] = useState<Conversation | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null),
    [launcherOpen, setLauncherOpen] = useState(false),
    [settingsOpen, setSettingsOpen] = useState(false);
  const [deleteConversation, setDeleteConversation] =
    useState<Conversation | null>(null);
  const [deleteError, setDeleteError] = useState("");
  const [deletingConversation, setDeletingConversation] = useState(false);
  const [initialDataset, setInitialDataset] = useState<string | undefined>();
  const [initialWorkflow, setInitialWorkflow] = useState<string | undefined>();
  const [refreshKey, setRefreshKey] = useState(0),
    [collapsed, setCollapsed] = useState(
      () => storage.get("ds.sidebarCollapsed", "0") === "1",
    ),
    [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false),
    [search, setSearch] = useState("");
  const projectRef = useRef(activeProjectId);
  projectRef.current = activeProjectId;
  const data = useWorkspace(activeProjectId, refreshKey);
  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);
  // One way in to the launcher. A caller may preselect the resource to work on
  // or the workflow to run; anything it leaves out is cleared, so a previous
  // launch never leaks into the next one.
  const openLauncher = useCallback(
    (options: { dataset?: string; workflow?: string } = {}) => {
      setInitialDataset(options.dataset);
      setInitialWorkflow(options.workflow);
      setLauncherOpen(true);
    },
    [],
  );
  const server = useCartridges(refreshKey);
  const project = projects.find((p) => p.id === activeProjectId);
  // The words this workspace speaks: the active project's cartridge, else the
  // first one the server loaded, else the harness's generic defaults.
  const boundCartridge = project?.cartridge ?? "";
  const activeCartridge = cartridgeFor(server.cartridges, boundCartridge);
  const config = useMemo<WorkspaceConfig>(
    () => ({
      experience: activeCartridge?.experience ?? DEFAULT_EXPERIENCE,
      cartridges: server.cartridges,
      workflows: server.workflows,
      personas: activeCartridge?.personas ?? [],
      cartridge: boundCartridge,
      loading: server.loading,
    }),
    [activeCartridge, server, boundCartridge],
  );
  const { experience } = config;
  useEffect(() => {
    document.title = documentTitle(experience.title);
  }, [experience.title]);
  useEffect(() => {
    let live = true;
    api
      .projects()
      .then((ps) => {
        if (!live) return;
        setProjects(ps);
        setProjectError("");
        setActiveProjectId((current) =>
          ps.some((p) => p.id === current) ? current : (ps[0]?.id ?? "default"),
        );
      })
      .catch(() => {
        if (live)
          setProjectError(
            "No pudimos cargar tus proyectos. Comprueba que el servidor esté disponible.",
          );
      });
    return () => {
      live = false;
    };
  }, [refreshKey]);
  useEffect(() => {
    let live = true;
    setConversations([]);
    setActive(null);
    api
      .sessions(activeProjectId)
      .then((headers) => {
        if (!live) return;
        const convs = headers.map(fromHeader);
        setConversations(convs);
        setActive((current) =>
          current?.projectId === activeProjectId
            ? current
            : (convs[0] ?? newConversation(activeProjectId)),
        );
      })
      .catch(() => {
        if (live)
          setActive((current) =>
            current?.projectId === activeProjectId
              ? current
              : newConversation(activeProjectId),
          );
      });
    return () => {
      live = false;
    };
  }, [activeProjectId]);
  useEffect(() => {
    const change = () => setView(initialView());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen((v) => !v);
      }
      if (e.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  useEffect(() => {
    const breakpoint = matchMedia("(min-width: 761px)");
    const close = () => {
      if (breakpoint.matches) setMobileOpen(false);
    };
    breakpoint.addEventListener("change", close);
    return () => breakpoint.removeEventListener("change", close);
  }, []);
  const navigate = (next: WorkspaceView) => {
    setView(next);
    location.hash = next;
    setMobileOpen(false);
  };
  const setProject = (id: string) => {
    setActiveProjectId(id);
    storage.set("ds.activeProject", id);
    setActiveRunId(null);
    setMobileOpen(false);
  };
  const newChat = () => {
    setActive(newConversation(activeProjectId));
    navigate("chat");
  };
  const onTitle = useCallback((id: string, title: string) => {
    if (!title || title === "New chat") return;
    setActive((a) => (a?.id === id ? { ...a, title } : a));
    const project = projectRef.current;
    api
      .sessions(project)
      .then((headers) => {
        if (projectRef.current === project)
          setConversations(headers.map(fromHeader));
      })
      .catch(() => {
        /* next project load retries */
      });
  }, []);
  const activeId = active?.id;
  const updateTitle = useCallback(
    (title: string) => {
      if (activeId) onTitle(activeId, title);
    },
    [activeId, onTitle],
  );
  async function removeConversation() {
    if (!deleteConversation) return;
    setDeletingConversation(true);
    setDeleteError("");
    try {
      await api.deleteSession(deleteConversation.id);
      setConversations((current) =>
        current.filter((c) => c.id !== deleteConversation.id),
      );
      setActive((current) =>
        current?.id === deleteConversation.id
          ? newConversation(activeProjectId)
          : current,
      );
      setDeleteConversation(null);
    } catch {
      setDeleteError(
        "No se pudo eliminar la conversación. Vuelve a intentarlo.",
      );
    } finally {
      setDeletingConversation(false);
    }
  }
  const projectName =
    project?.name === "Default"
      ? "Mi primer proyecto"
      : (project?.name ?? "Tu espacio de trabajo");
  const activeExperience = experienceLabel(experience.title);
  const NAV_ITEMS = navItems(experience);
  const currentNav = NAV_ITEMS.find((v) => v.id === view)!;
  const query = search.trim().toLowerCase();
  const byName = new Map(server.workflows.map((w) => [w.name, w] as const));
  const nounRun = experience.nouns.run;
  return (
    <WorkspaceConfigContext.Provider value={config}>
      <div className={`studio-shell ${collapsed ? "sidebar-compact" : ""}`}>
        <a
          href="#main-content"
          className="skip-link"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("main-content")?.focus();
          }}
        >
          Saltar al contenido
        </a>
        <Sidebar
          projects={projects}
          activeProjectId={activeProjectId}
          onSelectProject={setProject}
          onCreatedProject={(p) => {
            setProjects((ps) => [...ps, p]);
            setProject(p.id);
          }}
          conversations={conversations}
          activeConvId={active?.id ?? null}
          onSelectConv={(id) => {
            setActive(conversations.find((c) => c.id === id) ?? null);
            navigate("chat");
          }}
          onDeleteConv={(conversation) => {
            setDeleteError("");
            setDeleteConversation(conversation);
          }}
          onNewChat={newChat}
          view={view}
          onNavigate={navigate}
          collapsed={collapsed}
          onCollapse={() => {
            setCollapsed(!collapsed);
            storage.set("ds.sidebarCollapsed", collapsed ? "0" : "1");
          }}
          onOpenSettings={() => setSettingsOpen(true)}
          onNewRun={() => openLauncher()}
          gateCount={
            data.runs.filter((r) => r.status === "awaiting_gate").length
          }
          mobileOpen={mobileOpen}
          onCloseMobile={() => setMobileOpen(false)}
        />
        <div className="studio-main">
          <header className="workspace-topbar">
            <div className="breadcrumb">
              <button
                className="icon-btn mobile-menu"
                aria-label="Abrir navegación"
                aria-expanded={mobileOpen}
                onClick={() => setMobileOpen(true)}
              >
                <Icon name="menu" />
              </button>
              <span className="breadcrumb-project">
                <Icon name="folder" size={15} />
                {projectName}
              </span>
              {/* The experience labels the project, never the product. */}
              {activeExperience && (
                <span className="breadcrumb-experience">
                  {activeExperience}
                </span>
              )}
              <span className="breadcrumb-divider">/</span>
              <strong>{currentNav.label}</strong>
            </div>
            <div className="topbar-actions">
              <button
                className="command-trigger"
                onClick={() => {
                  setSearch("");
                  setSearchOpen(true);
                }}
                aria-label="Buscar en el espacio de trabajo"
              >
                <Icon name="search" size={16} />
                <span>Buscar en tu espacio</span>
                <kbd>⌘ K</kbd>
              </button>
              <button
                className="topbar-avatar"
                onClick={() => setSettingsOpen(true)}
                aria-label="Abrir preferencias"
              >
                TU
              </button>
            </div>
          </header>
          <main
            id="main-content"
            className={`main-content ${view === "chat" ? "chat-page" : ""}`}
            tabIndex={-1}
          >
            {(projectError || data.error || server.error) && (
              <div className="connection-banner" role="alert">
                <Icon name="alert" size={18} />
                <span>{projectError || data.error || server.error}</span>
                <button className="text-button" onClick={refresh}>
                  Reintentar
                  <Icon name="refresh" size={15} />
                </button>
              </div>
            )}
            <WorkspaceBoundary key={`${activeProjectId}:${view}`}>
              <Suspense
                fallback={
                  <div className="view-loading" role="status">
                    <span className="loading-ring" />
                    Preparando tu espacio…
                  </div>
                }
              >
                {view === "overview" && (
                  <Overview
                    {...data}
                    project={project}
                    unavailable={!!data.error}
                    onNewRun={(workflow) => openLauncher({ workflow })}
                    onChat={() => navigate("chat")}
                    onRuns={() => navigate("runs")}
                    onData={() => navigate("data")}
                    onOpenRun={setActiveRunId}
                  />
                )}
                {view === "runs" && (
                  <Analyses
                    runs={data.runs}
                    loading={data.loading}
                    onOpen={setActiveRunId}
                    onNewRun={() => openLauncher()}
                  />
                )}
                {view === "data" && (
                  <DataLibrary
                    key={activeProjectId}
                    project={activeProjectId}
                    datasets={data.datasets}
                    loading={data.loading}
                    onRefresh={refresh}
                    onNewRun={(dataset) => openLauncher({ dataset })}
                  />
                )}
                {view === "deliverables" && (
                  <Deliverables
                    project={activeProjectId}
                    runs={data.runs}
                    artifacts={data.artifacts}
                    onOpen={setActiveRunId}
                    onNewRun={() => openLauncher()}
                  />
                )}
                {view === "chat" && (
                  <>
                    <div className="chat-page-heading">
                      <div>
                        <Icon name="spark" size={19} />
                        <strong>
                          {active?.title && active.title !== "New chat"
                            ? active.title
                            : experience.nav.chat}
                        </strong>
                        <span>{experience.nav.chat.toUpperCase()} · IA</span>
                      </div>
                      <button className="btn ghost sm" onClick={newChat}>
                        <Icon name="plus" size={16} />
                        Nueva conversación
                      </button>
                    </div>
                    {active && active.projectId === activeProjectId ? (
                      <ChatCenter
                        key={active.id}
                        conversation={active}
                        project={activeProjectId}
                        onTitle={updateTitle}
                        onOpenRun={setActiveRunId}
                      />
                    ) : (
                      <div className="view-loading" role="status">
                        Cargando conversaciones…
                      </div>
                    )}
                  </>
                )}
              </Suspense>
            </WorkspaceBoundary>
          </main>
        </div>
        {activeRunId && (
          <Dialog
            title={`Espacio de ${nounRun.singular}`}
            chrome={false}
            className="run-dialog"
            onClose={() => setActiveRunId(null)}
          >
            <WorkspaceBoundary>
              <Suspense
                fallback={
                  <div className="view-loading" role="status">
                    Cargando {nounRun.singular}…
                  </div>
                }
              >
                <RunDock
                  key={activeRunId}
                  runId={activeRunId}
                  onClose={() => setActiveRunId(null)}
                />
              </Suspense>
            </WorkspaceBoundary>
          </Dialog>
        )}
        {launcherOpen && (
          <LauncherModal
            initialDataset={initialDataset}
            initialWorkflow={initialWorkflow}
            project={activeProjectId}
            cartridge={boundCartridge}
            session={view === "chat" ? (active?.threadId ?? "") : ""}
            onClose={() => setLauncherOpen(false)}
            onLaunched={(id) => {
              setLauncherOpen(false);
              refresh();
              setActiveRunId(id);
            }}
          />
        )}
        {deleteConversation && (
          <Dialog
            title="Eliminar conversación"
            onClose={() => {
              if (!deletingConversation) setDeleteConversation(null);
            }}
          >
            <div className="dialog-body stack">
              <p>
                Se eliminará <strong>{deleteConversation.title}</strong> de este
                proyecto. Esta acción no se puede deshacer.
              </p>
              {deleteError && (
                <div className="err-banner" role="alert">
                  {deleteError}
                </div>
              )}
              <div className="dialog-actions">
                <button
                  className="btn"
                  disabled={deletingConversation}
                  onClick={() => setDeleteConversation(null)}
                >
                  Conservar conversación
                </button>
                <button
                  className="btn danger"
                  disabled={deletingConversation}
                  onClick={removeConversation}
                >
                  {deletingConversation
                    ? "Eliminando…"
                    : "Eliminar conversación"}
                </button>
              </div>
            </div>
          </Dialog>
        )}
        {settingsOpen && <Settings onClose={() => setSettingsOpen(false)} />}
        {searchOpen && (
          <Dialog
            title="Encuentra tu siguiente paso"
            className="command-dialog"
            onClose={() => setSearchOpen(false)}
          >
            <div className="dialog-body">
              <label className="command-input">
                <Icon name="search" />
                <input
                  autoFocus
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  aria-label={`Buscar vistas, proyectos y ${nounRun.plural}`}
                  placeholder={`Busca vistas, proyectos o ${nounRun.plural}…`}
                />
              </label>
              <div className="command-results">
                {NAV_ITEMS.filter((i) =>
                  i.label.toLowerCase().includes(query),
                ).map((i) => (
                  <button
                    key={i.id}
                    onClick={() => {
                      navigate(i.id);
                      setSearchOpen(false);
                    }}
                  >
                    <Icon name={i.icon} />
                    <span>{i.label}</span>
                    <small>Ir a vista</small>
                    <Icon name="arrow" size={15} />
                  </button>
                ))}
                {projects
                  .filter((p) => p.name.toLowerCase().includes(query))
                  .slice(0, 5)
                  .map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setProject(p.id);
                        setSearchOpen(false);
                      }}
                    >
                      <Icon name="folder" />
                      <span>{p.name}</span>
                      <small>Proyecto</small>
                    </button>
                  ))}
                {data.runs
                  .filter((r) =>
                    `${runTitle(r, byName.get(r.workflow))} ${r.run_id}`
                      .toLowerCase()
                      .includes(query),
                  )
                  .slice(0, 5)
                  .map((r) => (
                    <button
                      key={r.run_id}
                      onClick={() => {
                        setActiveRunId(r.run_id);
                        setSearchOpen(false);
                      }}
                    >
                      <Icon name="chart" />
                      <span>{runTitle(r, byName.get(r.workflow))}</span>
                      <small>{cap(nounRun.singular)}</small>
                    </button>
                  ))}
                {!NAV_ITEMS.some((i) =>
                  i.label.toLowerCase().includes(query),
                ) &&
                  !projects.some((p) => p.name.toLowerCase().includes(query)) &&
                  !data.runs.some((r) =>
                    `${runTitle(r, byName.get(r.workflow))} ${r.run_id}`
                      .toLowerCase()
                      .includes(query),
                  ) && (
                    <p className="empty">
                      No hay resultados. Prueba con otro término.
                    </p>
                  )}
              </div>
              <div className="command-hint">
                <kbd>Tab</kbd> para navegar <span />
                <kbd>Esc</kbd> para cerrar
              </div>
            </div>
          </Dialog>
        )}
      </div>
    </WorkspaceConfigContext.Provider>
  );
}
