import {
  type ReactNode,
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { HttpAgent } from "@ag-ui/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  ExportedMessageRepository,
  MessagePrimitive,
  type TextMessagePartProps,
  type ThreadHistoryAdapter,
  type ToolCallMessagePartProps,
  ThreadPrimitive,
  useAssistantInstructions,
} from "@assistant-ui/react";
import {
  fromAgUiMessages,
  toAgUiMessages,
  useAgUiRuntime,
} from "@assistant-ui/react-ag-ui";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { api, ApiError, projectSource } from "../lib/api";
import { FocusOverlay, type FocusTarget } from "./FocusOverlay";
import { Icon } from "./Icon";
import { cost } from "../lib/format";
import { isVegaLite } from "../lib/vega-detect";
import type { RunSummary } from "../lib/types";
import { type Conversation, titleFrom } from "../lib/conversations";
import { useExperience } from "../lib/experience";
import { StatusPill } from "./StatusPill";

const ChartView = lazy(() => import("./ChartView"));

function mdHtml(src: string): string {
  return DOMPurify.sanitize(marked.parse(src, { async: false }) as string);
}

// A fenced ```vega-lite block becomes a live chart; markdown tables render as
// tables; the rest is normal markdown. This is what lets the chat show charts
// inline with no backend change — the model emits the spec with data inline.
const CHART_FENCE = /```(?:vega-?lite|vega|chart)\s*\r?\n([\s\S]*?)```/gi;

function MarkdownText({ text }: TextMessagePartProps) {
  const src = text || "";
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  CHART_FENCE.lastIndex = 0;
  while ((m = CHART_FENCE.exec(src)) !== null) {
    const before = src.slice(last, m.index);
    if (before.trim())
      out.push(
        <div
          key={key++}
          className="md"
          dangerouslySetInnerHTML={{ __html: mdHtml(before) }}
        />,
      );
    let spec: unknown = null;
    try {
      spec = JSON.parse(m[1]);
    } catch {
      /* not valid JSON */
    }
    if (spec && isVegaLite(spec)) {
      out.push(
        <Suspense
          key={key++}
          fallback={
            <div className="viewer">
              <div className="empty">Preparando gráfico…</div>
            </div>
          }
        >
          <ChartView spec={spec as Record<string, unknown>} />
        </Suspense>,
      );
    } else {
      out.push(<pre key={key++}>{m[1]}</pre>);
    }
    last = m.index + m[0].length;
  }
  const rest = src.slice(last);
  if (rest.trim() || out.length === 0)
    out.push(
      <div
        key={key++}
        className="md"
        dangerouslySetInnerHTML={{ __html: mdHtml(rest) }}
      />,
    );
  return <>{out}</>;
}

function ToolFallback({ toolName, status }: ToolCallMessagePartProps) {
  const { nouns } = useExperience();
  const running = status?.type === "running";
  // The harness's own tools, in the harness's words; what a run is called is
  // the cartridge's.
  const label: Record<string, string> = {
    task: "consultando a un especialista",
    run_workflow: `iniciando ${nouns.run.singular}`,
    run_python: "ejecutando código",
    run_skill_script: "aplicando una herramienta",
    show_chart: "preparando un gráfico",
    show_table: "organizando los resultados",
  };
  return (
    <div className={`tool-call ${running ? "running" : "done"}`}>
      <span className="tool-dot" />
      <span className="mono">{toolName}</span>
      <span className="muted">
        — {label[toolName] ?? (running ? "working" : "done")}
      </span>
      {running && (
        <span className="dots">
          <i />
          <i />
          <i />
        </span>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <div className="typing">
      <i />
      <i />
      <i />
    </div>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="cmsg user">
      <div className="cbubble">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="cmsg assistant">
      <div className="cavatar" aria-hidden>
        ◆
      </div>
      <div className="cbody">
        <MessagePrimitive.Parts
          components={{ Text: MarkdownText, tools: { Fallback: ToolFallback } }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

// The cartridge's own instructions for the chat model — how results should be
// presented in its domain (`experience.assistant.instructions`). Mounted only
// when the cartridge has something to say.
function CartridgeInstructions({ text }: { text: string }) {
  useAssistantInstructions(text);
  return null;
}

// The runs this chat launched — the link made visible. Polls the project's runs
// filtered by this session, so a run the orchestrator started shows up here as a
// card, and clicking it opens the run dock.
function SessionRuns({
  project,
  session,
  onOpen,
}: {
  project: string;
  session: string;
  onOpen: (runId: string) => void;
}) {
  const { nouns } = useExperience();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  useEffect(() => {
    let live = true;
    const load = () =>
      api
        .projectRuns(project, { session })
        .then((r) => live && setRuns(r))
        .catch(() => {});
    load();
    const t = setInterval(load, 4000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [project, session]);

  if (runs.length === 0) return null;
  return (
    <div className="session-runs">
      <span className="faint" style={{ fontSize: 11 }}>
        {nouns.run.plural.charAt(0).toUpperCase() + nouns.run.plural.slice(1)}{" "}
        de esta conversación:
      </span>
      {runs.map((r) => (
        <button
          key={r.run_id}
          className="run-chip"
          onClick={() => onOpen(r.run_id)}
          title={r.run_id}
        >
          <StatusPill status={r.status} />
          <span className="mono" style={{ fontSize: 11 }}>
            {r.workflow}
          </span>
          {r.cost_usd != null && (
            <span className="faint mono" style={{ fontSize: 10 }}>
              {cost(r.cost_usd)}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

// Files the chat wrote into the project workspace — a persona's report, its
// charts, a table. They live per-project (the chat workspace is the project's),
// so this is "what this project has produced", polled so a fresh report shows
// up. Clicking one opens it in the app's own viewer, the same one a run's
// deliverables use: markdown renders, a CSV draws as a table, a Vega spec draws
// as a chart. Only a full interactive HTML document still opens in its own tab,
// and the viewer says so with a button.
function ProjectArtifacts({ project }: { project: string }) {
  const [files, setFiles] = useState<
    { path: string; bytes: number; mtime: number }[]
  >([]);
  const [open, setOpen] = useState<FocusTarget | null>(null);
  useEffect(() => {
    let live = true;
    const load = () =>
      api
        .projectArtifacts(project)
        .then((f) => live && setFiles(f))
        .catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [project]);

  if (files.length === 0) return null;
  return (
    <div className="session-runs">
      <span className="faint" style={{ fontSize: 11 }}>
        Archivos del proyecto:
      </span>
      {files.slice(0, 12).map((f) => (
        <button
          key={f.path}
          className="run-chip"
          onClick={() => setOpen({ kind: "file", path: f.path })}
          title={f.path}
        >
          <span aria-hidden>▤</span>
          <span className="mono" style={{ fontSize: 11 }}>
            {f.path.split("/").pop()}
          </span>
        </button>
      ))}
      {open && (
        <FocusOverlay
          source={projectSource(project)}
          target={open}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}

interface Props {
  conversation: Conversation;
  project: string;
  onTitle: (title: string) => void;
  onOpenRun: (runId: string) => void;
}

// Fetch the chat's messages first, then mount the thread with them as the
// agent's initial messages — assistant-ui's runtime reads messages at agent
// creation, so loading past history has to happen before the agent exists.
export function ChatCenter(props: Props) {
  const [initial, setInitial] = useState<unknown[] | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let live = true;
    setInitial(null);
    setError("");
    async function load() {
      try {
        try {
          const rec = await api.getSession(props.conversation.id);
          if (!Array.isArray(rec.messages)) throw new Error("Invalid session");
          if (live) setInitial(rec.messages);
        } catch (e) {
          if (!(
            e instanceof ApiError &&
            e.status === 404 &&
            props.conversation.draft
          ))
            throw e;
          if (!live) return;
          await api.saveSession(props.conversation.id, {
            project: props.project,
            title: "",
            messages: [],
          });
          if (live) setInitial([]);
        }
      } catch {
        if (live)
          setError(
            "No se pudo abrir la conversación. Tu historial no se ha reemplazado.",
          );
      }
    }
    void load();
    return () => {
      live = false;
    };
  }, [props.conversation.id, props.conversation.draft, props.project, retry]);
  if (error)
    return (
      <div className="studio-empty">
        <Icon name="alert" size={25} />
        <p role="alert">{error}</p>
        <button className="btn primary" onClick={() => setRetry((n) => n + 1)}>
          Volver a intentar
        </button>
      </div>
    );
  if (initial === null)
    return (
      <div className="view-loading" role="status">
        <span className="loading-ring" />
        Preparando tu conversación…
      </div>
    );
  return (
    <ChatThread key={props.conversation.id} {...props} initial={initial} />
  );
}

function ChatThread({
  conversation,
  project,
  onTitle,
  onOpenRun,
  initial,
}: Props & { initial: unknown[] }) {
  const { assistant } = useExperience();
  const agent = useMemo(
    () => new HttpAgent({ url: "/agent", threadId: conversation.threadId }),
    [conversation.threadId],
  );
  // Restoring past messages goes through a history adapter — the runtime reads
  // its load() on mount (not the agent's messages), and the messages it returns
  // both render and become the context for the next run.
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      load: async () =>
        ExportedMessageRepository.fromArray(fromAgUiMessages(initial)),
      append: async () => {}, // saving is handled by the subscribe() below
    }),
    // `initial` is fixed for this mounted thread (keyed by conversation.id).
    [], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const runtime = useAgUiRuntime({ agent, adapters: { history } });
  // The runtime's thread holds the conversation in the right order (parent-linked
  // and deduplicated); the raw agent.messages array does not, so persistence
  // reads from here. A ref keeps the latest runtime without re-subscribing.
  const runtimeRef = useRef(runtime);
  runtimeRef.current = runtime;
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The title is set once — auto-derived from a new chat's first message — and
  // then sticks. An existing chat keeps its stored title; later turns never
  // overwrite it (so a rename would survive too).
  const titleRef = useRef(
    conversation.title && conversation.title !== "New chat"
      ? conversation.title
      : "",
  );

  // Persist the conversation as it grows. Debounced while typing/streaming, but
  // flushed immediately when the thread unmounts (switching chats) or the page
  // unloads (reload/close) with `keepalive`, so the last turn is never lost to a
  // save that was still pending or in flight.
  useEffect(() => {
    let dirty = false; // unsaved change since the last write — avoids re-saving on a pure view
    const save = (keepalive: boolean) => {
      let messages: unknown[];
      try {
        messages = toAgUiMessages(
          runtimeRef.current.thread.getState().messages as never,
        );
      } catch {
        messages = agent.messages as unknown[]; // fallback if the runtime shape shifts
      }
      if (!dirty || messages.length === 0) return;
      dirty = false;
      api
        .saveSession(
          conversation.id,
          {
            project,
            title: titleRef.current || "New chat",
            messages,
            updated_at: Date.now() / 1000,
          },
          { keepalive },
        )
        .catch(() => {
          dirty = true; // failed — let a later flush retry
        });
    };
    const flush = () => {
      if (saveTimer.current) {
        clearTimeout(saveTimer.current);
        saveTimer.current = null;
      }
      save(true);
    };
    const sub = agent.subscribe({
      onMessagesChanged: () => {
        dirty = true;
        if (!titleRef.current) {
          const derived = titleFrom(agent.messages);
          if (derived && derived !== "New chat") {
            titleRef.current = derived;
            onTitle(derived);
          }
        }
        if (saveTimer.current) clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(() => {
          saveTimer.current = null;
          save(false);
        }, 600);
      },
    });
    window.addEventListener("pagehide", flush);
    return () => {
      window.removeEventListener("pagehide", flush);
      const s = sub as unknown as { unsubscribe?: () => void };
      if (typeof sub === "function") (sub as () => void)();
      else s?.unsubscribe?.();
      if (saveTimer.current) clearTimeout(saveTimer.current);
      save(false); // Normal fetch on conversation switch; keepalive is only for pagehide.
    };
  }, [agent, conversation.id, project, onTitle]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {assistant.instructions && (
        <CartridgeInstructions text={assistant.instructions} />
      )}
      <div className="chat-center">
        <ThreadPrimitive.Root className="thread">
          <ThreadPrimitive.Viewport className="thread-viewport" autoScroll>
            <ThreadPrimitive.Empty>
              <div className="chat-empty">
                <div className="chat-empty-mark">
                  <Icon name="spark" size={30} />
                </div>
                <h2>{assistant.headline}</h2>
                <p className="muted">{assistant.intro}</p>
                <div className="suggestions">
                  {assistant.suggestions.map((s) => (
                    <ThreadPrimitive.Suggestion
                      key={s}
                      className="suggestion"
                      prompt={s}
                      method="replace"
                    >
                      {s}
                    </ThreadPrimitive.Suggestion>
                  ))}
                </div>
              </div>
            </ThreadPrimitive.Empty>

            <div className="thread-column">
              <ThreadPrimitive.Messages
                components={{ UserMessage, AssistantMessage }}
              />
              <ThreadPrimitive.If running>
                <div className="thread-running">
                  <span className="cavatar" aria-hidden>
                    ◆
                  </span>
                  <TypingDots />
                  <span className="muted">el equipo está trabajando…</span>
                </div>
              </ThreadPrimitive.If>
            </div>
          </ThreadPrimitive.Viewport>

          <div className="composer-wrap">
            <ProjectArtifacts project={project} />
            <SessionRuns
              project={project}
              session={conversation.threadId}
              onOpen={onOpenRun}
            />
            <ComposerPrimitive.Root className="ccomposer">
              <ComposerPrimitive.Input
                className="ccomposer-input"
                placeholder={assistant.placeholder}
                aria-label="Mensaje al asistente"
                rows={1}
              />
              <ComposerPrimitive.Send
                className="ccomposer-send"
                aria-label="Enviar mensaje"
              >
                <Icon name="arrow" size={18} />
              </ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
            <div className="composer-hint faint">
              La IA puede cometer errores. Revisa la evidencia antes de tomar
              decisiones.
            </div>
          </div>
        </ThreadPrimitive.Root>
      </div>
    </AssistantRuntimeProvider>
  );
}

export default ChatCenter;
