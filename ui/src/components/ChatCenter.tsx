import { type ReactNode, Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
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
import { fromAgUiMessages, toAgUiMessages, useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { api } from "../lib/api";
import { cost } from "../lib/format";
import { isVegaLite } from "../lib/vega-detect";
import type { RunSummary } from "../lib/types";
import { type Conversation, titleFrom } from "../lib/conversations";
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
      out.push(<div key={key++} className="md" dangerouslySetInnerHTML={{ __html: mdHtml(before) }} />);
    let spec: unknown = null;
    try {
      spec = JSON.parse(m[1]);
    } catch {
      /* not valid JSON */
    }
    if (spec && isVegaLite(spec)) {
      out.push(
        <Suspense key={key++} fallback={<div className="viewer"><div className="empty">Rendering chart…</div></div>}>
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
    out.push(<div key={key++} className="md" dangerouslySetInnerHTML={{ __html: mdHtml(rest) }} />);
  return <>{out}</>;
}

function ToolFallback({ toolName, status }: ToolCallMessagePartProps) {
  const running = status?.type === "running";
  const label: Record<string, string> = {
    task: "delegating to a persona",
    run_workflow: "starting a workflow",
    run_python: "running code",
    run_skill_script: "running a skill",
    show_chart: "drawing a chart",
    show_table: "building a table",
  };
  return (
    <div className={`tool-call ${running ? "running" : "done"}`}>
      <span className="tool-dot" />
      <span className="mono">{toolName}</span>
      <span className="muted">— {label[toolName] ?? (running ? "working" : "done")}</span>
      {running && <span className="dots"><i /><i /><i /></span>}
    </div>
  );
}

function TypingDots() {
  return <div className="typing"><i /><i /><i /></div>;
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
      <div className="cavatar" aria-hidden>◆</div>
      <div className="cbody">
        <MessagePrimitive.Parts components={{ Text: MarkdownText, tools: { Fallback: ToolFallback } }} />
      </div>
    </MessagePrimitive.Root>
  );
}

const VIZ_INSTRUCTIONS =
  "When you present quantitative results in this chat, VISUALIZE them rather than only " +
  "describing them. For a chart, output a fenced code block with language `vega-lite` " +
  "containing a complete Vega-Lite v5 spec WITH THE DATA INLINE (data.values = [...]); " +
  "aggregate first so the spec stays small, give every encoded field a tooltip, and put one " +
  "short sentence before it. For tabular results use a GitHub-flavoured markdown table.";

function VizInstructions() {
  useAssistantInstructions(VIZ_INSTRUCTIONS);
  return null;
}

const SUGGESTIONS = [
  "¿Qué workflows puedes ejecutar y qué hace cada persona?",
  "Inventa ventas mensuales y muéstralas como gráfica de barras y tabla",
  "Genera un dataset sintético pequeño y grafica una relación causal",
];

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
  const [runs, setRuns] = useState<RunSummary[]>([]);
  useEffect(() => {
    let live = true;
    const load = () =>
      api.projectRuns(project, { session }).then((r) => live && setRuns(r)).catch(() => {});
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
      <span className="faint" style={{ fontSize: 11 }}>Runs de esta sesión:</span>
      {runs.map((r) => (
        <button key={r.run_id} className="run-chip" onClick={() => onOpen(r.run_id)} title={r.run_id}>
          <StatusPill status={r.status} />
          <span className="mono" style={{ fontSize: 11 }}>{r.workflow}</span>
          {r.cost_usd != null && <span className="faint mono" style={{ fontSize: 10 }}>{cost(r.cost_usd)}</span>}
        </button>
      ))}
    </div>
  );
}

// Files the chat wrote into the project workspace — a persona's report, its
// charts. They live per-project (the chat workspace is the project's), so this
// is "what this project has produced", polled so a fresh report shows up. An
// HTML report opens in its own tab (a sandboxed iframe under the composited page
// paints blank — see the runs canvas); everything else opens in a tab too.
function ProjectArtifacts({ project }: { project: string }) {
  const [files, setFiles] = useState<{ path: string; bytes: number; mtime: number }[]>([]);
  useEffect(() => {
    let live = true;
    const load = () =>
      api.projectArtifacts(project).then((f) => live && setFiles(f)).catch(() => {});
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
      <span className="faint" style={{ fontSize: 11 }}>Archivos del proyecto:</span>
      {files.slice(0, 12).map((f) => (
        <a
          key={f.path}
          className="run-chip"
          href={api.projectFileUrl(project, f.path)}
          target="_blank"
          rel="noreferrer"
          title={f.path}
        >
          <span aria-hidden>▤</span>
          <span className="mono" style={{ fontSize: 11 }}>{f.path.split("/").pop()}</span>
        </a>
      ))}
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
  useEffect(() => {
    let live = true;
    setInitial(null);
    api
      .getSession(props.conversation.id)
      .then((rec) => {
        if (live) setInitial(Array.isArray(rec.messages) ? rec.messages : []);
      })
      .catch(() => {
        if (!live) return;
        setInitial([]);
        // New chat: register its project (empty, untitled — not listed) so a run
        // started here resolves to it before the first message is saved.
        api
          .saveSession(props.conversation.id, { project: props.project, title: "", messages: [] })
          .catch(() => {});
      });
    return () => {
      live = false;
    };
  }, [props.conversation.id, props.project]);

  if (initial === null)
    return (
      <div className="chat-center">
        <div className="thread-viewport">
          <div className="empty">Cargando chat…</div>
        </div>
      </div>
    );
  return <ChatThread key={props.conversation.id} {...props} initial={initial} />;
}

function ChatThread({
  conversation,
  project,
  onTitle,
  onOpenRun,
  initial,
}: Props & { initial: unknown[] }) {
  const agent = useMemo(
    () => new HttpAgent({ url: "/agent", threadId: conversation.threadId }),
    [conversation.threadId],
  );
  // Restoring past messages goes through a history adapter — the runtime reads
  // its load() on mount (not the agent's messages), and the messages it returns
  // both render and become the context for the next run.
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      load: async () => ExportedMessageRepository.fromArray(fromAgUiMessages(initial)),
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
    conversation.title && conversation.title !== "New chat" ? conversation.title : "",
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
        messages = toAgUiMessages(runtimeRef.current.thread.getState().messages as never);
      } catch {
        messages = agent.messages as unknown[]; // fallback if the runtime shape shifts
      }
      if (!dirty || messages.length === 0) return;
      dirty = false;
      api
        .saveSession(
          conversation.id,
          { project, title: titleRef.current || "New chat", messages, updated_at: Date.now() / 1000 },
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
      flush(); // switching away: write any pending change now
    };
  }, [agent, conversation.id, project, onTitle]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <VizInstructions />
      <div className="chat-center">
        <ThreadPrimitive.Root className="thread">
          <ThreadPrimitive.Viewport className="thread-viewport" autoScroll>
            <ThreadPrimitive.Empty>
              <div className="chat-empty">
                <div className="chat-empty-mark">◆</div>
                <h2>What should the team look at?</h2>
                <p className="muted">
                  Talk to the orchestrator — it can answer, delegate to Marie / Noel / Ana /
                  Pablo, or start a workflow. It shows charts and tables inline.
                </p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <ThreadPrimitive.Suggestion key={s} className="suggestion" prompt={s} method="replace" autoSend>
                      {s}
                    </ThreadPrimitive.Suggestion>
                  ))}
                </div>
              </div>
            </ThreadPrimitive.Empty>

            <div className="thread-column">
              <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
              <ThreadPrimitive.If running>
                <div className="thread-running">
                  <span className="cavatar" aria-hidden>◆</span>
                  <TypingDots />
                  <span className="muted">el equipo está trabajando…</span>
                </div>
              </ThreadPrimitive.If>
            </div>
          </ThreadPrimitive.Viewport>

          <div className="composer-wrap">
            <ProjectArtifacts project={project} />
            <SessionRuns project={project} session={conversation.threadId} onOpen={onOpenRun} />
            <ComposerPrimitive.Root className="ccomposer">
              <ComposerPrimitive.Input className="ccomposer-input" placeholder="Message the DS team…" autoFocus rows={1} />
              <ComposerPrimitive.Send className="ccomposer-send" aria-label="Send">↑</ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
            <div className="composer-hint faint">
              Historial guardado en el servidor · el orquestador puede iniciar runs (míralos en la barra izquierda).
            </div>
          </div>
        </ThreadPrimitive.Root>
      </div>
    </AssistantRuntimeProvider>
  );
}

export default ChatCenter;
