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
import { fromAgUiMessages, useAgUiRuntime } from "@assistant-ui/react-ag-ui";
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
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Persist the conversation as it grows — debounced, to the server.
  useEffect(() => {
    const sub = agent.subscribe({
      onMessagesChanged: () => {
        const title = titleFrom(agent.messages);
        onTitle(title);
        if (saveTimer.current) clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(() => {
          api
            .saveSession(conversation.id, {
              project,
              title,
              messages: agent.messages as unknown[],
              updated_at: Date.now() / 1000,
            })
            .catch(() => {});
        }, 600);
      },
    });
    return () => {
      const s = sub as unknown as { unsubscribe?: () => void };
      if (typeof sub === "function") (sub as () => void)();
      else s?.unsubscribe?.();
      if (saveTimer.current) clearTimeout(saveTimer.current);
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
