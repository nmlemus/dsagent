import { type ReactNode, Suspense, lazy, useEffect, useMemo, useState } from "react";
import { HttpAgent } from "@ag-ui/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  type TextMessagePartProps,
  type ToolCallMessagePartProps,
  ThreadPrimitive,
  useAssistantInstructions,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { api } from "../lib/api";
import { cost } from "../lib/format";
import { isVegaLite } from "../lib/vega-detect";
import type { RunSummary } from "../lib/types";
import { type Conversation, loadMessages, saveMessages, titleFrom } from "../lib/conversations";
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

export function ChatCenter({ conversation, project, onTitle, onOpenRun }: Props) {
  const agent = useMemo(
    () =>
      new HttpAgent({
        url: "/agent",
        threadId: conversation.threadId,
        initialMessages: loadMessages(conversation.id),
      }),
    [conversation.id, conversation.threadId],
  );
  const runtime = useAgUiRuntime({ agent });

  // Tell the server this chat's project, so a run the orchestrator starts here
  // lands in the right project (not `default`).
  useEffect(() => {
    api.registerSession(conversation.threadId, project).catch(() => {});
  }, [conversation.threadId, project]);

  // Persist the conversation as it grows (client owns the thread).
  useEffect(() => {
    const sub = agent.subscribe({
      onMessagesChanged: () => {
        saveMessages(conversation.id, agent.messages);
        onTitle(titleFrom(agent.messages));
      },
    });
    return () => {
      const s = sub as unknown as { unsubscribe?: () => void };
      if (typeof sub === "function") (sub as () => void)();
      else s?.unsubscribe?.();
    };
  }, [agent, conversation.id, onTitle]);

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
              Historial guardado en este navegador · el orquestador puede iniciar runs (míralos en la barra izquierda).
            </div>
          </div>
        </ThreadPrimitive.Root>
      </div>
    </AssistantRuntimeProvider>
  );
}

export default ChatCenter;
