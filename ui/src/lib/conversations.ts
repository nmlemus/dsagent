import type { Message } from "@ag-ui/client";

// Chats live on the server now (see api.sessions/getSession/saveSession). This
// module is just the UI-side shape + helpers; no localStorage.

export interface Conversation {
  id: string;
  threadId: string; // same as id — the AG-UI thread + the session id
  title: string;
  projectId: string;
  updatedAt: number;
}

export function newConversation(projectId = "default"): Conversation {
  const id = crypto.randomUUID();
  return { id, threadId: id, title: "New chat", projectId, updatedAt: Date.now() };
}

export function fromHeader(h: {
  id: string;
  title: string;
  project: string;
  updated_at: number;
}): Conversation {
  return { id: h.id, threadId: h.id, title: h.title, projectId: h.project, updatedAt: h.updated_at };
}

// A readable title from the first user message.
export function titleFrom(messages: Message[]): string {
  const first = messages.find((m) => (m as { role?: string }).role === "user");
  const content =
    first && typeof (first as { content?: unknown }).content === "string"
      ? (first as { content: string }).content
      : "";
  const t = content.trim().replace(/\s+/g, " ").slice(0, 40);
  return t || "New chat";
}
