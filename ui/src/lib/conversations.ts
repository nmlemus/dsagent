import type { Message } from "@ag-ui/client";

// Chat conversations live client-side: assistant-ui is the source of truth for a
// thread (it replays messages to /agent each turn; the server checkpointer only
// spans a single turn's interrupt). So localStorage gives real, working history
// — per browser, not synced. All access is guarded: blocked storage still works,
// it just doesn't remember.

export interface Conversation {
  id: string;
  threadId: string;
  title: string;
  updatedAt: number;
  projectId: string;
}

const LIST_KEY = "ds.chats.v1";
const msgsKey = (id: string) => `ds.chat.${id}`;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode / blocked — history just won't persist */
  }
}

export function listConversations(): Conversation[] {
  return read<Conversation[]>(LIST_KEY, []).sort((a, b) => b.updatedAt - a.updatedAt);
}

export function saveConversations(list: Conversation[]) {
  write(LIST_KEY, list);
}

export function newConversation(projectId = "default"): Conversation {
  const id = crypto.randomUUID();
  return { id, threadId: id, title: "New chat", updatedAt: Date.now(), projectId };
}

export function upsertConversation(conv: Conversation) {
  const list = listConversations().filter((c) => c.id !== conv.id);
  saveConversations([conv, ...list]);
}

export function deleteConversation(id: string) {
  saveConversations(listConversations().filter((c) => c.id !== id));
  try {
    localStorage.removeItem(msgsKey(id));
  } catch {
    /* ignore */
  }
}

export function loadMessages(id: string): Message[] {
  return read<Message[]>(msgsKey(id), []);
}

export function saveMessages(id: string, messages: Message[]) {
  write(msgsKey(id), messages);
}

// A readable title from the first user message.
export function titleFrom(messages: Message[]): string {
  const first = messages.find((m) => (m as { role?: string }).role === "user");
  const content = first && typeof (first as { content?: unknown }).content === "string"
    ? (first as { content: string }).content
    : "";
  const t = content.trim().replace(/\s+/g, " ").slice(0, 40);
  return t || "New chat";
}
