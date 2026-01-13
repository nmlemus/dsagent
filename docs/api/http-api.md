# HTTP API Reference

DSAgent provides a REST API with Server-Sent Events (SSE) for building custom UIs and integrations.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All endpoints require an API key passed in the header:

```bash
X-API-Key: your-api-key
```

Or as a query parameter:

```
?api_key=your-api-key
```

---

## Sessions

### Create Session

```http
POST /sessions
```

**Request Body:**
```json
{
  "name": "My Analysis Session",
  "model": "gpt-4o",
  "hitl_mode": "none"
}
```

**Response:**
```json
{
  "id": "20260112_215333_b5537c",
  "name": "My Analysis Session",
  "status": "active",
  "created_at": "2026-01-12T21:53:33.204Z",
  "updated_at": "2026-01-12T21:53:33.204Z",
  "message_count": 0,
  "kernel_variables": 0,
  "workspace_path": "/workspace/sessions/20260112_215333_b5537c"
}
```

### List Sessions

```http
GET /sessions
```

### Get Session

```http
GET /sessions/{session_id}
```

### Delete Session

```http
DELETE /sessions/{session_id}
```

---

## Chat

### Send Message (Synchronous)

For simple integrations that don't need real-time updates:

```http
POST /sessions/{session_id}/chat
Content-Type: application/json

{
  "message": "Analyze the iris dataset"
}
```

**Response:**
```json
{
  "content": "I'll analyze the iris dataset...",
  "code": "import pandas as pd\ndf = pd.read_csv('data/iris.csv')",
  "execution_result": {
    "stdout": "   sepal.length  sepal.width ...",
    "stderr": "",
    "error": null,
    "images": [],
    "success": true
  },
  "plan": {
    "steps": [...],
    "completed_steps": 1,
    "total_steps": 3,
    "is_complete": false
  },
  "has_answer": false,
  "answer": null,
  "is_complete": false
}
```

---

### Send Message (Streaming) - Recommended for UIs

```http
POST /sessions/{session_id}/chat/stream
Content-Type: application/json

{
  "message": "Analyze the iris dataset and create visualizations"
}
```

**Response:** Server-Sent Events stream

---

## SSE Events Reference

The streaming endpoint emits granular events that allow UIs to show real-time progress.

### Event Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     ROUND N                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  thinking ──► llm_response ──► plan ──► code_executing     │
│                                              │              │
│                                              ▼              │
│                                        code_result          │
│                                              │              │
│                                              ▼              │
│                                       round_complete        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                        (repeat for each round)
                              │
                              ▼
                      ┌──────────────┐
                      │     done     │
                      └──────────────┘
```

### Event Types

#### `thinking`

Emitted when the LLM starts processing. Use this to show a loading indicator.

```
event: thinking
data: {"message": "Processing..."}
```

**UI Action:** Show spinner or "Thinking..." indicator

---

#### `llm_response`

Emitted when the LLM response is received (before code execution).

```
event: llm_response
data: {
  "content": "I'll analyze the iris dataset. Let me start by loading the data...\n\n<plan>\n1. [ ] Load data\n2. [ ] Create visualizations\n</plan>\n\n<code>\nimport pandas as pd\ndf = pd.read_csv('data/iris.csv')\n</code>"
}
```

**UI Action:** Display the agent's response text (you may want to parse and hide `<plan>`, `<code>` tags)

---

#### `plan`

Emitted when a plan is extracted from the response.

```
event: plan
data: {
  "steps": [
    {"number": 1, "description": "Load and explore data", "completed": false},
    {"number": 2, "description": "Create visualizations", "completed": false},
    {"number": 3, "description": "Save to artifacts/", "completed": false}
  ],
  "raw_text": "1. [ ] Load and explore data\n2. [ ] Create visualizations\n3. [ ] Save to artifacts/",
  "total_steps": 3,
  "completed_steps": 0,
  "is_complete": false
}
```

**UI Action:** Update plan panel with steps and progress bar

---

#### `code_executing`

Emitted just before code is executed in the kernel.

```
event: code_executing
data: {
  "code": "import pandas as pd\ndf = pd.read_csv('data/iris.csv')\ndf.head()"
}
```

**UI Action:** Show code block with "Executing..." badge

---

#### `code_result`

Emitted after code execution completes.

```
event: code_result
data: {
  "stdout": "   sepal.length  sepal.width  petal.length  petal.width variety\n0           5.1          3.5           1.4          0.2  Setosa\n...",
  "stderr": "",
  "error": null,
  "images": [
    {"format": "png", "data": "iVBORw0KGgoAAAANSUhEUgAAA..."}
  ],
  "success": true
}
```

**UI Action:**
- Show output in console panel
- Render images from base64
- Update code block badge to "Success" or "Error"

---

#### `round_complete`

Emitted at the end of each autonomous execution round. Contains the full response for that round.

```
event: round_complete
data: {
  "round": 1,
  "content": "Full LLM response text...",
  "code": "import pandas as pd\n...",
  "execution_result": {
    "stdout": "...",
    "stderr": "",
    "error": null,
    "images": [],
    "success": true
  },
  "plan": {
    "steps": [...],
    "completed_steps": 1,
    "total_steps": 3,
    "is_complete": false
  },
  "has_answer": false,
  "answer": null,
  "thinking": null,
  "is_complete": false
}
```

**UI Action:** This is the "complete" event for the round - useful for logging or if you prefer to wait for complete data rather than granular events.

---

#### `done`

Emitted when the entire task is complete.

```
event: done
data: {}
```

**UI Action:**
- Hide loading indicators
- Enable input for new message
- Optionally show "Task Complete" notification

---

#### `error`

Emitted if an error occurs during processing.

```
event: error
data: {
  "error": "Session not found"
}
```

**UI Action:** Show error notification/toast

---

## Client Examples

### JavaScript/TypeScript

```typescript
async function streamChat(sessionId: string, message: string) {
  const response = await fetch(`/api/v1/sessions/${sessionId}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key',
    },
    body: JSON.stringify({ message }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        const eventType = line.slice(7);
        continue;
      }
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        handleEvent(eventType, data);
      }
    }
  }
}

function handleEvent(type: string, data: any) {
  switch (type) {
    case 'thinking':
      showSpinner();
      break;
    case 'llm_response':
      appendMessage(data.content);
      break;
    case 'plan':
      updatePlanPanel(data);
      break;
    case 'code_executing':
      showCodeBlock(data.code, 'executing');
      break;
    case 'code_result':
      updateCodeResult(data);
      renderImages(data.images);
      break;
    case 'done':
      hideSpinner();
      enableInput();
      break;
    case 'error':
      showError(data.error);
      break;
  }
}
```

### Using EventSource (alternative)

```typescript
// Note: EventSource only supports GET, so you'd need a different approach
// or use a library like eventsource-parser for POST requests
```

### React Hook Example

```typescript
import { useState, useCallback } from 'react';

interface StreamState {
  isLoading: boolean;
  plan: Plan | null;
  code: string | null;
  codeStatus: 'idle' | 'executing' | 'success' | 'error';
  output: string;
  images: Image[];
  error: string | null;
}

function useChatStream(sessionId: string) {
  const [state, setState] = useState<StreamState>({
    isLoading: false,
    plan: null,
    code: null,
    codeStatus: 'idle',
    output: '',
    images: [],
    error: null,
  });

  const sendMessage = useCallback(async (message: string) => {
    setState(s => ({ ...s, isLoading: true, error: null }));

    try {
      const response = await fetch(`/api/v1/sessions/${sessionId}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      // ... parse SSE events and update state

    } catch (err) {
      setState(s => ({ ...s, error: err.message, isLoading: false }));
    }
  }, [sessionId]);

  return { ...state, sendMessage };
}
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/SESSION_ID/chat/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "Analyze iris.csv"}' \
  --no-buffer
```

---

## Message History

### Get Messages

```http
GET /sessions/{session_id}/messages?limit=50&offset=0&role=assistant
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Max messages to return (1-200) |
| `offset` | int | 0 | Skip N messages |
| `role` | string | null | Filter by role: `user`, `assistant`, `execution`, `system` |

**Response:**
```json
{
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "Analyze the iris dataset",
      "timestamp": "2026-01-12T21:54:44.673Z",
      "metadata": {}
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "I'll analyze the iris dataset...",
      "timestamp": "2026-01-12T21:54:47.900Z",
      "metadata": {
        "has_code": true,
        "has_plan": true
      }
    }
  ],
  "total": 10,
  "has_more": false
}
```

### Get Conversation Turns (Recommended for UI)

Returns conversation history as structured turns, matching the `round_complete` SSE event format.
**Use this endpoint to load historical messages** so the UI can render them identically to live streaming.

!!! info "Data Source"
    This endpoint reads from `{workspace}/logs/events.jsonl` which contains the complete
    event history including all LLM responses, code executions, and plan updates.
    This ensures the UI receives the full conversation history, even after internal
    summarization events.

```http
GET /sessions/{session_id}/turns?limit=50&offset=0
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Max turns to return (1-200) |
| `offset` | int | 0 | Skip N turns |

**Response:**
```json
{
  "turns": [
    {
      "round": 1,
      "timestamp": "2026-01-12T21:54:47.900Z",
      "user_message": "Analyze the iris dataset and create visualizations",
      "content": "I'll analyze the iris dataset...\n\n<plan>\n1. [ ] Load data\n</plan>\n\n<code>\nimport pandas as pd\n</code>",
      "code": "import pandas as pd\ndf = pd.read_csv('data/iris.csv')",
      "execution_result": {
        "stdout": "   sepal.length  sepal.width...",
        "stderr": "",
        "error": null,
        "images": [],
        "success": true
      },
      "plan": {
        "steps": [
          {"number": 1, "description": "Load data", "completed": false}
        ],
        "total_steps": 3,
        "completed_steps": 0,
        "is_complete": false
      },
      "has_answer": false,
      "answer": null,
      "thinking": null,
      "is_complete": false
    },
    {
      "round": 2,
      "timestamp": "2026-01-12T21:54:52.100Z",
      "user_message": null,
      "content": "Now I'll create visualizations...",
      "code": "import matplotlib.pyplot as plt\n...",
      "execution_result": {...},
      "plan": {...},
      "has_answer": false,
      "answer": null,
      "thinking": null,
      "is_complete": false
    }
  ],
  "total": 6,
  "has_more": false
}
```

**Note:** `user_message` is `null` for autonomous continuation rounds (when the agent continues without user input).

**UI Usage Pattern:**
```typescript
// On session load, fetch historical turns
const history = await fetch(`/api/v1/sessions/${sessionId}/turns`);
const { turns } = await history.json();

// Render each turn using the same component as round_complete events
turns.forEach(turn => renderRoundComplete(turn));

// For live streaming, connect to SSE
const eventSource = connectToStream(sessionId);
eventSource.on('round_complete', renderRoundComplete);
```

---

## Files & Artifacts

### List Files in Data Directory

```http
GET /sessions/{session_id}/files?path=data
```

### Upload File

```http
POST /sessions/{session_id}/files
Content-Type: multipart/form-data

file: (binary)
path: data/
```

### Get Artifact

```http
GET /sessions/{session_id}/artifacts/{filename}
```

Returns the file content (image, CSV, etc.)

---

## Data Schemas

### ChatResponseModel

```typescript
interface ChatResponseModel {
  content: string;              // Full LLM response text
  code: string | null;          // Extracted code (if any)
  execution_result: ExecutionResult | null;
  plan: PlanResponse | null;
  has_answer: boolean;          // True if contains <answer> tag
  answer: string | null;        // Extracted answer text
  thinking: string | null;      // Model's thinking (if available)
  is_complete: boolean;         // True when task is done
}
```

### ExecutionResult

```typescript
interface ExecutionResult {
  stdout: string;               // Standard output
  stderr: string;               // Standard error
  error: string | null;         // Error message if failed
  images: Image[];              // Generated images (base64)
  success: boolean;             // True if code ran successfully
}

interface Image {
  format: string;               // "png", "jpeg", etc.
  data: string;                 // Base64 encoded image data
}
```

### PlanResponse

```typescript
interface PlanResponse {
  steps: PlanStep[];
  raw_text: string;             // Original plan text
  total_steps: number;
  completed_steps: number;
  is_complete: boolean;         // True when all steps done
}

interface PlanStep {
  number: number;
  description: string;
  completed: boolean;
}
```

---

## UI Component Mapping

| SSE Event | Suggested UI Component |
|-----------|----------------------|
| `thinking` | Loading spinner / "Thinking..." text |
| `llm_response` | Chat message bubble |
| `plan` | Collapsible plan panel with checkboxes |
| `code_executing` | Code block with syntax highlighting + "Running" badge |
| `code_result` | Output console + image gallery |
| `round_complete` | (Optional) Round divider |
| `done` | Enable input, hide spinner |
| `error` | Toast notification / error banner |

---

## Running the Server

```bash
# Start the API server
dsagent serve --port 8000

# With custom workspace
dsagent serve --port 8000 --workspace ./my-workspace

# With API key
DSAGENT_API_KEY=my-secret-key dsagent serve
```

---

## Next Steps

- [Python API Reference](overview.md)
- [ConversationalAgent](conversational-agent.md)
- [Examples](../examples/data-analysis.md)
