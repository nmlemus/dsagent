# dsagent-ui

The web UI for DSAgent. Talks to `dsagent serve` over
[AG-UI](https://docs.ag-ui.com) through CopilotKit.

```bash
# terminal 1 — the agent
dsagent serve                     # http://127.0.0.1:8000/agent

# terminal 2 — the UI
cd ui && npm install && npm run dev   # http://localhost:3000
```

`DSAGENT_URL` overrides where the runtime looks for the agent.

This is the shell only: chat on the left, an empty canvas on the right. The
canvas, the gate card and the DAG view come next — see `../docs/ui-slice.md` §4.
