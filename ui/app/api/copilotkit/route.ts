import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

// The Python side is the agent. This route is a pass-through: no LLM adapter,
// no prompt, no tools — `dsagent serve` owns all of that, and CopilotKit only
// relays AG-UI events to the browser. Anything that looks like agent behaviour
// belongs in the cartridge, not here.
// Telemetry off in code, not only in `ui/.env`.
//
// CopilotKit 1.72 has no constructor option for it: `isTelemetryDisabled()` in
// `@copilotkit/shared` reads `COPILOTKIT_TELEMETRY_DISABLED` (or `DO_NOT_TRACK`)
// from the environment and nothing else. So the switch is thrown here, in the
// module that builds the runtime, before it is constructed — because `.env` is a
// file someone can delete or forget to copy, and this runs on private data. The
// entry in `ui/.env` stays as documentation of the intent.
process.env.COPILOTKIT_TELEMETRY_DISABLED ??= "true";

const runtime = new CopilotRuntime({
  agents: {
    dsagent: new HttpAgent({
      url: process.env.DSAGENT_URL ?? "http://localhost:8000/agent",
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
