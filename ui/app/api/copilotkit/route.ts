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
