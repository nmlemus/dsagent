<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

<!-- Everything below is ours; `next dev` only rewrites the block above. -->

# And this is not a standalone app

`ui/` renders what `dsagent serve` sends. See the repository root `CLAUDE.md` for
the invariants that govern the rest of the project, and `docs/ui-slice.md` for the
event and gate contracts this directory consumes.
