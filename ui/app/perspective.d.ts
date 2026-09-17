/**
 * The inline Perspective bundles have no types of their own.
 *
 * They are the same modules as `@finos/perspective` and
 * `@finos/perspective-viewer`, built with their WebAssembly embedded in the
 * JavaScript rather than fetched beside it — which is what lets a pivot work
 * with no asset route and no request to anybody else's server. The packages
 * type their *default* entry points only, so the shapes actually used are
 * declared here.
 */

declare module "@finos/perspective/dist/esm/perspective.inline.js" {
  export type Table = unknown;
  export type Client = { table(data: unknown): Promise<Table> };
  const perspective: { worker(): Promise<Client> };
  export default perspective;
}

declare module "@finos/perspective-viewer/dist/esm/perspective-viewer.inline.js";
