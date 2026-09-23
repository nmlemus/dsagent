import type { FileSource } from "./api";

// Markdown files resolve figures beside the document; chat messages resolve
// them from the project workspace. Neither should resolve against the SPA URL.
export function artifactImageUrl(
  reference: string,
  source: Pick<FileSource, "url">,
  documentPath = "",
): string {
  if (
    /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(reference) ||
    /^\/(?:projects|runs|runs-x)\//.test(reference)
  ) {
    return reference;
  }

  try {
    const root = "https://workspace.invalid/";
    const resolved = new URL(reference, new URL(documentPath || ".", root));
    if (resolved.origin !== new URL(root).origin) return reference;
    const path = resolved.pathname
      .split("/")
      .map(decodeURIComponent)
      .join("/")
      .replace(/^\/+/, "");
    return source.url(path) + resolved.search + resolved.hash;
  } catch {
    return reference;
  }
}
