import DOMPurify from "dompurify";
import { marked } from "marked";
import type { FileSource } from "./api";
import { artifactImageUrl } from "./artifact-images";

export function markdownHtml(
  text: string,
  source?: FileSource,
  path = "",
): string {
  const clean = DOMPurify.sanitize(marked.parse(text, { async: false }) as string);
  if (!source) return clean;
  const document = new DOMParser().parseFromString(clean, "text/html");
  for (const image of document.querySelectorAll("img[src]")) {
    image.setAttribute(
      "src",
      artifactImageUrl(image.getAttribute("src")!, source, path),
    );
  }
  return document.body.innerHTML;
}
