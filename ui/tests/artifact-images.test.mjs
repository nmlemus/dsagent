import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

// Use the existing compiler rather than requiring a TS-aware test runner or a
// newer Node runtime. Exercise the actual helper used by both Markdown views.
const source = readFileSync(new URL("../src/lib/artifact-images.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { artifactImageUrl } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const files = (scope) => ({ url: (path) => `/${scope}/files/${path.split("/").map(encodeURIComponent).join("/")}` });

test("report figures stay beside their document in the correct project", () => {
  assert.equal(artifactImageUrl("figs/roi.png", files("projects/p1"), "report/mmm.md"), "/projects/p1/files/report/figs/roi.png");
  assert.equal(artifactImageUrl("figs/roi.png", files("projects/p2"), "report/mmm.md"), "/projects/p2/files/report/figs/roi.png");
});

test("run reports resolve parent paths within their file source", () => {
  assert.equal(artifactImageUrl("../artifacts/plot.png", files("runs/r1"), "report/findings.md"), "/runs/r1/files/artifacts/plot.png");
});

test("chat images and workspace-absolute images use the project root", () => {
  assert.equal(artifactImageUrl("report/figs/plot.png", files("projects/p1")), "/projects/p1/files/report/figs/plot.png");
  assert.equal(artifactImageUrl("/report/figs/plot.png", files("projects/p1"), "report/result.md"), "/projects/p1/files/report/figs/plot.png");
});

test("spaces and encoded filenames are encoded once; query and fragment survive", () => {
  assert.equal(artifactImageUrl("figs/ventas%20año.png?v=2#image", files("runs/r1"), "report/result.md"), "/runs/r1/files/report/figs/ventas%20a%C3%B1o.png?v=2#image");
});

test("external, inline and already scoped images are not rewritten", () => {
  for (const url of ["https://example.com/plot.png", "//example.com/plot.png", "data:image/png;base64,AA==", "/projects/p1/files/plot.png", "/runs/r1/files/plot.png"]) {
    assert.equal(artifactImageUrl(url, files("projects/p2"), "report/result.md"), url);
  }
});
