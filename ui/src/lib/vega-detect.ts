// Cheap check with no vega import, so the file viewer can decide whether a
// .json artifact is a chart before paying to load the (heavy) vega runtime.
export function isVegaLite(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  const schema = typeof v.$schema === "string" ? v.$schema : "";
  if (schema.includes("vega-lite")) return true;
  return "mark" in v || "encoding" in v || "layer" in v || "vconcat" in v || "hconcat" in v;
}
