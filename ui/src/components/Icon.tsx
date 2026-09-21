import type { CSSProperties } from "react";
import { COMPANY } from "../lib/brand";

const paths = {
  grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  chat: "M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-2 2V11.5A8.5 8.5 0 0 1 10.5 3h2a8.5 8.5 0 0 1 8.5 8.5ZM7 9h10M7 13h6",
  chart: "M4 3v17h17M8 15v-4M13 15V6M18 15V9",
  data: "M20 6c0 2-3.6 3-8 3S4 8 4 6s3.6-3 8-3 8 1 8 3ZM4 6v12c0 2 3.6 3 8 3s8-1 8-3V6M4 12c0 2 3.6 3 8 3s8-1 8-3",
  file: "M14 2H5v20h14V7l-5-5ZM14 2v6h5M8 12h8M8 16h5",
  plus: "M12 5v14M5 12h14",
  arrow: "M5 12h14M13 6l6 6-6 6",
  search: "M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z",
  chevron: "m9 5 7 7-7 7",
  down: "m6 9 6 6 6-6",
  close: "m6 6 12 12M6 18 18 6",
  menu: "M4 6h16M4 12h16M4 18h16",
  panel: "M3 3h18v18H3zM9 3v18",
  settings:
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2",
  upload: "M12 16V3M7 8l5-5 5 5M4 15v6h16v-6",
  download: "M12 3v13M7 11l5 5 5-5M4 17v4h16v-4",
  play: "m8 4 12 8-12 8V4Z",
  check: "m5 12 4 4L19 6",
  clock: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM12 7v5l3 2",
  spark: "m12 2 2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7L12 2Z",
  external: "M14 3h7v7M21 3 10 14M10 3H3v18h18v-7",
  shield: "m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3ZM8 12l3 3 5-6",
  trash: "M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7",
  alert: "m12 3 10 18H2L12 3ZM12 9v5M12 17v1",
  refresh: "M20 8a9 9 0 1 0 0 8M20 3v5h-5",
  folder: "M3 5h7l2 3h9v12H3V5Z",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v2M12 20v2M2 12h2M20 12h2M5 5l1 1M18 18l1 1M5 19l1-1M18 6l1-1",
  moon: "M20 15A9 9 0 0 1 9 4a9 9 0 1 0 11 11Z",
  monitor: "M3 3h18v14H3zM8 21h8M12 17v4",
} as const;
export type IconName = keyof typeof paths;
// A cartridge names its workflow icons from this set; anything else falls back.
export const iconName = (
  name: string | null | undefined,
  fallback: IconName = "file",
): IconName => (name && name in paths ? (name as IconName) : fallback);
export function Icon({
  name,
  size = 20,
  style,
}: {
  name: IconName;
  size?: number;
  style?: CSSProperties;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={style}
    >
      <path d={paths[name]} />
    </svg>
  );
}
export function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 32 32" fill="none">
        <path
          d="M8 6h8a10 10 0 0 1 0 20H8V6Z"
          stroke="currentColor"
          strokeWidth="3"
        />
        <path d="M4 12h13M4 20h13" stroke="currentColor" strokeWidth="3" />
      </svg>
    </span>
  );
}

// The company signature: present, constant and deliberately quiet. It is drawn
// with type rather than an asset, because the UI ships no remote fonts or
// images; the real wordmark can replace it without moving anything around.
export function CompanySignature() {
  return (
    <span className="company-signature">
      <span aria-hidden="true">
        &lt;<b>ai</b>/&gt;labs
      </span>
      <span className="sr-only">{COMPANY}</span>
    </span>
  );
}
