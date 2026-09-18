// A column collapsed to a thin vertical spine: an expand button and a rotated
// label so you can still tell what it is and click to bring it back.
export function Spine({
  label,
  icon,
  side,
  onExpand,
}: {
  label: string;
  icon: string;
  side: "left" | "right";
  onExpand: () => void;
}) {
  return (
    <div className="spine" title={`Expand ${label}`}>
      <button onClick={onExpand} aria-label={`Expand ${label}`}>
        {side === "left" ? "»" : "«"}
      </button>
      <button onClick={onExpand} className="spine-label-btn" style={{ flex: 1 }}>
        <span className="spine-label">
          {icon} {label}
        </span>
      </button>
    </div>
  );
}
