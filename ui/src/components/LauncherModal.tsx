import { useEffect } from "react";
import { Launcher } from "./Launcher";

export function LauncherModal({
  project,
  session,
  onLaunched,
  onClose,
}: {
  project: string;
  session: string;
  onLaunched: (runId: string) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="col-head">
          <span>New run</span>
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <Launcher project={project} session={session} onLaunched={onLaunched} />
      </div>
    </div>
  );
}
