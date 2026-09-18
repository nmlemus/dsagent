import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";
const KEY = "ds.theme";

function applyTheme(t: Theme) {
  const root = document.documentElement;
  if (t === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", t);
}

export function loadTheme(): Theme {
  try {
    return (localStorage.getItem(KEY) as Theme) || "system";
  } catch {
    return "system";
  }
}

// Apply the saved theme at startup (called from main).
export function initTheme() {
  applyTheme(loadTheme());
}

export function Settings({ onClose }: { onClose: () => void }) {
  const [theme, setTheme] = useState<Theme>(loadTheme);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  function pick(t: Theme) {
    setTheme(t);
    applyTheme(t);
    try {
      localStorage.setItem(KEY, t);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="col-head">
          <span>Settings</span>
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="col-body stack">
          <div className="field">
            <label>Theme</label>
            <div className="seg">
              {(["system", "light", "dark"] as Theme[]).map((t) => (
                <button key={t} className={`seg-btn ${theme === t ? "active" : ""}`} onClick={() => pick(t)}>
                  {t}
                </button>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Account</label>
            <div className="row" style={{ gap: 10 }}>
              <span className="user-avatar">N</span>
              <div>
                <div>You</div>
                <div className="faint" style={{ fontSize: 12 }}>Single-user · local instance (no login)</div>
              </div>
            </div>
          </div>

          <div className="muted" style={{ fontSize: 12 }}>
            Chat history is stored in this browser only (not synced). Runs live on the server and
            are shared across tabs, reloads and the CLI.
          </div>
        </div>
      </div>
    </div>
  );
}
