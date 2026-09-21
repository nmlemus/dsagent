import { useState } from "react";
import { Dialog } from "./Dialog";
import { Icon, type IconName } from "./Icon";
type Theme = "system" | "light" | "dark";
const KEY = "ds.theme";
function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme =
    theme === "system"
      ? matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : theme;
}
export function loadTheme(): Theme {
  try {
    const t = localStorage.getItem(KEY);
    return t === "dark" || t === "system" ? t : "light";
  } catch {
    return "light";
  }
}
export function initTheme() {
  applyTheme(loadTheme());
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (loadTheme() === "system") applyTheme("system");
  });
}
export function Settings({ onClose }: { onClose: () => void }) {
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const options: { id: Theme; label: string; icon: IconName }[] = [
    { id: "light", label: "Claro", icon: "sun" },
    { id: "dark", label: "Oscuro", icon: "moon" },
    { id: "system", label: "Sistema", icon: "monitor" },
  ];
  return (
    <Dialog title="Tu espacio, a tu manera" onClose={onClose}>
      <div className="dialog-body stack">
        <p className="muted">
          Una experiencia que se adapta a tu forma de trabajar.
        </p>
        <div className="field">
          <span className="field-label">Apariencia</span>
          <div className="theme-options">
            {options.map((t) => (
              <button
                key={t.id}
                aria-pressed={theme === t.id}
                className={`theme-option ${theme === t.id ? "active" : ""}`}
                onClick={() => {
                  setTheme(t.id);
                  applyTheme(t.id);
                  try {
                    localStorage.setItem(KEY, t.id);
                  } catch {
                    /* optional preference */
                  }
                }}
              >
                <Icon name={t.icon} size={24} />
                <strong>{t.label}</strong>
                {theme === t.id && <Icon name="check" size={15} />}
              </button>
            ))}
          </div>
        </div>
        <div className="settings-info">
          <Icon name="folder" />
          <div>
            <strong>Acerca de este espacio</strong>
            <p>
              Los proyectos, conversaciones y trabajos se guardan en el servidor
              de esta instancia. La apariencia y el proyecto seleccionado se
              recuerdan en este navegador.
            </p>
          </div>
        </div>
        <div className="dialog-actions">
          <button className="btn primary" onClick={onClose}>
            Listo
            <Icon name="check" size={16} />
          </button>
        </div>
      </div>
    </Dialog>
  );
}
