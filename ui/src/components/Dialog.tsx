import { type ReactNode, useEffect, useId, useRef } from "react";
import { useExperience } from "../lib/experience";
import { Icon } from "./Icon";

// Native modal: focus containment, inert background, Escape and focus restoration.
export function Dialog({
  title,
  onClose,
  children,
  className = "",
  chrome = true,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
  chrome?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const brand = useExperience().title;
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const dialog = ref.current!;
    const previous = document.activeElement as HTMLElement | null;
    dialog.showModal();
    return () => {
      dialog.close();
      previous?.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      className={`studio-dialog ${className}`}
      aria-labelledby={chrome ? titleId : undefined}
      aria-label={chrome ? undefined : title}
      onCancel={(event) => {
        event.preventDefault();
        closeRef.current();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          const r = event.currentTarget.getBoundingClientRect();
          if (
            event.clientX < r.left ||
            event.clientX > r.right ||
            event.clientY < r.top ||
            event.clientY > r.bottom
          )
            closeRef.current();
        }
      }}
    >
      {chrome && (
        <div className="dialog-header">
          <div>
            <span className="eyebrow">{brand.toUpperCase()} · WORKSPACE</span>
            <h2 id={titleId}>{title}</h2>
          </div>
          <button
            className="icon-btn"
            onClick={onClose}
            aria-label="Cerrar diálogo"
          >
            <Icon name="close" />
          </button>
        </div>
      )}
      {children}
    </dialog>
  );
}
