import { useEffect, useState } from "react";
import { api } from "./api";
import type { CartridgeInfo, WorkflowShape } from "./types";

// What this server can run and how each cartridge presents itself. Fetched once
// per session (and on an explicit refresh): cartridges are loaded at server
// start and do not change underneath a running UI.
export function useCartridges(refresh: number) {
  const [state, setState] = useState<{
    cartridges: CartridgeInfo[];
    workflows: WorkflowShape[];
    loading: boolean;
    error: string | null;
  }>({ cartridges: [], workflows: [], loading: true, error: null });
  useEffect(() => {
    let live = true;
    api
      .cartridges()
      .then((r) => {
        if (live)
          setState({
            cartridges: r.cartridges,
            workflows: r.workflows,
            loading: false,
            error: null,
          });
      })
      .catch(() => {
        if (live)
          setState((s) => ({
            ...s,
            loading: false,
            error:
              "No se pudo cargar la configuración del servidor. Comprueba que esté disponible.",
          }));
      });
    return () => {
      live = false;
    };
  }, [refresh]);
  return state;
}
