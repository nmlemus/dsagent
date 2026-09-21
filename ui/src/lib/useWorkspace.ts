import { useEffect, useState } from "react";
import { api } from "./api";
import type { RunSummary } from "./types";
export type Dataset = { name: string; bytes: number };
export type Artifact = { path: string; bytes: number; mtime: number };
export function useWorkspace(project: string, refresh: number) {
  const [state, setState] = useState<{
    project: string;
    runs: RunSummary[];
    datasets: Dataset[];
    artifacts: Artifact[];
    loading: boolean;
    error: string | null;
  }>({
    project,
    runs: [],
    datasets: [],
    artifacts: [],
    loading: true,
    error: null,
  });
  useEffect(() => {
    let live = true,
      fetching = false;
    setState({
      project,
      runs: [],
      datasets: [],
      artifacts: [],
      loading: true,
      error: null,
    });
    const load = async () => {
      if (fetching) return;
      fetching = true;
      const result = await Promise.allSettled([
        api.projectRuns(project, { limit: 100 }),
        api.datasets(project),
        api.projectArtifacts(project),
      ]);
      if (live)
        setState((previous) => ({
          project,
          runs:
            result[0].status === "fulfilled" ? result[0].value : previous.runs,
          datasets:
            result[1].status === "fulfilled"
              ? result[1].value
              : previous.datasets,
          artifacts:
            result[2].status === "fulfilled"
              ? result[2].value
              : previous.artifacts,
          loading: false,
          error: result.some((r) => r.status === "rejected")
            ? "No pudimos actualizar todos los datos. Comprueba la conexión con el servidor y vuelve a intentar."
            : null,
        }));
      fetching = false;
    };
    void load();
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, 8000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [project, refresh]);
  return state.project === project
    ? state
    : {
        project,
        runs: [],
        datasets: [],
        artifacts: [],
        loading: true,
        error: null,
      };
}
