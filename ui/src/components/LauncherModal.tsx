import { Launcher } from "./Launcher";
import { Dialog } from "./Dialog";
import { useExperience } from "../lib/experience";
export function LauncherModal({
  project,
  cartridge,
  session,
  initialDataset,
  onLaunched,
  onClose,
}: {
  project: string;
  cartridge: string;
  session: string;
  initialDataset?: string;
  onLaunched: (runId: string) => void;
  onClose: () => void;
}) {
  const { nouns } = useExperience();
  return (
    <Dialog
      title={`Preparar ${nouns.run.singular}`}
      className="launcher-dialog"
      onClose={onClose}
    >
      <Launcher
        project={project}
        cartridge={cartridge}
        session={session}
        initialDataset={initialDataset}
        onLaunched={onLaunched}
      />
    </Dialog>
  );
}
