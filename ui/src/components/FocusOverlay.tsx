import type { FileSource } from "../lib/api";
import { basename } from "../lib/format";
import { FileViewer } from "./FileViewer";
import { Dialog } from "./Dialog";
export type FocusTarget = { kind: "file"; path: string };
// One file, filling the window. Used for a run's deliverable and for a file a
// chat wrote, which is why it takes a source rather than a run id.
export function FocusOverlay({
  source,
  target,
  onClose,
}: {
  source: FileSource;
  target: FocusTarget;
  onClose: () => void;
}) {
  return (
    <Dialog
      title={basename(target.path)}
      className="focus-dialog"
      onClose={onClose}
    >
      <div className="focus-body">
        <FileViewer source={source} path={target.path} full />
      </div>
    </Dialog>
  );
}
