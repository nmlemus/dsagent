// The active experience: which cartridge's words the workspace speaks.
//
// The UI's structure is fixed — home, assistant, runs, resources, deliverables.
// Everything a user reads inside it comes from the cartridge the active project
// is bound to (`/cartridges` → `experience`). Nothing in `ui/src` may name a
// domain; a component that needs a word asks `useExperience()` for it.

import { createContext, useContext } from "react";
import type {
  CartridgeInfo,
  Experience,
  PersonaInfo,
  RunSummary,
  StepShape,
  WorkflowShape,
} from "./types";

// Shown before `/cartridges` answers, or when it cannot. Generic on purpose:
// the same defaults `dsagent.cartridge.models.Experience` carries.
export const DEFAULT_EXPERIENCE: Experience = {
  version: 1,
  title: "DSAgent",
  tagline: "Tu equipo, en tu espacio de trabajo.",
  nouns: {
    run: { singular: "trabajo", plural: "trabajos" },
    resource: { singular: "recurso", plural: "recursos" },
    deliverable: { singular: "entregable", plural: "entregables" },
  },
  nav: {
    overview: "Inicio",
    chat: "Asistente",
    runs: "Trabajos",
    resources: "Recursos",
    deliverables: "Entregables",
  },
  home: {
    eyebrow: "TU ESPACIO DE TRABAJO",
    headline: "Tu próximo resultado empieza aquí.",
    intro:
      "Reúne recursos, ejecuta trabajos y revisa entregables en un solo lugar.",
    stages: ["Recursos", "Trabajo", "Entregables"],
    steps: [
      {
        title: "Trae tus recursos",
        text: "Los archivos con los que trabajará el equipo.",
      },
      {
        title: "Define el objetivo",
        text: "Elige un trabajo o conversa con el asistente.",
      },
      {
        title: "Revisa los entregables",
        text: "Aprueba cada revisión y explora el resultado.",
      },
    ],
  },
  assistant: {
    headline: "¿En qué trabajamos hoy?",
    intro: "Un equipo de especialistas listo para ayudarte en este proyecto.",
    placeholder: "Escribe tu mensaje…",
    suggestions: [],
    instructions: "",
  },
  resources: {
    intro: "Sube una vez. Reutiliza en cada trabajo de este proyecto.",
    formats: [],
  },
};

export interface WorkspaceConfig {
  experience: Experience;
  cartridges: CartridgeInfo[];
  workflows: WorkflowShape[];
  personas: PersonaInfo[];
  // The cartridge the active project is bound to ("" = any loaded cartridge).
  cartridge: string;
  loading: boolean;
}

export const DEFAULT_CONFIG: WorkspaceConfig = {
  experience: DEFAULT_EXPERIENCE,
  cartridges: [],
  workflows: [],
  personas: [],
  cartridge: "",
  loading: true,
};

export const WorkspaceConfigContext =
  createContext<WorkspaceConfig>(DEFAULT_CONFIG);

export const useWorkspaceConfig = () => useContext(WorkspaceConfigContext);
export const useExperience = () =>
  useContext(WorkspaceConfigContext).experience;

// The cartridge a project is bound to, else the first one loaded, else nothing.
export function cartridgeFor(
  cartridges: CartridgeInfo[],
  cartridge: string,
): CartridgeInfo | undefined {
  return cartridges.find((c) => c.name === cartridge) ?? cartridges[0];
}

// A bound project only offers its cartridge's workflows; an unbound one (the
// default project, or one created before cartridges were a choice) sees all.
export function workflowsFor(
  workflows: WorkflowShape[],
  cartridge: string,
): WorkflowShape[] {
  return cartridge
    ? workflows.filter((w) => w.cartridge === cartridge)
    : workflows;
}

export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export const humanize = (id: string) => id.replace(/[-_]/g, " ");

export const workflowTitle = (w: WorkflowShape) => w.title || humanize(w.name);

export const workflowSummary = (w: WorkflowShape) => w.summary || w.description;

export const stepTitle = (s: StepShape) => s.title || humanize(s.id);

// What a run is called: the input the workflow declared as its title, else the
// workflow's own title. Never the name of an input — that is the cartridge's.
export function runTitle(r: RunSummary, wf?: WorkflowShape | null): string {
  const key = wf?.title_input;
  const value = key ? r.inputs[key] : undefined;
  if (typeof value === "string" && value.trim()) return value;
  if (typeof r.inputs.title === "string" && r.inputs.title.trim())
    return r.inputs.title;
  return wf ? workflowTitle(wf) : humanize(r.workflow);
}
