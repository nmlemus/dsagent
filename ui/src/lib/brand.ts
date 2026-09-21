// Three levels, deliberately separated (see the identity review, ID-01):
//
//   AiudaLabs          the company that develops the product — a discreet,
//                      constant signature, never the page's subject.
//   DSAgent            the product. It is the harness itself, so it is the one
//                      name the UI owns: the wordmark and the tab title keep it
//                      whatever cartridge is loaded.
//   <cartridge title>  the active experience. It labels the project, not the
//                      application: switching cartridge changes the vocabulary,
//                      not the product the user thinks they opened.
//
// Neither name below is a domain word — invariant 1 forbids naming a cartridge,
// a persona or a skill, and these two name the company and the harness itself.
// Neither comes from a cartridge either: the cartridge supplies
// `experience.title`, read through `experienceLabel()` and shown by the project.

export const PRODUCT = "DSAgent";
export const COMPANY = "AiudaLabs";

// The label for the active experience, or "" when there is nothing to add — a
// cartridge whose `experience.title` is the product name (the `ds` cartridge
// declares exactly that today) would otherwise render "DSAgent · DSAgent".
export function experienceLabel(title: string): string {
  const label = title.trim();
  return label.toLowerCase() === PRODUCT.toLowerCase() ? "" : label;
}

// What the browser tab says. The product first so a pinned tab is recognisable
// at any width; the experience after it when it adds something.
export function documentTitle(title: string): string {
  const label = experienceLabel(title);
  return label ? `${PRODUCT} · ${label}` : PRODUCT;
}
