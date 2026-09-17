import next from "eslint-config-next";

/**
 * `next lint` is gone in Next 16; ESLint runs directly, in flat config.
 * The Next preset is the whole rule set — this project adds nothing to it,
 * because a house style nobody agreed to is just noise in a diff.
 */
const config = [
  { ignores: [".next/**", "node_modules/**", "fixtures/**"] },
  ...next,
];

export default config;
