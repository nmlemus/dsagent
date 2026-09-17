/**
 * The run workspace is proxied, not fetched cross-origin.
 *
 * The canvas reads `.md` and `.csv` with `fetch`, which is subject to CORS —
 * `<iframe>` and `<img>` are not, which is why the first version of the canvas
 * showed images and failed on text with "Failed to fetch". Rewriting `/runs/*`
 * onto `dsagent serve` makes every artifact same-origin, so nothing has to be
 * relaxed on the Python side and there is no CORS policy to get wrong later.
 *
 * The report iframe keeps `sandbox=""`, which denies same-origin access anyway.
 */
const DSAGENT_ORIGIN = process.env.DSAGENT_ORIGIN ?? "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [{ source: "/runs/:path*", destination: `${DSAGENT_ORIGIN}/runs/:path*` }];
  },
};

export default nextConfig;
