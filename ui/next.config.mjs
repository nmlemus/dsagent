/**
 * The agent server is proxied, not fetched cross-origin.
 *
 * The canvas reads `.md` and `.csv` with `fetch`, which is subject to CORS —
 * `<iframe>` and `<img>` are not, which is why the first canvas showed figures and
 * failed on text with "Failed to fetch". Rewriting onto `dsagent serve` makes
 * every artifact and every API call same-origin, so nothing has to be relaxed on
 * the Python side and there is no CORS policy to get wrong later.
 *
 * The prefix is `/dsa` and not `/runs`, because `/runs/<id>` is a *page* in this
 * app: a rewrite on that path would proxy the run screen away to the API.
 */
const DSAGENT_ORIGIN = process.env.DSAGENT_ORIGIN ?? "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [{ source: "/dsa/:path*", destination: `${DSAGENT_ORIGIN}/:path*` }];
  },
};

export default nextConfig;
