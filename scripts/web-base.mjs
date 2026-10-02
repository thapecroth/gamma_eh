export function webBase(environment = process.env) {
  const base = environment.GAMMA_WEB_BASE ?? '/';
  // Keep executable code and model downloads on the site's own origin.
  if (!/^\/(?:[A-Za-z0-9_-]+\/)*$/u.test(base)) {
    throw new Error('GAMMA_WEB_BASE must be an absolute directory path with a trailing slash, such as /gamma_eh/.');
  }
  return base;
}
