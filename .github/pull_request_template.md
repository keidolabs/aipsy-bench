<!--
  Heads up: aipsy-bench is solely maintained by Keido Labs and we are NOT accepting external
  pull requests at this time. If you're an outside contributor, please close this and open an
  issue instead — https://github.com/keidolabs/aipsy-bench/issues/new/choose — it's the fastest
  path and genuinely appreciated. See CONTRIBUTING.md.

  The checklist below is for maintainer PRs.
-->

## Summary

<!-- What and why. -->

## Maintainer checklist

- [ ] `uv run pytest` green (offline, no keys) and `uv run ruff check .` clean
- [ ] Parity / golden tests still pass (no scoring/parse drift)
- [ ] **Frozen instrument untouched** — no change to `data/v1/` content, judge pins in `spec.py`,
      or scoring math. _(If this IS a deliberate `data/` version bump: MANIFEST regenerated, version
      bumped, revalidation noted, and comparability implications called out.)_
- [ ] No new network dependency in the core/offline path
- [ ] Docs / CHANGELOG updated if user-facing
