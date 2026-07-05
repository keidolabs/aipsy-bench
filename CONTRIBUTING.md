# Contributing to aipsy-bench

Thanks for your interest! aipsy-bench is developed and **solely maintained by
[Keido Labs](https://www.keidolabs.com)**. Please read this before opening anything.

## The one thing to know: this is a *frozen benchmark*

aipsy-bench's value is **comparable, reproducible scores**. That only works if the measuring
instrument never quietly moves — so the scenarios, the rubric, the judge prompt, the judge
model pins, and the scoring math are **frozen and versioned**. Changing any of them changes
*what a score means*, so they are stewarded centrally as a deliberate, revalidated `data/`
version bump — never a drive-by edit.

## How to contribute right now: **open an issue**

We are **not accepting external pull requests at this time.** The most valuable thing you can
do is **[open an issue](../../issues/new/choose)**. All of these are read and welcome:

- 🐛 **Bug reports** — something crashes, mis-scores infra failures, or behaves wrong.
- 🔌 **Adapter / provider / DX requests** — "I wish it supported X target/provider/flag."
- 🧪 **Benchmark feedback** — a concern about a scenario, rubric metric, or the scoring. This is
  exactly the kind of input we want as an *issue* (so it can inform a future versioned release),
  **not** as a content edit.
- ❓ **Questions** — how to wire your app, interpret a report, pick a judge lane.

If you open a pull request, we may close it with thanks and ask you to file an issue instead.
The contribution policy may open up later; this is the posture for launch.

## Reporting a good bug

Run against the offline self-test first, and include the version so we can reproduce:

```bash
uv sync
uv run aipsy-bench --version
uv run aipsy-bench run --target mock --quick      # offline, no keys — should exit 0/1 + write a report
```

In the issue, include: the exact command, what happened vs. expected, the **judge lane**
(`local` / `single` / `gold`), your OS + hardware (for local-judge issues), and any error output
(`aipsy-bench provenance` output is gold for version/pin context).

## Security

**Do not** file security vulnerabilities as public issues — see [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
