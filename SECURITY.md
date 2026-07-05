# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately, either way:

- **Preferred:** GitHub → the repo's **Security** tab → **Report a vulnerability** (private
  advisory), or
- **Email:** [michael@keidolabs.com](mailto:michael@keidolabs.com) with subject
  `aipsy-bench security`.

Please include a description, reproduction steps, affected version (`aipsy-bench --version`),
and impact. We aim to acknowledge within **5 business days** and will coordinate a fix and
disclosure timeline with you.

## Scope

aipsy-bench runs locally and processes **untrusted input** at several boundaries — submitted
`.eval` logs, HTTP target responses, and YAML config. We especially want to hear about:

- Anything that lets untrusted input escape those boundaries (code execution, path traversal,
  resource exhaustion, deserialization issues).
- Leakage of secrets (provider API keys, `EVAL_SECRET`) into logs, `result.json`, or share cards.
- Ways to make the tool report a wrong/forged score as if it were a genuine frozen-instrument run.

## Supported versions

aipsy-bench is pre-1.0; security fixes land on the latest released version. Please test against
the current release before reporting.
