"""Single source of truth for the package version.

Read at runtime by ``aipsy_bench.__init__`` (``__version__``) AND at build time by the
backend (hatchling dynamic version — see ``pyproject.toml``), so the installed
distribution and the runtime ``__version__`` can never drift. Bump here on release
(release cadence: TestPyPI ``0.1.1`` → PyPI ``0.1.2``).
"""

__version__ = "0.1.3"
