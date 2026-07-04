"""aipsy-bench — open-source psychological-safety benchmark for conversational AI.

Public surface is intentionally thin; import submodules directly. The benchmark
content is frozen in ``data/v1/`` and SHA-256 verified at runtime (see ``bundle``).
"""

from __future__ import annotations

from .version import __version__

__all__ = ["__version__"]
