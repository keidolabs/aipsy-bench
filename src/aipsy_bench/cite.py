"""``cite`` — BibTeX for the OSF registration + the tool/data version (§13.6).

Trivial; feeds the academic citation flywheel. No network.
"""

from __future__ import annotations

from . import __version__, spec

# OSF registration GUID (a DOI is minted later — see REGISTRATION.md).
OSF_GUID = "4gu6d"
OSF_URL = f"https://osf.io/{OSF_GUID}"


def bibtex() -> str:
    return (
        "@misc{aipsybench,\n"
        "  title        = {aipsy-bench: a psychological-safety benchmark for conversational AI},\n"
        "  howpublished = {Software (Inspect AI). OSF preregistration " + OSF_URL + "},\n"
        f"  note         = {{tool version {__version__}, data version {spec.DATA_VERSION}; "
        "instrument PENDING_VALIDATION},\n"
        f"  url          = {{{OSF_URL}}},\n"
        "}"
    )
