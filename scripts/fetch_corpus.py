"""Fetch the ReguLumen public-domain regulatory corpus to disk.

Downloads the focused IP-safe corpus (EU MDR HTML + FDA/IMDRF PDFs) into
``corpus/raw/`` and writes a provenance manifest to ``corpus/manifest.json``.

Usage::

    python scripts/fetch_corpus.py

The script is idempotent: re-running re-downloads all sources and refreshes
the manifest.  Individual source failures are logged as warnings and recorded
in the manifest — the whole script never aborts due to a single source failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CORPUS_RAW = PROJECT_ROOT / "corpus" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "corpus" / "manifest.json"

CORPUS_RAW.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
TODAY = date.today().isoformat()

SOURCES: list[dict[str, str]] = [
    {
        "id": "eu-mdr-2017-745",
        "title": "EU MDR — Regulation (EU) 2017/745 on medical devices",
        "short_title": "EU MDR",
        "source_type": "eurlex_html",
        "jurisdiction": "EU",
        # EUR-Lex fronts this document with an AWS-WAF JS challenge that blocks
        # scripted (httpx) GETs (returns HTTP 202 + interstitial). The file is
        # therefore obtained once via a real browser and preserved on disk; this
        # script will keep an existing valid copy rather than clobbering it. See
        # corpus/README.md for the manual-refresh procedure.
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0745",
        "filename": "eu-mdr-2017-745.html",
        "license_note": (
            "Public EUR-Lex text of Regulation (EU) 2017/745; sourced from "
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0745 "
            f"on {TODAY}. Redistributed corpus excludes copyrighted standards."
        ),
    },
    {
        "id": "fda-premarket-software-functions",
        "title": (
            "FDA — Content of Premarket Submissions for Device Software Functions (2023)"
        ),
        "short_title": "FDA Premarket SW",
        "source_type": "guidance_pdf",
        "jurisdiction": "US-FDA",
        "url": "https://www.fda.gov/media/170714/download",
        "filename": "fda-premarket-software-functions-2023.pdf",
        "license_note": (
            "Public FDA final guidance document (June 2023); sourced from "
            "https://www.fda.gov/media/170714/download "
            f"on {TODAY}. Redistributed corpus excludes copyrighted standards."
        ),
    },
    {
        "id": "fda-samd-clinical-evaluation",
        "title": (
            "FDA — Software as a Medical Device (SaMD): Clinical Evaluation (2017)"
        ),
        "short_title": "FDA SaMD Clinical Eval",
        "source_type": "guidance_pdf",
        "jurisdiction": "US-FDA",
        "url": "https://www.fda.gov/media/122535/download",
        "filename": "fda-samd-clinical-evaluation-2017.pdf",
        "license_note": (
            "Public FDA guidance document (2017); sourced from "
            "https://www.fda.gov/media/122535/download "
            f"on {TODAY}. Redistributed corpus excludes copyrighted standards."
        ),
    },
    {
        "id": "imdrf-samd-key-definitions",
        "title": "IMDRF — Software as a Medical Device (SaMD): Key Definitions (N10, 2013)",
        "short_title": "IMDRF SaMD N10",
        "source_type": "guidance_pdf",
        "jurisdiction": "IMDRF",
        # Official imdrf.org URL sometimes returns 403 from scripted clients;
        # using a stable GMP-compliance.org mirror of the same public document.
        "url": (
            "https://academy.gmp-compliance.org/guidemgr/files/"
            "IMDRF-TECH-131209-SAMD-KEY-DEFINITIONS.PDF"
        ),
        "filename": "imdrf-samd-key-definitions-n10-2013.pdf",
        "license_note": (
            "Public IMDRF guidance document (N10FINAL:2013); sourced from a "
            "public GMP-compliance.org mirror of the IMDRF document "
            f"on {TODAY}. Redistributed corpus excludes copyrighted standards."
        ),
    },
]

# ---------------------------------------------------------------------------
# Module-level constants (hoisted for testability and easy tuning)
# ---------------------------------------------------------------------------

# Reject any downloaded body smaller than this threshold regardless of type.
MIN_BYTES = 10_000

USER_AGENT = (
    "Mozilla/5.0 (compatible; ReguLumen-CorpusFetcher/0.1; "
    "+https://github.com/example/regulumen)"
)

TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

def is_valid_body(source_type: str, content: bytes) -> tuple[bool, str]:
    """Validate downloaded bytes against source-type expectations.

    Returns ``(True, "")`` when the body passes all checks, or
    ``(False, reason)`` with a human-readable reason string on failure.
    """
    if len(content) < MIN_BYTES:
        return False, f"body too small: {len(content)} bytes (min {MIN_BYTES})"

    if source_type == "guidance_pdf":
        if not content.startswith(b"%PDF"):
            preview = content[:20]
            return False, f"PDF magic bytes missing; got {preview!r}"

    elif source_type == "eurlex_html":
        needle = content[:5000].lower()
        if b"<html" not in needle:
            return False, "no <html> tag found in first 5000 bytes"

    return True, ""


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_source(client: httpx.Client, source: dict[str, str]) -> dict[str, Any]:
    """Attempt to download *source* and save to ``corpus/raw/``.

    Returns a manifest entry dict regardless of success or failure.
    """
    url = source["url"]
    dest = CORPUS_RAW / source["filename"]
    entry: dict[str, Any] = {
        "id": source["id"],
        "title": source["title"],
        "short_title": source["short_title"],
        "source_type": source["source_type"],
        "jurisdiction": source["jurisdiction"],
        "url": url,
        "license_note": source["license_note"],
        "retrieved_at": TODAY,
        "local_filename": source["filename"],
        "http_status": None,
        "content_type": "",
        "sha256": None,
        "byte_size": None,
        "success": False,
        "error": None,
    }

    try:
        log.info("Fetching %s  ->  %s", source["id"], url)
        response = client.get(url, follow_redirects=True)
        entry["http_status"] = response.status_code
        entry["content_type"] = response.headers.get("content-type", "")

        if response.status_code != 200:
            return _fallback_to_existing(source, dest, entry, f"HTTP {response.status_code}")

        body = response.content
        valid, reason = is_valid_body(source["source_type"], body)
        if not valid:
            return _fallback_to_existing(source, dest, entry, f"Validation failed: {reason}")

        dest.write_bytes(body)
        entry["sha256"] = sha256_of_bytes(body)
        entry["byte_size"] = len(body)
        entry["success"] = True
        sha_short = entry["sha256"][:16] + "..."
        log.info("  OK  %s -- %d bytes  sha256=%s", source["id"], len(body), sha_short)

    except httpx.TimeoutException as exc:
        return _fallback_to_existing(source, dest, entry, f"Timeout: {exc}")
    except httpx.RequestError as exc:
        return _fallback_to_existing(source, dest, entry, f"Request error: {exc}")

    return entry


def _fallback_to_existing(
    source: dict[str, str],
    dest: Path,
    entry: dict[str, Any],
    failure_reason: str,
) -> dict[str, Any]:
    """Preserve an already-valid on-disk copy when re-downloading fails.

    Some sources (notably EUR-Lex, fronted by an AWS-WAF JS challenge) cannot be
    fetched by a scripted client. If a valid file is already present on disk we
    keep it and mark the entry successful (noting the download was skipped),
    rather than clobbering a good corpus file with a failure.
    """
    if dest.exists():
        body = dest.read_bytes()
        valid, _reason = is_valid_body(source["source_type"], body)
        if valid:
            entry["sha256"] = sha256_of_bytes(body)
            entry["byte_size"] = len(body)
            entry["success"] = True
            entry["error"] = f"download skipped ({failure_reason}); kept existing file"
            log.warning(
                "KEEP %s -- download failed (%s); preserving existing %d-byte file",
                source["id"],
                failure_reason,
                len(body),
            )
            return entry

    log.warning("SKIP %s -- %s", source["id"], failure_reason)
    entry["error"] = failure_reason
    return entry


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(entries: list[dict[str, Any]]) -> None:
    print()
    print("=" * 72)
    print(f"{'Source':<35} {'Status':<12} {'Bytes':>12}")
    print("-" * 72)
    for e in entries:
        status = "OK" if e["success"] else (e.get("error") or f"HTTP {e['http_status']}")
        bytes_str = str(e["byte_size"]) if e["byte_size"] is not None else "-"
        print(f"{e['short_title']:<35} {status:<12} {bytes_str:>12}")
    print("=" * 72)
    ok = sum(1 for e in entries if e["success"])
    print(f"  {ok}/{len(entries)} sources downloaded successfully.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("ReguLumen corpus fetcher starting -- saving to %s", CORPUS_RAW)

    entries: list[dict[str, Any]] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        for source in SOURCES:
            entry = fetch_source(client, source)
            entries.append(entry)

    # Write manifest
    MANIFEST_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Manifest written to %s", MANIFEST_PATH)

    print_summary(entries)

    # Exit 0 even if some sources failed (logged as warnings).
    return 0


if __name__ == "__main__":
    sys.exit(main())
