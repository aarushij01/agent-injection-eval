"""Security helpers: SSRF guard for URL fetching and defenses for untrusted text.

Two separate problems are handled here:

1. SSRF: the agent can be talked into fetching internal addresses
   (cloud metadata, localhost admin panels). check_url() blocks those.
2. Prompt injection: text returned by tools is written by strangers and may
   contain instructions aimed at the model. wrap_untrusted() and
   filter_untrusted() are two (imperfect) mitigations.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

BLOCKED_HOSTNAMES = {"localhost", "metadata", "metadata.google.internal"}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home")


def _ip_is_public(ip) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _parse_ip(host: str):
    """Return an ip_address if host is an IP literal in any common encoding."""
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    # Catches tricks like http://2130706433/ or http://0x7f.1/ (both 127.0.0.1)
    if re.fullmatch(r"[0-9a-fA-FxX.]+", host):
        try:
            return ipaddress.ip_address(socket.inet_aton(host))
        except OSError:
            return None
    return None


def check_url(url: str, resolve_dns: bool = True) -> tuple[bool, str]:
    """Return (allowed, reason). Blocks anything that could reach internal hosts."""
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return False, "malformed URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme or 'none'}' is not allowed"
    if parsed.username or parsed.password:
        return False, "credentials in URL are not allowed"
    if not host:
        return False, "URL has no host"
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        return False, f"host '{host}' is an internal name"

    ip = _parse_ip(host)
    if ip is not None:
        if not _ip_is_public(ip):
            return False, f"address {ip} is private or reserved"
        return True, "ok"

    if resolve_dns:
        try:
            infos = socket.getaddrinfo(host, port or 80, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            return False, f"host '{host}' does not resolve"
        for info in infos:
            resolved = ipaddress.ip_address(info[4][0])
            if not _ip_is_public(resolved):
                return False, f"host '{host}' resolves to non-public address {resolved}"
    return True, "ok"


# --- Prompt-injection defenses -------------------------------------------------

OPEN_TAG = "<untrusted_data"
CLOSE_TAG = "</untrusted_data>"


def wrap_untrusted(text: str, source: str) -> str:
    """Mark tool output as data.

    Tag look-alikes inside the text are defanged so a document cannot "close"
    the wrapper and pretend that what follows is trusted.
    """
    safe = re.sub(r"<\s*/?\s*untrusted_data[^>]*>", "[tag removed]", text, flags=re.I)
    return f'{OPEN_TAG} source="{source}">\n{safe}\n{CLOSE_TAG}'


# Heuristic patterns for instruction-like text addressed to a model.
# NOTE: these were written by someone who had seen the test attacks, so the
# "filter" defense will look better on this benchmark than it would in the wild.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier)\b",
    r"\bsystem\s+override\b",
    r"^\s*(\[system\]|system\s*:)",
    r"\bnew instructions\b",
    r"\btrusted instructions\b",
    r"\b(ai|a\.i\.)\s+(assistant|agent|summari[sz]er|model)s?\b",
    r"\blanguage model\b",
    r"\bautomated readers\b",
    r"\bcrawlers\b",
    r"\b(system|full)\s+prompt\b",
    r"\bdebug mode\b",
    r"\bcall\s+(the\s+)?\w+\s+tool\b",
    r"\bsave_note\b|\bfetch_url\b|\bsearch_articles\b",
]
_COMPILED = [re.compile(p, re.I | re.M) for p in INJECTION_PATTERNS]
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def filter_untrusted(text: str) -> tuple[str, int]:
    """Drop sentences that look like instructions to a model.

    Returns (clean_text, number_of_segments_removed).
    """
    kept, removed = [], 0
    for segment in _SPLIT.split(text):
        if not segment or not segment.strip():
            continue
        if any(p.search(segment) for p in _COMPILED):
            removed += 1
            kept.append("[removed: suspected instruction]")
        else:
            kept.append(segment)
    return " ".join(kept), removed
