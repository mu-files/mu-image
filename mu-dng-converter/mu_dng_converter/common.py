"""Shared helpers used by the PyWebView bridge and the CLI."""

from datetime import timedelta


def parse_time_shift(offset_str: str) -> float:
    """Parse a time shift string to signed seconds.

    Format: "[+|-][D ]HH:MM[:SS]", e.g. "+1:30", "-2 04:00:00".
    Hours can be large (e.g. "178:00" = 178 hours).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    offset_str = offset_str.strip()
    if not offset_str:
        return 0.0
    sign = -1 if offset_str[0] == "-" else 1
    body = offset_str[1:] if offset_str[0] in "+-" else offset_str
    if " " in body:
        days_part, time_part = body.split(" ", 1)
        days = int(days_part)
    else:
        time_part = body
        days = 0
    h_m_s = [int(x) for x in time_part.split(":")]
    if not h_m_s or len(h_m_s) > 3:
        raise ValueError(f"Invalid time shift: {offset_str}")
    hours = h_m_s[0]
    minutes = h_m_s[1] if len(h_m_s) > 1 else 0
    seconds = h_m_s[2] if len(h_m_s) > 2 else 0
    delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    return sign * delta.total_seconds()


def parse_metadata_ops(metadata_ops, log):
    """Parse UI metadata ops into engine arguments.

    Each op is a dict with a "type" of "strip", "set", "shift-time", or
    "shift-timezone". Shift ops accept their value under "value" (web UI)
    or legacy "offset"/"timezone" keys.

    Args:
        metadata_ops: Iterable of op dicts, or None.
        log: Callback(msg: str) for parse warnings.

    Returns:
        (strip_tags, extra_tags, time_offset_seconds, time_timezone) where
        strip_tags is a set or None and extra_tags is a MetadataTags or None.
    """
    from muimg.tiff_metadata import MetadataTags

    strip_tags = set()
    extra_tags = MetadataTags()
    time_offset_seconds = 0.0
    time_timezone = None

    for op in metadata_ops or []:
        op_type = op.get("type")
        if op_type == "strip":
            strip_tags.add(op.get("name", "").strip())
        elif op_type == "set":
            extra_tags.add_tag(op.get("name", ""), op.get("value", ""))
        elif op_type == "shift-time":
            offset_str = (op.get("offset") or op.get("value") or "").strip()
            if offset_str:
                try:
                    time_offset_seconds += parse_time_shift(offset_str)
                except (ValueError, IndexError):
                    log(f"[warn] Could not parse time offset: {offset_str}")
        elif op_type == "shift-timezone":
            tz = (op.get("timezone") or op.get("value") or "").strip()
            if tz:
                time_timezone = tz

    strip_tags = {t for t in strip_tags if t} or None
    if not extra_tags._tags:
        extra_tags = None
    return strip_tags, extra_tags, time_offset_seconds, time_timezone
