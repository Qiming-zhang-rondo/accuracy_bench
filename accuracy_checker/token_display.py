"""Display-only formatting for decoded token labels.

Token IDs and decoded text used for numerical comparison are never changed by
this module.  It only makes whitespace/control characters unambiguous in CLI
and HTML labels.
"""

from __future__ import annotations


_SPECIAL_CHARS = {
    " ": "·",
    "\t": "⇥",
    "\n": "↵",
    "\r": "␍",
    "\v": "␋",
    "\f": "␌",
}


def _is_invisible_or_control(codepoint: int) -> bool:
    return (
        codepoint < 0x20
        or 0x7F <= codepoint <= 0x9F
        or 0x200B <= codepoint <= 0x200F
        or 0x202A <= codepoint <= 0x202E
        or 0x2060 <= codepoint <= 0x206F
        or codepoint == 0xFEFF
    )


def token_display_text(raw: object, token_id: int | None = None) -> str:
    """Return a readable, reversible display label for one decoded token.

    The fallback ``#<id>`` is used only when the tokenizer produced an empty
    string.  U+0100 and other normal printable Unicode characters are kept as
    is; only actual whitespace/control/zero-width code points are annotated.
    """
    text = "" if raw is None else str(raw)
    out = []
    for char in text:
        if char in _SPECIAL_CHARS:
            out.append(_SPECIAL_CHARS[char])
            continue
        codepoint = ord(char)
        if _is_invisible_or_control(codepoint):
            out.append(f"<U+{codepoint:04X}>")
        else:
            out.append(char)
    rendered = "".join(out)
    if rendered:
        return rendered
    return f"#{token_id}" if token_id is not None else "<empty>"
