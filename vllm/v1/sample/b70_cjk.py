# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-CJK: serving-side Han mask, host-side helpers (env ``B70_CJK_MASK=1``).

Import-light (no torch): the API server uses it to decide per request whether
the mask applies, the worker uses it to build the vocabulary mask.

Per-request gate. The mask is ON for a request unless
  1. the request sets ``cjk_mask`` explicitly (``vllm_xargs`` or a top-level
     extra body field; false / 0 / "off" turn it off, true forces it on), or
  2. its prompt carries Han: for chat, the system / developer / user turns
     (earlier assistant turns are not scanned, so a Han leak in the history
     does not switch the mask off); for other endpoints, the prompt tokens.
The decision lands in ``SamplingParams.extra_args["cjk_mask"]`` as a bool.

Vocabulary mask. A token is masked when
  1. its byte-level text contains a Han code point (U+3400-4DBF, U+4E00-9FFF,
     U+F900-FAFF), or
  2. it ends in an incomplete UTF-8 sequence with a lead byte 0xE3-0xE9 or
     0xEF whose code-point range reaches Han and does not reach kana
     (U+3040-30FF, U+31F0-31FF, halfwidth U+FF65-FF9F). That keeps the lone
     0xE3 byte (it leads kana, CJK punctuation and Han Ext-A alike), every
     0xE3 0x80-0x8F pair and the lone 0xEF byte; 15 kana (e.g. U+3062, U+30C5)
     have no single token and are spelled through those byte tokens, so kana
     stays emittable. The price: Ext-A (U+3400-3FFF) and the compatibility
     ideographs (U+F900-FAFF) that have no full token stay reachable through
     the lone 0xE3 / 0xEF byte tokens; both blocks are rare.
Special and added tokens are never masked. Continuation-only byte tokens are
never masked (they also spell Cyrillic and every other multi-byte script).
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

ENV = "B70_CJK_MASK"
XARG = "cjk_mask"

HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))
KANA_RANGES = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F))
_HAN_RE = re.compile("[㐀-䶿一-鿿豈-﫿]")
_LEADS = frozenset(range(0xE3, 0xEA)) | {0xEF}  # 3-byte leads reaching Han
_PROMPT_ROLES = ("system", "developer", "user")

_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def has_han(text: str) -> bool:
    return _HAN_RE.search(text) is not None


def parse_flag(value: Any) -> bool | None:
    """Explicit per-request value -> bool; None when absent.

    Raises ValueError on a value that is neither."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
        return value.strip().lower() in _TRUE
    raise ValueError(
        f"{XARG} must be a boolean (true/false, 1/0, on/off); got {value!r}"
    )


def resolve(explicit: Any, prompt_has_han: bool) -> bool:
    """Mask on unless the prompt has Han; an explicit value wins."""
    flag = parse_flag(explicit)
    return (not prompt_has_han) if flag is None else flag


def _content_texts(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        yield content
    elif isinstance(content, Iterable) and not isinstance(content, (bytes, Mapping)):
        for part in content:
            if isinstance(part, str):
                yield part
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    yield text
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    yield text


def messages_have_han(messages: Iterable[Any], roles=_PROMPT_ROLES) -> bool:
    """Han in the text of the system / developer / user turns."""
    for msg in messages:
        if isinstance(msg, Mapping):
            role, content = msg.get("role"), msg.get("content")
        else:
            role, content = getattr(msg, "role", None), getattr(msg, "content", None)
        if role not in roles:
            continue
        if any(has_han(t) for t in _content_texts(content)):
            return True
    return False


def chat_extra_args(
    extra_args: dict[str, Any], messages: Iterable[Any], model_extra: Any
) -> dict[str, Any]:
    """Resolve the chat request's flag into ``extra_args`` (a new dict)."""
    explicit = extra_args.get(XARG)
    if explicit is None and isinstance(model_extra, Mapping):
        explicit = model_extra.get(XARG)
    if explicit is not None:
        flag = resolve(explicit, False)
    else:
        flag = not messages_have_han(messages)
    out = dict(extra_args)
    out[XARG] = flag
    return out


# ---------------------------------------------------------------- vocabulary


def _bytes_to_unicode() -> dict[str, int]:
    """GPT-2 byte-level alphabet, inverted: printable char -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_U2B = _bytes_to_unicode()
_SP_BYTE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def token_bytes(token: str) -> bytes:
    """Raw bytes of a vocabulary string (byte-level BPE or SentencePiece)."""
    if all(c in _U2B for c in token):
        return bytes(_U2B[c] for c in token)
    m = _SP_BYTE.match(token)
    if m:
        return bytes([int(m.group(1), 16)])
    return token.replace("▁", " ").encode("utf-8")


def _overlaps(lo: int, hi: int, ranges) -> bool:
    return any(lo <= r_hi and r_lo <= hi for r_lo, r_hi in ranges)


def _trailing_lead_masked(b: bytes) -> bool:
    """Incomplete trailing 3-byte sequence that can only finish as Han."""
    n = len(b)
    for k in (1, 2):  # lead at n-1 (lone) or n-2 (lead + one continuation)
        i = n - k
        if i < 0:
            break
        lead = b[i]
        if lead in _LEADS and all(0x80 <= x <= 0xBF for x in b[i + 1 :]):
            base = (lead & 0x0F) << 12
            if k == 1:
                lo, hi = base, base + 0xFFF
            else:
                lo = base | ((b[i + 1] & 0x3F) << 6)
                hi = lo + 0x3F
            return _overlaps(lo, hi, HAN_RANGES) and not _overlaps(lo, hi, KANA_RANGES)
    return False


def classify(b: bytes) -> str | None:
    """'han', 'partial' or None (allowed)."""
    try:
        text = b.decode("utf-8")
        return "han" if has_han(text) else None
    except UnicodeDecodeError:
        if has_han(b.decode("utf-8", "ignore")):
            return "han"
        return "partial" if _trailing_lead_masked(b) else None


def build_mask(tokenizer: Any, vocab_size: int):
    """Bool mask [vocab_size] (numpy) and counts, from the tokenizer vocabulary."""
    import numpy as np

    mask = np.zeros(vocab_size, dtype=bool)
    vocab = tokenizer.get_vocab()
    skip = set(getattr(tokenizer, "all_special_ids", None) or ())
    with contextlib.suppress(Exception):
        skip |= set(tokenizer.get_added_vocab().values())
    counts = {"han": 0, "partial": 0, "vocab": len(vocab), "kana_only": 0}
    for tok, idx in vocab.items():
        if idx in skip or not 0 <= idx < vocab_size:
            continue
        b = token_bytes(tok)
        kind = classify(b)
        if kind is not None:
            mask[idx] = True
            counts[kind] += 1
    counts["masked"] = int(mask.sum())
    return mask, counts


_PROMPT_MASK: Any = None


def prompt_has_han_tokens(tokenizer: Any, token_ids: Any) -> bool:
    """Fallback scan for non-chat prompts (the whole prompt, history included):
    a token whose text has Han, or a byte token that is not valid UTF-8 alone
    and holds a 0xE3-0xE9 byte (a character split across tokens may be Han)."""
    global _PROMPT_MASK
    import numpy as np

    if token_ids is None or tokenizer is None:
        return False
    if _PROMPT_MASK is None:
        vocab = tokenizer.get_vocab()
        size = max(vocab.values()) + 1 if vocab else 0
        m = np.zeros(size, dtype=bool)
        for tok, idx in vocab.items():
            b = token_bytes(tok)
            try:
                m[idx] = has_han(b.decode("utf-8"))
            except UnicodeDecodeError:
                m[idx] = classify(b) is not None or any(x in _LEADS for x in b)
        _PROMPT_MASK = m
    ids = np.asarray(token_ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < _PROMPT_MASK.shape[0])]
    return bool(_PROMPT_MASK[ids].any())
