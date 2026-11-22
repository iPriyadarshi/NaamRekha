"""Phase 5: turn free-text LLM output into IOB2 tags. Pure Python (no GPU, no torch).

The model is asked for a JSON array like
    [{"text": "नरेंद्र मोदी", "type": "PER"}, {"text": "दिल्ली", "type": "LOC"}]
and `parse_llm_output(text, words)` maps it back onto the sentence's word tokens.

Written rules (applied everywhere, tested in scripts/test_phase5_parser.py)
---------------------------------------------------------------------------
R1  Malformed output is an ERROR, never dropped. Malformed means: empty text,
    no JSON value, broken/truncated JSON, or a JSON value of the wrong shape.
    The sentence is then scored as all `O` and `ParseResult.malformed` is True.
    Only the first `[` or `{` outside a code fence is tried, so truncated output
    is never half-rescued.
R2  Extra text around valid JSON and markdown fences are tolerated; they are
    counted in `noisy` (format drift) but are not malformed.
R3  An item must be an object with a non-empty string "text" and a string
    "type". Anything else is an invalid item: skipped and counted.
R4  Type is upper-cased and mapped through LABEL_ALIASES (PERSON -> PER, ...).
    An unknown type skips the item and is counted in `n_unknown_label`.
R5  Alignment: a span matches a contiguous run of whole word tokens when the
    two strings are equal after Unicode NFC and with all whitespace and
    zero-width joiners removed. (Naamapadam splits punctuation and initials
    into their own tokens, e.g. "एम ." while a model writes "एम.".)
R6  Repeated mentions: EVERY occurrence of a listed span is tagged. A name that
    appears twice in a sentence is the same entity both times.
R7  Overlaps: spans are applied longest first (most tokens), ties by listing
    order. A token that already has a tag is never re-tagged, so a shorter or
    later span only gets the free occurrences. The same text listed twice is
    one span; with two different types the first listed wins (`n_conflict`).
R8  A span that matches nowhere in the sentence (translated, altered or made
    up) is skipped and counted in `n_unaligned`. A span whose every occurrence
    was already taken by R7 is counted in `n_overlap_dropped`.
R9  Tags are valid IOB2: the first token of a span is B-X, the rest I-X, so two
    adjacent spans never merge.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

TYPES = ("PER", "ORG", "LOC")

LABEL_ALIASES = {
    "PER": "PER", "PERSON": "PER",
    "ORG": "ORG", "ORGANIZATION": "ORG", "ORGANISATION": "ORG",
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC",
}

_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.S)
_INVISIBLE = re.compile(r"[\s​‌‍﻿]+")


@dataclass
class ParseResult:
    tags: list[str]
    malformed: bool = False
    reason: str = ""              # why it was malformed ("" if it was not)
    noisy: bool = False           # text outside the JSON (prose or fence)
    n_items: int = 0              # valid items the model listed
    n_invalid_items: int = 0
    n_unknown_label: int = 0
    n_unaligned: int = 0
    n_overlap_dropped: int = 0
    n_conflict: int = 0
    unaligned_texts: list[str] = field(default_factory=list)  # the spans R8 skipped, as the model wrote them
    spans: list[tuple[int, int, str]] = field(default_factory=list)  # (first, last, type) word indices


def squash(s: str) -> str:
    """Comparison form used for alignment (R5)."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFC", s))


def tags_to_entities(words: list[str], tags: list[str]) -> list[dict[str, str]]:
    """Gold IOB2 -> [{"text", "type"}]. An orphan I- starts a new entity (as seqeval does)."""
    ents: list[dict[str, str]] = []
    cur = None
    for w, t in zip(words, tags):
        if t == "O":
            cur = None
            continue
        pre, typ = t.split("-", 1)
        if pre == "B" or cur is None or cur["type"] != typ:
            cur = {"text": w, "type": typ}
            ents.append(cur)
        else:
            cur["text"] += " " + w
    return ents


def format_entities(ents: list[dict[str, str]]) -> str:
    """The exact answer format the prompt asks for (also used for few-shot answers)."""
    return json.dumps(ents, ensure_ascii=False)


def _load_json(text: str) -> tuple[object, bool]:
    """Return (value, noisy). Raise ValueError(reason) when R1 applies."""
    t = text.strip()
    if not t:
        raise ValueError("empty")
    try:
        return json.loads(t), False
    except json.JSONDecodeError:
        pass
    for m in _FENCE.finditer(t):
        try:
            return json.loads(m.group(1).strip()), True
        except json.JSONDecodeError:
            continue
    body = _FENCE.sub(lambda m: m.group(1), t)          # fence markers removed, content kept
    starts = [i for i in (body.find("["), body.find("{")) if i != -1]
    if not starts:
        raise ValueError("no_json")
    try:
        value, _ = json.JSONDecoder().raw_decode(body[min(starts):])
    except json.JSONDecodeError:
        raise ValueError("broken_json") from None
    return value, True


def _items(value: object) -> list[object]:
    """The list of entity items inside a decoded JSON value, or ValueError('bad_shape')."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k.lower() == "entities" and isinstance(v, list):
                return v
        if "text" in value and "type" in value:        # a single bare entity object
            return [value]
    raise ValueError("bad_shape")


def parse_llm_output(text: str, words: list[str]) -> ParseResult:
    """Map model output onto `words`; always returns one tag per word (R1-R9)."""
    res = ParseResult(tags=["O"] * len(words))
    try:
        value, res.noisy = _load_json(text)
        items = _items(value)
    except ValueError as e:
        res.malformed, res.reason = True, str(e)
        return res

    # R3/R4: validate items, keep (squashed text, type, original order)
    wanted: list[tuple[str, str, str]] = []
    for it in items:
        if not (isinstance(it, dict) and isinstance(it.get("text"), str) and isinstance(it.get("type"), str)):
            res.n_invalid_items += 1
            continue
        key = squash(it["text"])
        if not key:
            res.n_invalid_items += 1
            continue
        typ = LABEL_ALIASES.get(it["type"].strip().upper())
        if typ is None:
            res.n_unknown_label += 1
            continue
        res.n_items += 1
        wanted.append((key, typ, it["text"]))

    # R7: dedupe by text (first type wins)
    seen: dict[str, tuple[str, str]] = {}
    for key, typ, raw in wanted:
        if key in seen:
            if seen[key][0] != typ:
                res.n_conflict += 1
            continue
        seen[key] = (typ, raw)

    # token boundaries of the squashed sentence (R5)
    toks = [squash(w) for w in words]
    starts, ends, pos = {}, {}, 0
    for i, tk in enumerate(toks):
        if tk:
            starts.setdefault(pos, i)
            pos += len(tk)
            ends[pos] = i
    sent = "".join(toks)

    def occurrences(key: str) -> list[tuple[int, int]]:
        out, i = [], sent.find(key)
        while i != -1:
            if i in starts and (i + len(key)) in ends:
                out.append((starts[i], ends[i + len(key)]))
            i = sent.find(key, i + 1)
        return out

    def n_tokens(key: str) -> int:           # token length of the span, 0 if it matches nowhere
        occ = occurrences(key)
        return occ[0][1] - occ[0][0] + 1 if occ else 0

    ordered = sorted(enumerate(seen.items()), key=lambda p: (-n_tokens(p[1][0]), p[0]))
    for _, (key, (typ, raw)) in ordered:
        occ = occurrences(key)
        if not occ:
            res.n_unaligned += 1
            res.unaligned_texts.append(raw)
            continue
        placed = False
        for a, b in occ:                                       # R6
            if all(t == "O" for t in res.tags[a:b + 1]):       # R7
                res.tags[a] = f"B-{typ}"
                for j in range(a + 1, b + 1):
                    res.tags[j] = f"I-{typ}"
                res.spans.append((a, b, typ))
                placed = True
        if not placed:
            res.n_overlap_dropped += 1
    res.spans.sort()
    return res

