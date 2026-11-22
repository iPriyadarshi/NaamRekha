"""Tests for scripts/phase5_parser.py. No GPU, no model, no network.

Run from the repo root:  uv run python scripts/test_phase5_parser.py
(also collected by pytest if it is installed).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase5_parser import (  # noqa: E402
    format_entities,
    parse_llm_output,
    tags_to_entities,
)

W = ["राम", "दिल्ली", "में", "गूगल", "के", "साथ", "काम", "करता", "है"]


def j(*pairs):
    return json.dumps([{"text": t, "type": y} for t, y in pairs], ensure_ascii=False)


def test_clean_list():
    r = parse_llm_output(j(("राम", "PER"), ("दिल्ली", "LOC"), ("गूगल", "ORG")), W)
    assert r.tags == ["B-PER", "B-LOC", "O", "B-ORG", "O", "O", "O", "O", "O"]
    assert not r.malformed and not r.noisy and r.n_items == 3


def test_entities_wrapper_object():
    r = parse_llm_output('{"entities": [{"text": "राम", "type": "PER"}]}', W)
    assert r.tags[0] == "B-PER" and not r.malformed


def test_single_bare_entity_object():
    r = parse_llm_output('{"text": "राम", "type": "PER"}', W)
    assert r.tags[0] == "B-PER" and not r.malformed


def test_markdown_fence():
    r = parse_llm_output("```json\n" + j(("राम", "PER")) + "\n```", W)
    assert r.tags[0] == "B-PER" and not r.malformed and r.noisy


def test_fence_without_language():
    r = parse_llm_output("```\n" + j(("राम", "PER")) + "\n```", W)
    assert r.tags[0] == "B-PER" and not r.malformed


def test_prose_around_json():
    r = parse_llm_output("Sure! Here you go:\n" + j(("राम", "PER")) + "\nHope that helps.", W)
    assert r.tags[0] == "B-PER" and not r.malformed and r.noisy


def test_empty_list_is_a_valid_answer():
    r = parse_llm_output("[]", W)
    assert r.tags == ["O"] * 9 and not r.malformed and r.n_items == 0


def test_empty_text_is_malformed():
    r = parse_llm_output("   ", W)
    assert r.malformed and r.reason == "empty" and r.tags == ["O"] * 9


def test_prose_only_is_malformed():
    r = parse_llm_output("I am unable to find any entities in this sentence.", W)
    assert r.malformed and r.reason == "no_json" and r.tags == ["O"] * 9


def test_truncated_json_is_malformed_not_rescued():
    # The first entity is complete, but R1 says broken JSON is an error: no partial credit.
    r = parse_llm_output('[{"text": "राम", "type": "PER"}, {"text": "दिल्', W)
    assert r.malformed and r.reason == "broken_json" and r.tags == ["O"] * 9


def test_wrong_json_shape_is_malformed():
    for bad in ('"राम"', "42", '{"result": "ok"}', "true", "null"):
        r = parse_llm_output(bad, W)
        assert r.malformed and r.reason == "bad_shape", bad


def test_unknown_label_skipped_and_counted():
    r = parse_llm_output(j(("राम", "PER"), ("दिल्ली", "CITY"), ("गूगल", "MISC")), W)
    assert r.tags[0] == "B-PER" and r.tags[1] == "O" and r.tags[3] == "O"
    assert r.n_unknown_label == 2 and not r.malformed


def test_label_aliases_and_case():
    r = parse_llm_output(j(("राम", "person"), ("दिल्ली", "Location"), ("गूगल", "organisation")), W)
    assert r.tags[:4] == ["B-PER", "B-LOC", "O", "B-ORG"]


def test_invalid_items_counted():
    r = parse_llm_output('[{"text": "राम"}, "दिल्ली", {"text": 5, "type": "LOC"}, {"text": "  ", "type": "PER"}, '
                         '{"text": "गूगल", "type": "ORG"}]', W)
    assert r.n_invalid_items == 4 and r.tags[3] == "B-ORG" and not r.malformed


def test_span_not_in_sentence_is_unaligned():
    r = parse_llm_output(j(("मुंबई", "LOC"), ("राम", "PER")), W)
    assert r.n_unaligned == 1 and r.tags[0] == "B-PER" and r.tags[1] == "O"
    assert r.unaligned_texts == ["मुंबई"]


def test_translated_span_is_unaligned():
    r = parse_llm_output(j(("Delhi", "LOC")), W)
    assert r.n_unaligned == 1 and r.tags == ["O"] * 9


def test_partial_word_does_not_match():
    words = ["दिल्लीवासी", "खुश", "हैं"]
    r = parse_llm_output(j(("दिल्ली", "LOC")), words)
    assert r.n_unaligned == 1 and r.tags == ["O"] * 3


def test_multiword_span():
    words = ["अटल", "बिहारी", "वाजपेयी", "आए"]
    r = parse_llm_output(j(("अटल बिहारी वाजपेयी", "PER")), words)
    assert r.tags == ["B-PER", "I-PER", "I-PER", "O"]


def test_punctuation_spacing_is_forgiven():
    # Naamapadam splits "एम." into "एम" and "."; the model writes it joined (or spaced).
    words = ["एम", ".", "वेंकैया", "नायडू", "ने", "कहा"]
    for text in ("एम. वेंकैया नायडू", "एम . वेंकैया नायडू", "एम.वेंकैया नायडू"):
        r = parse_llm_output(j((text, "PER")), words)
        assert r.tags == ["B-PER", "I-PER", "I-PER", "I-PER", "O", "O"], text


def test_unicode_normalisation_nukta():
    composed = "ज़"            # ज + nukta
    words = ["ज़ाकिर", "आए"]
    r = parse_llm_output(j(("ज़" + "ाकिर", "PER")), words)   # precomposed ज़ (U+095B) + ाकिर
    assert composed and r.tags == ["B-PER", "O"]


def test_zero_width_joiner_ignored():
    words = ["क्रिकेट", "खेलो"]
    r = parse_llm_output(j(("क्‍रिकेट", "ORG")), words)
    assert r.tags == ["B-ORG", "O"]


def test_repeated_mention_all_occurrences_tagged():
    words = ["राम", "ने", "श्याम", "से", "कहा", "कि", "राम", "आएगा"]
    r = parse_llm_output(j(("राम", "PER")), words)
    assert r.tags == ["B-PER", "O", "O", "O", "O", "O", "B-PER", "O"]
    assert r.n_items == 1


def test_same_span_listed_twice_counts_once():
    words = ["राम", "ने", "कहा", "राम"]
    r = parse_llm_output(j(("राम", "PER"), ("राम", "PER")), words)
    assert r.tags == ["B-PER", "O", "O", "B-PER"] and r.n_conflict == 0


def test_same_span_two_types_first_wins():
    r = parse_llm_output(j(("दिल्ली", "LOC"), ("दिल्ली", "ORG")), W)
    assert r.tags[1] == "B-LOC" and r.n_conflict == 1


def test_nested_longest_first_and_shorter_gets_free_occurrences():
    words = ["नई", "दिल्ली", "और", "दिल्ली", "में", "मेला"]
    # Listed shortest first on purpose: the longer span must still win its tokens.
    r = parse_llm_output(j(("दिल्ली", "LOC"), ("नई दिल्ली", "LOC")), words)
    assert r.tags == ["B-LOC", "I-LOC", "O", "B-LOC", "O", "O"]


def test_overlap_with_equal_length_uses_listing_order():
    words = ["अ", "ब", "स", "कहा"]
    r = parse_llm_output(j(("अ ब", "PER"), ("ब स", "ORG")), words)
    assert r.tags == ["B-PER", "I-PER", "O", "O"]
    assert r.n_overlap_dropped == 1


def test_adjacent_same_type_spans_do_not_merge():
    words = ["राम", "श्याम", "आए"]
    r = parse_llm_output(j(("राम", "PER"), ("श्याम", "PER")), words)
    assert r.tags == ["B-PER", "B-PER", "O"]


def test_length_always_matches_even_for_garbage():
    garbage = ["", "[", "{", "[[[", "]]]", "null", '{"entities": 3}', "```", "```json", '[{"text":',
               "\u0000", '[{"text": "राम", "type": "PER"}]]]', "{'text': 'राम', 'type': 'PER'}",
               '{"entities": [{"text": "राम", "type": "PER"}]} trailing {"x": 1}']
    for g in garbage:
        for words in (W, ["राम"], []):
            assert len(parse_llm_output(g, words).tags) == len(words), g


def test_tags_are_valid_iob2_for_any_output():
    outs = [j(("राम", "PER"), ("दिल्ली", "LOC"), ("राम दिल्ली", "ORG")),
            j(("में गूगल", "ORG"), ("गूगल के", "LOC"), ("के", "PER")),
            j(("करता है", "ORG"), ("है", "LOC"), ("काम करता है", "PER"))]
    for o in outs:
        tags = parse_llm_output(o, W).tags
        prev = "O"
        for t in tags:
            if t.startswith("I-"):
                assert prev != "O" and prev[2:] == t[2:], tags
            prev = t


def test_gold_round_trip():
    words = ["श्री", "नरेंद्र", "मोदी", "ने", "दिल्ली", "में", "भाजपा", "की", "बैठक", "ली"]
    gold = ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O", "O", "O"]
    ents = tags_to_entities(words, gold)
    assert ents == [{"text": "नरेंद्र मोदी", "type": "PER"}, {"text": "दिल्ली", "type": "LOC"},
                    {"text": "भाजपा", "type": "ORG"}]
    assert parse_llm_output(format_entities(ents), words).tags == gold


def test_orphan_i_tag_starts_an_entity():
    ents = tags_to_entities(["क", "ख"], ["I-PER", "I-PER"])
    assert ents == [{"text": "क ख", "type": "PER"}]


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS ", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("  FAIL ", name, "->", repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} parser tests passed")
    return failed


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(1 if _run_all() else 0)
