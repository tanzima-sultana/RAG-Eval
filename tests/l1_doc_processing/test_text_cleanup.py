"""
test_text_cleanup.py

Tests for l1_doc_processing.text_cleanup.clean_text, covering its three
documented rules: repeated header/footer removal, in-line whitespace
collapsing, and blank-line collapsing.
"""

from l1_doc_processing.text_cleanup import clean_text


def test_clean_text_removes_short_lines_that_repeat_at_least_three_times():
    """
    Input: text made of "CONFIDENTIAL" repeated three times (a short
    line, well under the default max_repeated_line_length), each
    followed by a page label and one line of body content, with blank
    lines separating pages.
    Output: the cleaned text.

    Checks that a line repeating at least min_repeats_to_strip times is
    treated as a running header/footer and removed, while lines that
    appear only once (the page labels and body content) are kept.
    """
    lines = [
        "CONFIDENTIAL",
        "Page 1",
        "Some real content line one.",
        "",
        "",
        "CONFIDENTIAL",
        "Page 2",
        "Some real content line two.",
        "",
        "CONFIDENTIAL",
        "Page 3",
    ]
    text = "\n".join(lines)

    cleaned = clean_text(text)

    expected = "\n".join(
        [
            "Page 1",
            "Some real content line one.",
            "",
            "Page 2",
            "Some real content line two.",
            "",
            "Page 3",
        ]
    )
    assert cleaned == expected


def test_clean_text_keeps_repeated_lines_longer_than_max_repeated_line_length():
    """
    Input: the same line repeated three times, with
    max_repeated_line_length set shorter than that line's length.
    Output: the cleaned text.

    Checks that a repeated line longer than max_repeated_line_length is
    left alone rather than treated as a header/footer, since a long
    repeated line is more likely to be genuine repeated content.
    """
    lines = ["LONGLONGLONG", "LONGLONGLONG", "LONGLONGLONG"]
    text = "\n".join(lines)

    cleaned = clean_text(text, min_repeats_to_strip=3, max_repeated_line_length=5)

    assert cleaned == text


def test_clean_text_collapses_internal_whitespace():
    """
    Input: a single line containing runs of spaces and a tab.
    Output: the cleaned text.

    Checks that runs of spaces/tabs within a line are collapsed to a
    single space, and that leading/trailing whitespace on the line is
    stripped.
    """
    text = "  This   has\tmultiple    spaces  "

    cleaned = clean_text(text)

    assert cleaned == "This has multiple spaces"


def test_clean_text_collapses_consecutive_blank_lines_by_default():
    """
    Input: two lines of content separated by three consecutive blank
    lines.
    Output: the cleaned text.

    Checks that, with the default collapse_blank_lines=True, a run of
    consecutive blank lines collapses into a single blank line.
    """
    text = "A\n\n\n\nB"

    cleaned = clean_text(text)

    assert cleaned == "A\n\nB"


def test_clean_text_keeps_all_blank_lines_when_collapse_disabled():
    """
    Input: two lines of content separated by two consecutive blank
    lines, with collapse_blank_lines set to False.
    Output: the cleaned text.

    Checks that blank-line collapsing can be turned off, leaving every
    blank line in place.
    """
    text = "A\n\n\nB"

    cleaned = clean_text(text, collapse_blank_lines=False)

    assert cleaned == "A\n\n\nB"


def test_clean_text_returns_empty_string_for_empty_input():
    """
    Input: an empty string.
    Output: the cleaned text.

    Checks that clean_text handles empty input without error, and
    returns an empty string rather than raising.
    """
    assert clean_text("") == ""
