"""
text_cleanup.py

Cleans extracted document text before it's stored on Document.text.
Text extraction (e.g. pulling text out of a PDF page by page) carries
over "noise" that was never part of the document's actual content:

  - Running headers/footers: the same line (a title, a confidentiality
    notice, "Page 3 of 9", etc.) printed on every page, which shows up
    in extracted text as that line repeating once per page.
  - Irregular whitespace: multiple spaces/tabs left over from how the
    source visually laid text out (columns, indentation, table cells),
    and extra blank lines from page breaks — none of which carry
    meaning once the text is flattened into a single string.

Removing this noise makes the text shorter and more consistent for
downstream chunking/embedding, without changing what the document
actually says. Rules are parameterized (not hardcoded) so callers can
tune them per document type/dataset.
"""

import re


def _count_line_frequencies(lines: list[str]) -> dict:
    """
    lines: the document's text, split into individual lines.

    Counts how many times each stripped, non-empty line appears in
    lines. A line that repeats many times across the document is the
    signature of a running header/footer (the same line printed on
    every page); genuine body content rarely repeats verbatim.

    Returns a dict mapping each distinct stripped line to how many
    times it occurs.
    """
    frequencies = {}
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "":
            continue
        if stripped_line in frequencies:
            frequencies[stripped_line] += 1
        else:
            frequencies[stripped_line] = 1
    return frequencies


def _find_header_footer_lines(
    lines: list[str],
    min_repeats_to_strip: int,
    max_repeated_line_length: int,
) -> set:
    """
    lines: the document's text, split into individual lines.
    min_repeats_to_strip: how many times a line must repeat before it's
        treated as a header/footer instead of a coincidentally-repeated
        piece of content.
    max_repeated_line_length: only lines at most this many characters
        long are eligible to be treated as a header/footer. Headers and
        footers (titles, page numbers, confidentiality notices) are
        normally short; a long line that happens to repeat is more
        likely genuine repeated content (e.g. a boilerplate clause), so
        it's left alone rather than risk deleting real content.

    Identifies which stripped lines look like a repeated header/footer,
    using _count_line_frequencies.

    Returns the set of stripped lines to remove.
    """
    frequencies = _count_line_frequencies(lines)

    header_footer_lines = set()
    for stripped_line, count in frequencies.items():
        if count < min_repeats_to_strip:
            continue
        if len(stripped_line) > max_repeated_line_length:
            continue
        header_footer_lines.add(stripped_line)

    return header_footer_lines


def _collapse_line_whitespace(line: str) -> str:
    """
    line: a single line of text.

    Collapses runs of spaces and tabs into a single space, and strips
    leading/trailing whitespace. This removes spacing that only existed
    to visually align text in the original document layout (e.g. PDF
    columns, table cells, indentation), which has no meaning once the
    text is extracted as a flat string.

    Returns the cleaned line.
    """
    collapsed = re.sub(r"[ \t]+", " ", line)
    return collapsed.strip()


def clean_text(
    text: str,
    min_repeats_to_strip: int = 3,
    max_repeated_line_length: int = 200,
    collapse_blank_lines: bool = True,
) -> str:
    """
    text: raw extracted document text (e.g. from PDFParser).
    min_repeats_to_strip: a line that repeats at least this many times
        across the document is treated as a header/footer and removed.
    max_repeated_line_length: only lines at most this many characters
        long are eligible to be removed as a header/footer.
    collapse_blank_lines: if True, runs of two or more consecutive
        blank lines are collapsed into a single blank line, so page
        breaks/paragraph spacing don't turn into large empty gaps.

    Removes noise from text, noise: repeated header/footer
    lines, excess in-line whitespace, and optionally redundant blank
    lines. Every rule is a parameter with a default, so a caller can
    tune or effectively disable a rule (e.g. min_repeats_to_strip set
    very high) per document type without editing this function.

    Returns the cleaned text.
    """
    lines = text.splitlines()

    header_footer_lines = _find_header_footer_lines(
        lines, min_repeats_to_strip, max_repeated_line_length
    )

    cleaned_lines = []
    previous_line_was_blank = False
    for line in lines:
        cleaned_line = _collapse_line_whitespace(line)

        if cleaned_line in header_footer_lines:
            continue

        if cleaned_line == "":
            if collapse_blank_lines and previous_line_was_blank:
                continue
            previous_line_was_blank = True
        else:
            previous_line_was_blank = False

        cleaned_lines.append(cleaned_line)

    cleaned_text = ""
    for index, cleaned_line in enumerate(cleaned_lines):
        if index > 0:
            cleaned_text += "\n"
        cleaned_text += cleaned_line

    return cleaned_text.strip()
