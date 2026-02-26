import pytest

from pymupdf4llm.helpers.pymupdf_rag import matrix_to_ascii


def test_ascii_does_not_break_words_when_unnecessary():
    """A continuous word sequence should not be split in the middle if the
    column width can accommodate it.

    The input resembles the failing example from the discussion. Previously the
    ASCII table could insert a newline inside the word "vibration"; this test
    guards against regressions by checking the resulting string contains the
    full phrase on the same line.
    """
    matrix = [[{"text": "Carbon hydrogen stretching vibration", "row": 0, "col": 0}]]
    ascii_tbl = matrix_to_ascii(matrix)

    # normalize spacing to allow simple substring search
    one_line = ascii_tbl.replace("\n", " ")
    assert "stretching vibration" in one_line


def test_ascii_collapses_real_newlines():
    """Any actual newline characters should be treated as spaces so words
    are not broken inadvertently.
    """
    matrix = [[{"text": "Carbon hydrogen stretchin\ngvibration", "row": 0, "col": 0}]]
    ascii_tbl = matrix_to_ascii(matrix)
    one_line = ascii_tbl.replace("\n", " ")
    assert "stretching vibration" in one_line


def test_ascii_collapses_internal_spaces():
    """Spaces erroneously inserted inside words should be removed.

    The table generator previously produced results like "vibr ation" or
    "hydr ogen" when the underlying extraction added stray spaces.  These
    should be collapsed to the proper word.
    """
    matrix = [[{"text": "vibr ation hydr ogen nitrogen and", "row": 0, "col": 0}]]
    ascii_tbl = matrix_to_ascii(matrix)
    one_line = ascii_tbl.replace("\n", " ")
    assert "vibration hydrogen" in one_line
    # ensure legitimate two-word phrase is not collapsed
    assert "nitrogen and" in one_line


def test_ascii_width_expansion(monkeypatch):
    """When overall width is artificially constrained smaller than a word,
    the column should expand beyond the limit to avoid splitting the word.

    We pick a single cell containing a word of length 8 and force
    ``MAX_TOTAL_WIDTH`` just below that size; the resulting ASCII table must
    end up wider than the imposed limit if the word cannot be broken.
    """
    import pymupdf4llm.helpers.pymupdf_rag as rag

    # force the table to be extremely narrow (available space 7 characters)
    monkeypatch.setattr(rag, "MAX_TOTAL_WIDTH", 9)
    matrix = [[{"text": "abcdefgh", "row": 0, "col": 0}]]
    ascii_tbl = rag.matrix_to_ascii(matrix)

    # the top border line length should exceed the old MAX_TOTAL_WIDTH
    first_line = ascii_tbl.splitlines()[0]
    assert len(first_line) > rag.MAX_TOTAL_WIDTH

    # also verify the word appears intact in the output
    assert "abcdefgh" in ascii_tbl.replace("\n", " ")
