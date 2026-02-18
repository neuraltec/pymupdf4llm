"""
Shared logic to validate supplier tables.
"""
from pathlib import Path
import os
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()

base_path = Path(__file__).parent.parent.parent.parent
pymupdf_path = base_path / "pymupdf4llm" / "pymupdf4llm"
if str(pymupdf_path) not in sys.path:
    sys.path.insert(0, str(pymupdf_path))

helpers_path = base_path / "pymupdf4llm" / "pymupdf4llm" / "helpers"
if str(helpers_path) not in sys.path:
    sys.path.insert(0, str(helpers_path))

import pymupdf4llm as llm
import fitz
from pymupdf_rag import matrix_to_ascii


def _get_pdf_path(pdf_env_var: str) -> Path:
    pdf_path_str = os.getenv(pdf_env_var)
    assert pdf_path_str, f"{pdf_env_var} environment variable not found in .env file"
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    return pdf_path


def _extract_table_llm(pdf_path: Path, strategy: str, page: int = None, table_index: int = 0):
    chunks = llm.to_markdown(str(pdf_path), page_chunks=True, table_strategy=strategy)

    if page is None:
        for chunk in chunks:
            tables = chunk.get("tables") or []
            if tables:
                return _extract_table_data(tables[table_index]), tables[table_index]
    else:
        page_idx = page - 1
        if page_idx < len(chunks):
            chunk = chunks[page_idx]
            tables = chunk.get("tables") or []
            if len(tables) > table_index:
                return _extract_table_data(tables[table_index]), tables[table_index]

    return None, None


def _extract_table_pymupdf(pdf_path: Path, strategy: str, page: int = None, table_index: int = 0):
    doc = fitz.open(str(pdf_path))

    try:
        if page is None:
            for page_num in range(len(doc)):
                page_obj = doc[page_num]
                tables = page_obj.find_tables(strategy=strategy)
                if tables.tables:
                    table = tables.tables[table_index]
                    return _extract_table_from_pymupdf(table)
        else:
            page_idx = page - 1
            if page_idx < len(doc):
                page_obj = doc[page_idx]
                tables = page_obj.find_tables(strategy=strategy)
                if len(tables.tables) > table_index:
                    table = tables.tables[table_index]
                    return _extract_table_from_pymupdf(table)
    finally:
        doc.close()

    return None, None


def _extract_table_data(table: dict):
    if "matriz" in table:
        return table["matriz"]
    if "matrix" in table:
        return table["matrix"]
    if "data" in table:
        return table["data"]
    if "markdown" in table:
        return table["markdown"]
    return table


def _extract_table_from_pymupdf(table):
    try:
        matrix = table.extract()
        formatted_matrix = []
        for row_idx, row in enumerate(matrix):
            matrix_row = []
            for col_idx, cell in enumerate(row):
                if isinstance(cell, dict):
                    matrix_row.append(cell)
                else:
                    matrix_row.append({
                        "text": str(cell) if cell is not None else "",
                        "row": row_idx,
                        "col": col_idx,
                        "rowspan": 1,
                        "colspan": 1
                    })
            formatted_matrix.append(matrix_row)

        ascii_matrix = matrix_to_ascii(formatted_matrix)
        structure = {
            "matrix_ascii": ascii_matrix,
            "matriz_ascii": ascii_matrix,
            "matrix": matrix,
            "matriz": matrix,
            "bbox": table.bbox,
            "rows": table.row_count,
            "cols": table.col_count,
            "markdown": table.to_markdown()
        }
        return formatted_matrix, structure
    except Exception as e:
        print(f"  Error extracting table: {e}")
        return None, None


def _find_table_with_fallback(pdf_path: Path, page: int = None, table_index: int = 0):
    strategies = ["lines_strict", "lines", "text"]

    for strategy in strategies:
        complete_structure = _extract_table_llm(pdf_path, strategy, page, table_index)[1]
        if complete_structure:
            return complete_structure, strategy, "pymupdf4llm"

    for strategy in strategies:
        complete_structure = _extract_table_pymupdf(pdf_path, strategy, page, table_index)[1]
        if complete_structure:
            return complete_structure, strategy, "pymupdf_direto"

    return None, None, None


def _compare_ascii_matrices(expected: str, obtained: str, test_name: str = "", page: int = None):
    normalized_expected = "\n".join(line.rstrip() for line in expected.split("\n"))
    normalized_obtained = "\n".join(line.rstrip() for line in obtained.split("\n"))

    page_info = f"Page {page}" if page is not None else "First table found"
    print(f"\nTest: {test_name} ({page_info})")

    are_equal = normalized_expected == normalized_obtained

    print("\nExpected Table:")
    print(normalized_expected)

    print("\nObtained Table:")
    print(normalized_obtained)

    differences = []
    if not are_equal:
        expected_lines = normalized_expected.split("\n")
        obtained_lines = normalized_obtained.split("\n")

        if len(obtained_lines) != len(expected_lines):
            differences.append({
                "type": "line_count",
                "expected": len(expected_lines),
                "obtained": len(obtained_lines)
            })

        max_lines = max(len(expected_lines), len(obtained_lines))
        for i in range(max_lines):
            if i < len(expected_lines) and i < len(obtained_lines):
                if expected_lines[i] != obtained_lines[i]:
                    differences.append({
                        "type": "different",
                        "line": i + 1,
                        "expected": expected_lines[i],
                        "obtained": obtained_lines[i]
                    })
            elif i < len(expected_lines):
                differences.append({
                    "type": "missing",
                    "line": i + 1,
                    "expected": expected_lines[i]
                })
            else:
                differences.append({
                    "type": "extra",
                    "line": i + 1,
                    "obtained": obtained_lines[i]
                })

    return differences, are_equal


def run_table_test(
    pdf_env_var: str,
    test_id: str,
    page: int,
    table_index: int,
    expected_ascii_matrix: str
):
    pdf_path = _get_pdf_path(pdf_env_var)
    complete_structure, used_strategy, used_method = _find_table_with_fallback(
        pdf_path, page, table_index
    )

    if complete_structure is None:
        page_info = f"Page {page}" if page is not None else "First table found"
        print(f"\nTest: {test_id} ({page_info})")
        print("ERROR: No table detected in PDF.")
        pytest.fail(f"No table was detected in the PDF for {test_id}.")

    ascii_matrix = (
        complete_structure.get("matriz_ascii")
        or complete_structure.get("matrix_ascii")
    )

    if ascii_matrix is None:
        page_info = f"Page {page}" if page is not None else "First table found"
        print(f"\nTest: {test_id} ({page_info})")
        print("ERROR: Extracted table does not have 'matriz_ascii' or 'matrix_ascii' field.")
        print(f"Available keys: {list(complete_structure.keys())}")
        pytest.fail(f"The extracted table does not have the ASCII matrix field for {test_id}.")

    differences, are_equal = _compare_ascii_matrices(expected_ascii_matrix, ascii_matrix, test_id, page)

    if are_equal:
        print("✓ RESULT: PASSED")
    else:
        print("✗ RESULT: FAILED")
        if differences:
            print(f"\nDifferences found ({len(differences)}):")
            for diff in differences:
                if diff["type"] == "line_count":
                    print(f"  Line count mismatch: expected {diff['expected']}, obtained {diff['obtained']}")
                elif diff["type"] == "different":
                    print(f"  Line {diff['line']} differs:")
                    print(f"    Expected: {diff['expected']}")
                    print(f"    Obtained: {diff['obtained']}")
                elif diff["type"] == "missing":
                    print(f"  Line {diff['line']} missing:")
                    print(f"    Expected: {diff['expected']}")
                elif diff["type"] == "extra":
                    print(f"  Line {diff['line']} extra:")
                    print(f"    Obtained: {diff['obtained']}")
        pytest.fail(
            f"The extracted ASCII matrix does not match exactly the expected format for {test_id}.\n"
            f"Total differences: {len(differences)}"
        )

