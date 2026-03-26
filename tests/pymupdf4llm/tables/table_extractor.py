"""
Table Extraction and Testing Module

This module centralizes logic to:
1. Extract tables from PDFs
2. Compare against expected values
3. Run automated tests

Usage:
    from table_extractor import run_table_test, extract_and_compare_tables

    # Run tests
    run_table_test(PDF_ENV_VAR, test_id, page, table_index, expected_ascii_matrix)

    # Extract and compare new tables
    extract_and_compare_tables(pdf_path, output_file=None)
"""

from pathlib import Path
import os
import sys
import json
from typing import Tuple, Dict, List, Optional

import pytest

# Setup paths
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
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_all_tables_from_pdf(
    pdf_path: Path,
    strategy: str = "lines_strict"
) -> List[Dict]:
    """
    Extract all tables from a PDF.

    Args:
        pdf_path: PDF file path
        strategy: Extraction strategy ('lines_strict', 'lines', 'text')

    Returns:
        List of dictionaries with table data
    """
    tables_found = []

    try:
        doc = fitz.open(str(pdf_path))
        try:
            total_pages = len(doc)
        finally:
            doc.close()

        for page_idx in range(total_pages):
            text = llm.to_markdown(
                str(pdf_path),
                pages=[page_idx],
                show_progress=False,
            )
            blocks = _extract_ascii_table_blocks(text)

            for table_idx, ascii_matrix in enumerate(blocks):
                table_structure = {
                    "matrix_ascii": ascii_matrix,
                    "source": "to_markdown_text",
                }
                tables_found.append({
                    "page": page_idx + 1,
                    "page_0based": page_idx,
                    "table_index": table_idx,
                    "ascii_matrix": ascii_matrix,
                    "full_structure": table_structure,
                })

    except Exception as e:
        print(f"Error extracting tables: {e}")

    return tables_found


def _extract_ascii_table_blocks(text: str) -> List[str]:
    """Extract ASCII table blocks from `to_markdown` output."""
    if not text:
        return []

    blocks = []
    current_block = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        is_border = bool(stripped) and set(stripped) == {"-"}
        is_table_line = stripped.startswith("|") or is_border

        if is_table_line:
            current_block.append(stripped)
        else:
            if current_block:
                block = "\n".join(current_block)
                if "|" in block:
                    blocks.append(block)
                current_block = []

    if current_block:
        block = "\n".join(current_block)
        if "|" in block:
            blocks.append(block)

    return blocks


def _get_pdf_path(pdf_env_var: str) -> Path:
    """Get PDF path from an environment variable."""
    pdf_path_str = os.getenv(pdf_env_var)
    assert pdf_path_str, f"Environment variable {pdf_env_var} not found in .env"
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists(), f"PDF not found at {pdf_path}"
    return pdf_path


def _extract_table_llm(
    pdf_path: Path,
    strategy: str,
    page: int = None,
    table_index: int = 0
) -> Tuple:
    """Extract table directly from `to_markdown` text."""
    if page is None:
        text = llm.to_markdown(str(pdf_path), show_progress=False)
    else:
        text = llm.to_markdown(str(pdf_path), pages=[page - 1], show_progress=False)

    blocks = _extract_ascii_table_blocks(text)
    if len(blocks) > table_index:
        ascii_matrix = blocks[table_index]
        structure = {
            "matrix_ascii": ascii_matrix,
            "source": "to_markdown_text",
        }
        return ascii_matrix, structure

    return None, None


def _extract_table_pymupdf(
    pdf_path: Path,
    strategy: str,
    page: int = None,
    table_index: int = 0
) -> Tuple:
    """Extract table directly with pymupdf."""
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
    """Extract table payload from dict."""
    if "matrix" in table:
        return table["matrix"]
    if "data" in table:
        return table["data"]
    if "markdown" in table:
        return table["markdown"]
    return table


def _extract_table_from_pymupdf(table):
    """Extract table from pymupdf `Table` object."""
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
            "matrix": matrix,
            "bbox": table.bbox,
            "rows": table.row_count,
            "cols": table.col_count,
            "markdown": table.to_markdown()
        }
        return formatted_matrix, structure
    except Exception as e:
        print(f"Error extracting table: {e}")
        return None, None


def _find_table_with_fallback(
    pdf_path: Path,
    page: int = None,
    table_index: int = 0
) -> Tuple:
    """
    Find table with strategy fallback.
    Tries: llm -> pymupdf.
    """
    complete_structure = _extract_table_llm(pdf_path, "to_markdown", page, table_index)[1]
    if complete_structure:
        return complete_structure, "to_markdown", "pymupdf4llm_text"

    return None, None, None


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def _compare_ascii_matrices(
    expected: str,
    obtained: str,
    test_name: str = "",
    page: int = None
) -> Tuple[List[Dict], bool]:
    """
    Compare two ASCII matrices and return differences.

    Returns:
        (differences, are_equal)
    """
    normalized_expected = "\n".join(line.rstrip() for line in expected.split("\n"))
    normalized_obtained = "\n".join(line.rstrip() for line in obtained.split("\n"))

    page_info = f"Page {page}" if page is not None else "First detected table"
    print(f"\nTest: {test_name} ({page_info})")

    are_equal = normalized_expected == normalized_obtained

    print("\nExpected table:")
    print(normalized_expected)

    print("\nObtained table:")
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


# ============================================================================
# MAIN TEST FUNCTION (importable)
# ============================================================================

def run_table_test(
    pdf_env_var: str,
    test_id: str,
    page: int,
    table_index: int,
    expected_ascii_matrix: str
):
    """
    Main function to run table tests.

    Args:
        pdf_env_var: Environment variable with PDF path
        test_id: Unique test id
        page: Table page (1-based)
        table_index: Table index on page
        expected_ascii_matrix: Expected ASCII matrix

    Raises:
        AssertionError: If PDF is not found
        pytest.fail: If comparison fails
    """
    pdf_path = _get_pdf_path(pdf_env_var)
    complete_structure, _used_strategy, _used_method = _find_table_with_fallback(
        pdf_path, page, table_index
    )

    if complete_structure is None:
        page_info = f"Page {page}" if page is not None else "First detected table"
        print(f"\nTest: {test_id} ({page_info})")
        print("ERROR: No table detected in PDF.")
        pytest.fail(f"No table was detected in PDF for {test_id}.")

    ascii_matrix = (
        complete_structure.get("matrix_ascii")
    )

    if ascii_matrix is None:
        page_info = f"Page {page}" if page is not None else "First detected table"
        print(f"\nTest: {test_id} ({page_info})")
        print("ERROR: Extracted table has no 'matrix_ascii' field.")
        print(f"Available keys: {list(complete_structure.keys())}")
        pytest.fail(f"Extracted table has no ASCII field for {test_id}.")

    differences, are_equal = _compare_ascii_matrices(expected_ascii_matrix, ascii_matrix, test_id, page)

    if are_equal:
        print("✓ RESULT: PASSED")
    else:
        print("✗ RESULT: FAILED")
        if differences:
            print(f"\nDifferences found ({len(differences)}):")
            for diff in differences:
                if diff["type"] == "line_count":
                    print(f"  Line count: expected {diff['expected']}, obtained {diff['obtained']}")
                elif diff["type"] == "different":
                    print(f"  Line {diff['line']} differs:")
                    print(f"    Expected: {diff['expected']}")
                    print(f"    Obtained: {diff['obtained']}")
                elif diff["type"] == "missing":
                    print(f"  Missing line {diff['line']}:")
                    print(f"    Expected: {diff['expected']}")
                elif diff["type"] == "extra":
                    print(f"  Extra line {diff['line']}:")
                    print(f"    Obtained: {diff['obtained']}")
        pytest.fail(
            f"Extracted ASCII matrix does not match expected format for {test_id}.\n"
            f"Total differences: {len(differences)}"
        )


# ============================================================================
# EXTRACTION AND COMPARISON FOR NEW TABLES
# ============================================================================

def extract_and_compare_tables(
    pdf_path: Path,
    supplier_id: str = "supplier",
    output_file: Optional[str] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Extract tables from a PDF and save as JSON.

    Args:
        pdf_path: PDF path
        supplier_id: Supplier id
        output_file: JSON output path (None = auto-generate)
        verbose: Print extraction output

    Returns:
        List of extracted tables
    """
    tables_data = extract_all_tables_from_pdf(pdf_path)

    if not tables_data and verbose:
        print(f"⚠️  No table found for {supplier_id}")
        return tables_data

    if verbose:
        print(f"✅ Found {len(tables_data)} table(s)")
        for table_data in tables_data:
            page = table_data["page"]
            table_idx = table_data["table_index"]
            matrix = table_data["ascii_matrix"]
            lines = matrix.split('\n')
            print(f"   - Page {page}, Table {table_idx}: {len(lines)} lines")

    if output_file is None:
        current_dir = Path(__file__).parent
        output_file = current_dir / f"{supplier_id}_tables_extracted.json"
    else:
        output_file = Path(output_file)

    json_data = []
    for table_data in tables_data:
        json_data.append({
            "page": table_data["page"],
            "table_index": table_data["table_index"],
            "ascii_matrix": table_data["ascii_matrix"]
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✅ Saved to: {output_file}")

    return tables_data


def print_extracted_tables(
    pdf_env_var: str,
    supplier_id: str = "supplier"
) -> None:
    """
    Print extracted tables in copy/paste-ready format.

    Args:
        pdf_env_var: Environment variable with PDF path
        supplier_id: Supplier id (used in stats naming)
    """
    pdf_path = _get_pdf_path(pdf_env_var)
    tables_data = extract_and_compare_tables(pdf_path, supplier_id)

    print(f"\n{'='*100}")
    print(f"TEST_CONFIGURATIONS = [")
    print(f"{'='*100}\n")

    for idx, table_data in enumerate(tables_data):
        page = table_data["page"]
        table_index = table_data["table_index"]
        ascii_matrix = table_data["ascii_matrix"]
        test_id = f"table_{idx + 1}_page{page}"

        escaped_matrix = ascii_matrix.replace('"""', r'\"\"\"')

        print(f'    (')
        print(f'        "{test_id}",')
        print(f'        {page},')
        print(f'        {table_index},')
        print(f'        """{escaped_matrix}"""')
        print(f'    ),')

    print(f"\n{'='*100}")
    print(f"]")
    print(f"{'='*100}\n")


# ============================================================================
# CLI USAGE
# ============================================================================

if __name__ == "__main__":
    """
    CLI usage for table extraction:

    python table_extractor.py <supplier_id> <pdf_env_var>

    Examples:
        python table_extractor.py jubilant JUBILANT_PDF_PATH
        python table_extractor.py finerenona_hinye FINERENONA_HINYE_PDF_PATH
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python table_extractor.py <supplier_id> <pdf_env_var>")
        print("\nExample:")
        print("  python table_extractor.py jubilant JUBILANT_PDF_PATH")
        sys.exit(1)

    supplier_id = sys.argv[1]
    pdf_env_var = sys.argv[2]

    print(f"\n🔍 Extracting tables for: {supplier_id.upper()}")
    print(f"{'='*80}\n")

    pdf_path = _get_pdf_path(pdf_env_var)
    print_extracted_tables(pdf_env_var, supplier_id)
