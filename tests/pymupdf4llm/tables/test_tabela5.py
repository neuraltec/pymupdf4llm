import pytest
from pathlib import Path
import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add path to the second level of the package
base_path = Path(__file__).parent.parent.parent.parent
pymupdf_path = base_path / "pymupdf4llm" / "pymupdf4llm"
if str(pymupdf_path) not in sys.path:
    sys.path.insert(0, str(pymupdf_path))

import pymupdf4llm as llm
import fitz  # PyMuPDF


@pytest.fixture
def pdf_test(tmp_path):
    """
    Define the test PDF path.
    The path is read from the PDF_PATH environment variable in the .env file.
    """
    pdf_path_str = os.getenv("PDF_PATH")
    assert pdf_path_str, "PDF_PATH environment variable not found in .env file"
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    return pdf_path


def extract_first_table_llm(pdf_path: Path, strategy="lines_strict", page=14):
    """
    Extracts the table from the specified page using PyMuPDF4LLM (local fork).
    Returns a tuple (table_data, complete_structure) where:
    - table_data: list of lists (matrix) or None
    - complete_structure: dict with complete information about the found table
    """
    chunks = llm.to_markdown(str(pdf_path), page_chunks=True, table_strategy=strategy)
    
    # Debug: show all chunks and their tables
    print(f"\nSearching for tables on page {page} with strategy '{strategy}'...")
    print(f"Total chunks (pages): {len(chunks)}")
    
    # Adjust index (pages start at 1, but indices start at 0)
    page_idx = page - 1
    
    if page_idx < len(chunks):
        chunk = chunks[page_idx]
        tables = chunk.get("tables") or []
        print(f"  Page {page} (chunk {page_idx + 1}): {len(tables)} table(s) found")
        
        if tables:
            table = tables[0]
            print(f"  First table found on page {page}")
            
            # Try to extract the matrix in different ways
            data = None
            if "matriz" in table:
                data = table["matriz"]
            elif "data" in table:
                data = table["data"]
            elif "markdown" in table:
                # If only markdown is available, try to convert
                data = table["markdown"]
            else:
                # Return the complete structure for debug
                data = table
            
            return data, table
    else:
        print(f"  Page {page} does not exist in PDF (total pages: {len(chunks)})")
    
    return None, None


def extract_first_table_pymupdf(pdf_path: Path, strategy="lines_strict", page=14):
    """
    Extracts the first table from the specified page using PyMuPDF directly (fallback).
    Returns a tuple (table_data, complete_structure).
    """
    doc = fitz.open(str(pdf_path))
    
    print(f"\nTrying PyMuPDF directly on page {page} with strategy '{strategy}'...")
    
    # Adjust index (pages start at 1, but indices start at 0)
    page_idx = page - 1
    
    if page_idx < len(doc):
        page_obj = doc[page_idx]
        tables = page_obj.find_tables(strategy=strategy)
        
        if tables.tables:
            print(f"  Table found on page {page}")
            first_table = tables.tables[0]
            
            # Convert to matrix
            try:
                matrix = first_table.extract()
                structure = {
                    "bbox": first_table.bbox,
                    "rows": first_table.row_count,
                    "cols": first_table.col_count,
                    "markdown": first_table.to_markdown()
                }
                doc.close()
                return matrix, structure
            except Exception as e:
                print(f"  Error extracting table: {e}")
                doc.close()
                return None, None
    else:
        print(f"  Page {page} does not exist in PDF (total pages: {len(doc)})")
    
    doc.close()
    return None, None




def test_ascii_matrix_comparison(pdf_test):
    """
    Tests if the extracted ASCII matrix matches
    exactly the expected format.
    """
    
    # Define the exact expected result
    expected_ascii_matrix = """---------------------------------------------
|Name of Solvents   |Limit                  |
|-------------------|-----------------------|
|Acetonitrile       |Not more than 200 ppm  |
|-------------------|-----------------------|
|Isopropyl alcohol  |Not more than 2000     |
|                   |ppm                    |
|-------------------|-----------------------|
|Cyclohexane        |Not more than 1000     |
|                   |ppm                    |
---------------------------------------------"""
    
    # Try different strategies with pymupdf4llm
    strategies = ["lines_strict", "lines", "text"]
    complete_structure = None
    used_strategy = None
    used_method = None
    
    for strategy in strategies:
        chunks = llm.to_markdown(str(pdf_test), page_chunks=True, table_strategy=strategy)
        
        # Search specifically on page 14 
        page_idx = 14 - 1
        if page_idx < len(chunks):
            chunk = chunks[page_idx]
            tables = chunk.get("tables") or []
            if tables:
                complete_structure = tables[0]
                used_strategy = strategy
                used_method = "pymupdf4llm"
                break
        
        if complete_structure:
            break
    
    # If not found with pymupdf4llm, try PyMuPDF directly
    if complete_structure is None:
        for strategy in strategies:
            doc = fitz.open(str(pdf_test))
            try:
                # Search specifically on page 14 
                page_idx = 14 - 1
                if page_idx < len(doc):
                    page_obj = doc[page_idx]
                    tables = page_obj.find_tables(strategy=strategy)
                    if tables.tables:
                        first_table = tables.tables[0]
                        matrix = first_table.extract()
                        # Convert to expected format
                        # Import matriz_to_ascii function from helpers module
                        # Use relative path to project
                        base_path = Path(__file__).parent.parent.parent.parent
                        helpers_path = base_path / "pymupdf4llm" / "pymupdf4llm" / "helpers"
                        if str(helpers_path) not in sys.path:
                            sys.path.insert(0, str(helpers_path))
                        from pymupdf_rag import matriz_to_ascii
                        # Convert simple matrix to dictionary format if necessary
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
                        ascii_matrix = matriz_to_ascii(formatted_matrix)
                        complete_structure = {
                            "matriz_ascii": ascii_matrix,
                            "matriz": matrix
                        }
                        used_strategy = strategy
                        used_method = "pymupdf_direto"
                        break
            finally:
                doc.close()
            if complete_structure:
                break
    
    # Show what was found
    print("\n" + "="*80)
    print("EXACT ASCII MATRIX COMPARISON TEST")
    print("="*80)
    
    if complete_structure is None:
        print("No table was detected in the PDF.")
        pytest.fail("No table was detected in the PDF.")
    
    print(f"Table found using method: '{used_method}' with strategy: '{used_strategy}'")
    
    # Get the ASCII matrix
    ascii_matrix = complete_structure.get("matriz_ascii")
    
    if ascii_matrix is None:
        print("\nThe table does not have the 'matriz_ascii' field.")
        print(f"Available keys in structure: {list(complete_structure.keys())}")
        pytest.fail("The extracted table does not have the 'matriz_ascii' field.")
    
    # Normalize both matrices for comparison (remove trailing whitespace from lines)
    normalized_ascii_matrix = "\n".join(line.rstrip() for line in ascii_matrix.split("\n"))
    normalized_expected_ascii_matrix = "\n".join(line.rstrip() for line in expected_ascii_matrix.split("\n"))
    
    print(f"\nEXPECTED ASCII MATRIX:")
    print("-"*80)
    print(normalized_expected_ascii_matrix)
    print("-"*80)
    
    print(f"\nEXTRACTED ASCII MATRIX:")
    print("-"*80)
    print(normalized_ascii_matrix)
    print("-"*80)
    
    # Exact line-by-line comparison
    expected_lines = normalized_expected_ascii_matrix.split("\n")
    obtained_lines = normalized_ascii_matrix.split("\n")
    
    print(f"\nLINE-BY-LINE COMPARISON:")
    print("-"*80)
    errors = []
    
    # Check if the number of lines is the same
    if len(obtained_lines) != len(expected_lines):
        errors.append(
            f"Different number of lines:\n"
            f"   Expected: {len(expected_lines)} lines\n"
            f"   Obtained: {len(obtained_lines)} lines"
        )
        print(f"✗ Different number of lines: expected {len(expected_lines)}, obtained {len(obtained_lines)}")
    else:
        print(f"✓ Correct number of lines: {len(expected_lines)}")
    
    # Compare each line
    max_lines = max(len(expected_lines), len(obtained_lines))
    for i in range(max_lines):
        if i < len(expected_lines) and i < len(obtained_lines):
            expected = expected_lines[i]
            obtained = obtained_lines[i]
            if expected == obtained:
                print(f"✓ Line {i+1}: OK")
            else:
                errors.append(
                    f"Line {i+1} different:\n"
                    f"   Expected: '{expected}'\n"
                    f"   Obtained:   '{obtained}'"
                )
                print(f"✗ Line {i+1}: DIFFERENT")
                print(f"    Expected: '{expected}'")
                print(f"    Obtained:   '{obtained}'")
        elif i < len(expected_lines):
            errors.append(f"Line {i+1} missing in extracted matrix. Expected: '{expected_lines[i]}'")
            print(f"✗ Line {i+1}: MISSING (expected: '{expected_lines[i]}')")
        else:
            errors.append(f"Line {i+1} extra in extracted matrix. Obtained: '{obtained_lines[i]}'")
            print(f"✗ Line {i+1}: EXTRA (obtained: '{obtained_lines[i]}')")
    
    # Exact complete comparison
    if normalized_ascii_matrix == normalized_expected_ascii_matrix:
        print("\n" + "="*80)
        print("FINAL RESULT: ASCII matrix matches EXACTLY the expected format")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("FINAL RESULT: differences found between ASCII matrix and expected format")
        print("="*80)
        print(f"\nTotal errors: {len(errors)}")
        print("\nErrors found:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nExpected ASCII matrix:")
        print("-"*80)
        print(normalized_expected_ascii_matrix)
        print("-"*80)
        print(f"\nObtained ASCII matrix:")
        print("-"*80)
        print(normalized_ascii_matrix)
        print("-"*80)
        pytest.fail(f"The extracted ASCII matrix does not match exactly the expected format.\nTotal errors: {len(errors)}")

