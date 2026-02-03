"""
Consolidated and optimized tests for all tables in the Jubilant PDF.
"""
import pytest
from pathlib import Path
import sys
import os
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
from pymupdf_rag import matriz_to_ascii


@pytest.fixture
def pdf_test(tmp_path):
    """Returns PDF path from PDF_PATH environment variable."""
    pdf_path_str = os.getenv("PDF_PATH")
    assert pdf_path_str, "PDF_PATH environment variable not found in .env file"
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    return pdf_path


def extract_table_llm(pdf_path: Path, strategy: str, page: int = None, table_index: int = 0):
    """Extract table using PyMuPDF4LLM."""
    chunks = llm.to_markdown(str(pdf_path), page_chunks=True, table_strategy=strategy)
    
    if page is None:
        for chunk_idx, chunk in enumerate(chunks):
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


def extract_table_pymupdf(pdf_path: Path, strategy: str, page: int = None, table_index: int = 0):
    """Extract table using PyMuPDF directly (fallback)."""
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
    """Extract table data from different possible formats."""
    if "matriz" in table:
        return table["matriz"]
    elif "data" in table:
        return table["data"]
    elif "markdown" in table:
        return table["markdown"]
    else:
        return table


def _extract_table_from_pymupdf(table):
    """Extract and format table from PyMuPDF to expected format."""
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
        
        ascii_matrix = matriz_to_ascii(formatted_matrix)
        structure = {
            "matriz_ascii": ascii_matrix,
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


def find_table_with_fallback(pdf_path: Path, page: int = None, table_index: int = 0):
    """Find table using different strategies and methods."""
    strategies = ["lines_strict", "lines", "text"]
    
    for strategy in strategies:
        complete_structure = extract_table_llm(pdf_path, strategy, page, table_index)[1]
        if complete_structure:
            return complete_structure, strategy, "pymupdf4llm"
    
    for strategy in strategies:
        complete_structure = extract_table_pymupdf(pdf_path, strategy, page, table_index)[1]
        if complete_structure:
            return complete_structure, strategy, "pymupdf_direto"
    
    return None, None, None


def compare_ascii_matrices(expected: str, obtained: str, test_name: str = "", page: int = None):
    """Compare two ASCII matrices and return detailed differences."""
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


# Test configuration: (test_id, page, table_index, expected_ascii_matrix)
TEST_CONFIGURATIONS = [
    (
        "tabela1",
        None,
        0,
        """--------------------------------------
|STAGE : ARP-3                       |
|------------------------------------|
|Input batch size  |Output batch size|
|------------------|-----------------|
|55 – 60 Kg of ARP2|43.18 to 57.6    |
--------------------------------------"""
    ),
    (
        "tabela5",
        14,
        0,
        """-----------------------------------------
|Name of Solvents |Limit                |
|-----------------|---------------------|
|Acetonitrile     |Not more than 200 ppm|
|-----------------|---------------------|
|Isopropyl alcohol|Not more than 2000   |
|                 |ppm                  |
|-----------------|---------------------|
|Cyclohexane      |Not more than 1000   |
|                 |ppm                  |
-----------------------------------------"""
    ),
    (
        "tabela11",
        21,
        0,
        """---------------------------------------------------------------------------------------------------
|Source                              |Are there   |Remarks                                        |
|                                    |any direct  |                                               |
|                                    |source of   |                                               |
|                                    |nitrosamines|                                               |
|                                    |(Like sodium|                                               |
|                                    |nitrites and|                                               |
|                                    |amines)     |                                               |
|                                    |(Yes/No)    |                                               |
|------------------------------------|------------|-----------------------------------------------|
|Solvents used in key starting       |No          |Risk of formation of nitroso impurities due to |
|materials and  drug substance       |            |solvents is eliminated.                        |
|manufacturing                       |            |                                               |
|------------------------------------|------------|-----------------------------------------------|
|Reagents used in key starting       |No          |Risk of formation of nitroso impurities due to |
|materials and drug substance        |            |reagents is eliminated.  There is a possibility|
|manufacturing                       |            |for carryover of secondary amines (DBA) & Tetra|
|                                    |            |Butyl Ammonium  Iodide (TBAI).  Since, there is|
|                                    |            |no source of nitrite is used during the        |
|                                    |            |manufacturing process of  drug substance, risk |
|                                    |            |of formation of nitrosamine impurities due to  |
|                                    |            |secondary amines  from DBA and TBAI is ruled   |
|                                    |            |out.                                           |
|------------------------------------|------------|-----------------------------------------------|
|All the possible process and        |            |Risk of formation of nitroso impurities due to |
|degradation                         |            |the possible process and                       |
|------------------------------------|------------|-----------------------------------------------|
|impurities in key starting materials|No          |degradation impurities in key starting         |
|and drug substance                  |            |materials and drug substance is eliminated.    |
|------------------------------------|------------|-----------------------------------------------|
|Recovered solvents used             |No          |Risk of formation of nitroso impurities due to |
|                                    |            |use of recovered solvents is eliminated  as    |
|                                    |            |recovered solvents are not used in the         |
|                                    |            |manufacturing process of  Aripiprazole.        |
|------------------------------------|------------|-----------------------------------------------|
|Is there a risk of nitrosamines     |No          |Risk of formation of nitroso impurities due    |
|forming in the API synthetic process|            |combination of reagents, solvents, catalysts   |
|taking  into consideration the      |            |and  starting materials used, intermediates    |
|combination of reagents, solvents,  |            |formed, impurities and degradants is           |
|catalysts  and starting materials   |            |eliminated.                                    |
|used, intermediates formed,         |            |                                               |
|impurities and  degradants          |            |                                               |
---------------------------------------------------------------------------------------------------"""
    ),
    (
        "tabela13",
        23,
        0,
        """---------------------------------------------------------------------------
|Impurity           |Specifications|Batch No.                             |
|                   |              |--------------------------------------|
|                   |              |3ARP321002  |3ARP321003  |3ARP321004  |
|-------------------|--------------|------------|------------|------------|
|Aripiprazole       |Not more      |0.03%       |0.04%       |0.02%       |
|Related compound-G |than 0.10%    |            |            |            |
|-------------------|--------------|------------|------------|------------|
|Aripiprazole       |Not more      |Not detected|Not detected|Not detected|
|Related compound-F |than 0.10%    |            |            |            |
|-------------------|--------------|------------|------------|------------|
|Aripiprazole 4,    |Not more      |0.02%       |0.02%       |0.02%       |
|4’-dimer           |than 0.10%    |            |            |            |
|-------------------|--------------|------------|------------|------------|
|Any other          |Not more      |0.06%       |0.05%       |0.05%       |
|individual impurity|than 0.10%    |            |            |            |
|-------------------|--------------|------------|------------|------------|
|Total impurities   |Not more      |0.013%      |0.13%       |0.11%       |
|                   |than 0.50%    |            |            |            |
---------------------------------------------------------------------------"""
    ),
    (
        "tabela14",
        23,
        1,
        """------------------------------------------------------------------------------
|Solvents          |Specifications|Batch No.                                 |
|                  |              |------------------------------------------|
|                  |              |3ARP321002|3ARP321003     |3ARP321004     |
|------------------|--------------|----------|---------------|---------------|
|Acetonitrile      |Not more than |Below     |Below Detection|Below          |
|[LOD:15 ppm; LOQ: |200 ppm       |Detection |Limit          |quantitation   |
|52 ppm]           |              |Limit     |               |limit          |
|------------------|--------------|----------|---------------|---------------|
|Isopropyl alcohol |Not more than |116 ppm   |141 ppm        |140 ppm        |
|[LOD:30 ppm; LOQ: |2000 ppm      |          |               |               |
|75 ppm]           |              |          |               |               |
|------------------|--------------|----------|---------------|---------------|
|Cyclohexane [LOD:1|Not more than |Below     |Below Detection|Below Detection|
|ppm; LOQ: 4 ppm]  |1000 ppm      |Detection |Limit          |Limit          |
|                  |              |Limit     |               |               |
------------------------------------------------------------------------------"""
    ),
    (
        "tabela15",
        24,
        0,
        """----------------------------------------------
|S.No               |Batch. No |Residue on   |
|                   |          |Ignition     |
|-------------------|----------|-------------|
|1                  |3ARP321002|0.03%        |
|-------------------|----------|-------------|
|2                  |3ARP321003|0.03%        |
|-------------------|----------|-------------|
|3                  |3ARP321004|0.03%        |
|-------------------|----------|-------------|
|Specification Limit           |Not more than|
|                              |0.1%         |
----------------------------------------------"""
    ),
    (
        "tabela16",
        25,
        0,
        """-------------------------------------------------------------------------
|Element|Class  |Detection|Quantitation|Batch #                         |
|       |       |Limit    |Limit       |                                |
|       |       |         |            |--------------------------------|
|       |       |         |            |3ARP317001|3ARP317002|3ARP318001|
|-------|-------|---------|------------|----------|----------|----------|
|Cd     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Pb     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|As     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Hg     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Co     |2A     |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|V      |2A     |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Ni     |2A     |0.03ppm  |0.10 ppm    |BQL       |BQL       |0.17      |
-------------------------------------------------------------------------"""
    ),
    (
        "tabela17",
        25,
        1,
        """-----------------------------------------------------------------------------
|Element|Class   |Detection|Quantitation|Batch #                            |
|       |        |Limit    |Limit       |                                   |
|       |        |         |            |-----------------------------------|
|       |        |         |            |3APR3/12001|3APR3/12002|3APR3/12003|
|-------|--------|---------|------------|-----------|-----------|-----------|
|Al     |Other   |2.2 ppm  |6.5 ppm     |BDL        |BDL        |BDL        |
|       |elements|         |            |           |           |           |
-----------------------------------------------------------------------------"""
    ),
    (
        "tabela18",
        26,
        0,
        """------------------------------------------------------------------------------------------------------------
|Intended route of administration/Use of the substance: Oral                                               |
|----------------------------------------------------------------------------------------------------------|
|Element                                                    |Class  |Intentionally|Considered in|Conclusion|
|                                                           |       |added?       |risk         |          |
|                                                           |       |             |management?  |          |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Cd                                                         |1      |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Pb                                                         |1      |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|As                                                         |1      |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Hg                                                         |1      |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Co                                                         |2A     |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|V                                                          |2A     |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Ni                                                         |2A     |No           |Yes          |Absent    |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Tl                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Au                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Pd                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Ir                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Os                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Rh                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Ru                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Se                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Ag                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Pt                                                         |2B     |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Li                                                         |3      |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Sb                                                         |3      |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Ba                                                         |3      |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Mo                                                         |3      |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Cu                                                         |3      |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Sn                                                         |3      |No           |No           |*         |
|-----------------------------------------------------------|-------|-------------|-------------|----------|
|Cr                                                         |3      |No           |No           |*         |
------------------------------------------------------------------------------------------------------------"""
    ),
    (
        "tabela19",
        27,
        0,
        """------------------------------------------------------------------------------------------------------------------------
|Compound |Source   |Limit    |Class    |Batch results in the                 |Methodolo|LOD  (%) |LOQ  (%) |Remark    |
|/impurity|         |(ppm)    |         |final API                            |gy       |         |         |          |
|         |         |         |         |                                     |used     |         |         |          |
|         |         |         |         |-------------------------------------|         |         |         |          |
|         |         |         |         |3APR3/12001      |3APR3/120|3APR3/120|         |         |         |          |
|         |         |         |         |                 |02       |03       |         |         |         |          |
|---------|---------|---------|---------|-----------------|---------|---------|---------|---------|---------|----------|
|Benzene  |Isopropyl|NMT 2 ppm|ICH,     |Below            |Below    |Below    |HS-GC    |0.2      |0.7      |No        |
|         |alcohol  |         |Class- 1.|detection        |detection|detection|         |         |         |carry-over|
|         |and      |         |         |limit            |limit    |limit    |         |         |         |to  the   |
|         |Cyclohexa|         |         |                 |         |         |         |         |         |Aripiprazo|
|         |ne       |         |         |                 |         |         |         |         |         |le.       |
|         |         |         |         |                 |         |         |         |         |         |Hence     |
|         |         |         |         |                 |         |         |         |         |         |control   |
|         |         |         |         |                 |         |         |         |         |         |in  the   |
|         |         |         |         |                 |         |         |         |         |         |Aripiprazo|
|         |         |         |         |                 |         |         |         |         |         |le        |
|         |         |         |         |                 |         |         |         |         |         |is not    |
|         |         |         |         |                 |         |         |         |         |         |proposed. |
|---------|---------|---------|---------|-----------------|---------|---------|---------|---------|---------|          |
|Aluminium|7-HDQ    |NMT 50   |Non-     |Below            |Below    |Below    |ICP-MS   |2.2      |6.5      |          |
|         |         |ppm      |Genotoxic|detection        |detection|detection|         |         |         |          |
|         |         |         |         |limit            |limit    |limit    |         |         |         |          |
------------------------------------------------------------------------------------------------------------------------"""
    ),
]


@pytest.mark.parametrize(
    "test_id,page,table_index,expected_ascii_matrix",
    TEST_CONFIGURATIONS,
    ids=[config[0] for config in TEST_CONFIGURATIONS]
)
def test_ascii_matrix_comparison(pdf_test, test_id, page, table_index, expected_ascii_matrix):
    """Parametrized test that verifies if extracted ASCII matrix matches expected format."""
    complete_structure, used_strategy, used_method = find_table_with_fallback(
        pdf_test, page, table_index
    )
    
    if complete_structure is None:
        page_info = f"Page {page}" if page is not None else "First table found"
        print(f"\nTest: {test_id} ({page_info})")
        print("ERROR: No table detected in PDF.")
        pytest.fail(f"No table was detected in the PDF for {test_id}.")
    
    ascii_matrix = complete_structure.get("matriz_ascii")
    
    if ascii_matrix is None:
        page_info = f"Page {page}" if page is not None else "First table found"
        print(f"\nTest: {test_id} ({page_info})")
        print("ERROR: Extracted table does not have 'matriz_ascii' field.")
        print(f"Available keys: {list(complete_structure.keys())}")
        pytest.fail(f"The extracted table does not have the 'matriz_ascii' field for {test_id}.")
    
    differences, are_equal = compare_ascii_matrices(expected_ascii_matrix, ascii_matrix, test_id, page)
    
    if are_equal:
        print(f"✓ RESULT: PASSED")
    else:
        print(f"✗ RESULT: FAILED")
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
