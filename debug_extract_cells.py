#!/usr/bin/env python
"""Debug script to trace table cell content through the pipeline."""

import sys
sys.path.insert(0, 'pymupdf4llm')

import pymupdf
from pymupdf4llm.helpers.pymupdf_rag import to_markdown

# Open PDF and get page 15 (index 14)
doc = pymupdf.open('examples/execution/Finerenona_Hinye.pdf')
page = doc[14]

# Find tables on the page
table_finder = page.find_tables(clip=page.rect)
tables = table_finder.tables

print(f"Found {len(tables)} tables on page 15")
print()

# Focus on the second table (index 1)
if len(tables) >= 2:
    t = tables[1]
    print("Second table structure:")
    print(f"  Rows: {t.row_count}, Cols: {t.col_count}")
    print()
    
    # Extract matrix using PyMuPDF's method
    matrix = t.extract()
    
    print("Matrix content (row 4, problematic row with 'Pyridine'):")
    if len(matrix) > 4:
        for col_idx, cell in enumerate(matrix[4]):
            print(f"  Col {col_idx}: {repr(cell)}")
    print()
    
    # Now check what extract_cells returns
    from pymupdf4llm.helpers.utils import extract_cells
    
    print("Testing extract_cells on the problematic cells:")
    textpage = page.get_textpage()
    
    # Get cell bbox from the table using cell_bbox method
    try:
        print(f"  Row 4:")
        
        # Test columns 2 and 3 (the problematic ones)
        for col_idx in [2, 3]:
            try:
                cell_bbox = t.cell_bbox(4, col_idx)
                print(f"\n  Col {col_idx}: bbox={cell_bbox}")
                if cell_bbox:
                    extracted_plain = extract_cells(textpage, cell_bbox, markdown=False)
                    extracted_md = extract_cells(textpage, cell_bbox, markdown=True)
                    print(f"    markdown=False: {repr(extracted_plain)}")
                    print(f"    markdown=True:  {repr(extracted_md)}")
            except Exception as e2:
                print(f"    Error on col {col_idx}: {e2}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

doc.close()
