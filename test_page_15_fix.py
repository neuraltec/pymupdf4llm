#!/usr/bin/env python
"""Test page 15 extraction to verify br normalization fix."""

import sys
sys.path.insert(0, 'pymupdf4llm')

from pathlib import Path
import pymupdf4llm

pdf_path = 'examples/execution/Finerenona_Hinye.pdf'
if not Path(pdf_path).exists():
    print(f"PDF file not found: {pdf_path}")
    sys.exit(1)

print(f"Processing: {pdf_path}\n")

# Extract markdown from page 15 only
try:
    text = pymupdf4llm.to_markdown(pdf_path, pages=[14], show_progress=False)  # page 15 is index 14
    
    # Look for the second table with problematic <br> tags
    # Should have "Pyridine ring, benzene ring" not "Pyridine rin g, benzene r ing"
    if "Pyridine ring, benzene ring" in text:
        print("✓ PASS: Correctly normalized 'Pyridine ring, benzene ring' (no isolated letters)")
    elif "Pyridine rin" in text and "benzene r" in text:
        print("✗ FAIL: Still has isolated letters: 'Pyridine rin' and 'benzene r'")
        print("\nExtracted text:")
        print(text)
        sys.exit(1)
    else:
        print("? UNCLEAR: Pattern not found in output")
        print("\nExtracted text:")
        print(text)
        
    # Also check for "Carbon hydrogen stretching vibration" (merged correctly)
    if "stretching vibration" in text and "stretchin g" not in text:
        print("✓ PASS: Correctly normalized 'stretching vibration' (no split)")
    elif "stretchin g" in text:
        print("✗ FAIL: Still has split: 'stretchin g'")
        sys.exit(1)
        
    print("\n✓ All integration tests passed!")
    
except Exception as e:
    print(f"Error during extraction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
