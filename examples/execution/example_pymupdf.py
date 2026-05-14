"""
Example script demonstrating PyMuPDF (fitz) basic operations.
Allows selecting a specific page to extract text and view information.
"""

import sys
from pathlib import Path

import pymupdf


def main():
    pdf_path = Path("Finerenona_Hinye.pdf")

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    # Open the PDF
    doc = pymupdf.open(str(pdf_path))

    try:
        print(f"PDF: {pdf_path.name}")
        print(f"Total pages: {doc.page_count}")
        print()

        # Get page number from user
        while True:
            try:
                page_input = input("Enter page number (1-based) or 'quit' to exit: ").strip()
                if page_input.lower() == "quit":
                    break

                page_number = int(page_input)
                if not 1 <= page_number <= doc.page_count:
                    print(f"Invalid page. Please enter a number between 1 and {doc.page_count}")
                    continue

                # Get the page
                page = doc[page_number - 1]

                # Display page information
                print()
                print("=" * 60)
                print(f"Page {page_number} Information")
                print("=" * 60)
                print()

                # Page dimensions
                rect = page.rect
                print(f"Page size: {rect.width:.1f} x {rect.height:.1f} points")
                print()

                # Extract text
                text = page.get_text()
                print("--- Text Content ---")
                print(text if text.strip() else "(No text on this page)")
                print()

                # Extract blocks (more structured)
                blocks = page.get_text("blocks")
                print(f"--- Blocks ({len(blocks)} total) ---")
                for i, block in enumerate(blocks[:5]):  # Show first 5 blocks
                    if isinstance(block, dict) and block.get("type") == 0:  # Text block
                        print(f"Block {i}: {block.get('text', '')[:100]}")
                print()

                # Images on page
                image_list = page.get_images()
                print(f"--- Images ---")
                print(f"Total images on page: {len(image_list)}")
                for i, img in enumerate(image_list[:5]):
                    print(f"  Image {i}: xref={img[0]}, size={img[2]}x{img[3]}")
                print()

                # Links on page
                links = page.get_links()
                print(f"--- Links ---")
                print(f"Total links on page: {len(links)}")
                for i, link in enumerate(links[:5]):
                    print(f"  Link {i}: {link}")
                print()

                # Tables on page
                tables = page.find_tables()
                print(f"--- Tables ({len(tables.tables)} total) ---")
                table_output_lines = []
                if tables.tables:
                    for i, table in enumerate(tables.tables):
                        table_markdown = table.to_markdown()
                        print(f"  Table {i}: bbox={table.bbox}")
                        print("  Markdown:")
                        print(table_markdown)
                        table_output_lines.append(f"--- Table {i + 1} ---")
                        table_output_lines.append(f"bbox: {table.bbox}")
                        table_output_lines.append("Markdown:")
                        table_output_lines.append(table_markdown)
                        table_output_lines.append("")
                        print()
                else:
                    print("(No tables detected on this page.)")
                    table_output_lines.append("(No tables detected on this page.)")
                    print()

                # Option to save page text
                save_choice = input("Save page text to file? (y/n): ").strip().lower()
                if save_choice == "y":
                    output_file = pdf_path.with_name(
                        f"{pdf_path.stem}_page_{page_number}_pymupdf_ComLayout.txt"
                    )
                    output_lines = [
                        text.rstrip(),
                        "",
                        "--- Tables ---",
                        "",
                        *table_output_lines,
                    ]
                    output_file.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
                    print(f"Saved to: {output_file}")
                    print()

            except ValueError:
                print("Invalid input. Please enter a valid page number.")
                continue

    finally:
        doc.close()
        print("PDF closed. Goodbye!")


if __name__ == "__main__":
    main()
