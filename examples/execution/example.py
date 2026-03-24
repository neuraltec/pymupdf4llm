import sys
from pathlib import Path


def ensure_local_import():
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg_root = repo_root / "pymupdf4llm"
    sys.path.insert(0, str(local_pkg_root))
    sys.path.insert(1, str(repo_root))
    if "pymupdf4llm" in sys.modules:
        del sys.modules["pymupdf4llm"]
    import pymupdf4llm  # noqa: F401


def _format_cell(cell) -> str:
    if not isinstance(cell, dict):
        return f"  content: {cell!r}\n"
    lines = [
        f"  content: {cell.get('text', '')!r}",
        f"  row={cell.get('row')}, col={cell.get('col')}",
        f"  rowspan={cell.get('rowspan', 1)}, colspan={cell.get('colspan', 1)}",
        f"  is_merged={cell.get('is_merged', False)}",
        f"  merged_from={cell.get('merged_from')}",
        f"  bbox={cell.get('bbox')}",
    ]
    return "\n".join(lines) + "\n"


def write_page_tables_dump(chunk: dict, page_label: str, output_txt: Path) -> None:
    lines = [
        f"=== Page: {page_label} ===",
        "",
        "--- Page text (markdown/ascii) ---",
        "",
        chunk.get("text_ascii") or chunk.get("text", ""),
        "",
        "=" * 60,
        "",
    ]

    tables = chunk.get("tables") or []
    if not tables:
        lines.append("(No tables detected on this page.)")
    else:
        for idx, tab in enumerate(tables):
            lines.append(f"--- Table {idx + 1} ---")
            lines.append(f"  bbox: {tab.get('bbox')}")
            lines.append(f"  rows: {tab.get('rows')}, columns: {tab.get('columns')}")
            lines.append("")
            lines.append("  Markdown:")
            lines.append(tab.get("markdown", "(not available)"))
            lines.append("")
            lines.append("  ASCII:")
            lines.append(tab.get("matrix_ascii") or "(not available)")
            lines.append("")
            lines.append("  Cells (attributes per cell):")
            matrix = tab.get("matrix") or []
            for row_idx, row in enumerate(matrix):
                for col_idx, cell in enumerate(row):
                    lines.append(f"  [row={row_idx}, col={col_idx}]")
                    lines.append(_format_cell(cell))
            lines.append("")

    output_txt.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ensure_local_import()

    import pymupdf4llm as llm

    pdf_path = Path("Finerenona_Hinye.pdf")

    print("Processing mode:")
    print("  1) Entire document")
    print("  2) Specific page (full table dump: markdown, cells, merged, etc.)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        text = llm.to_markdown(str(pdf_path), show_progress=True)
        with open("documento_Finerenona.txt", "w", encoding="utf-8") as file:
            file.write(text)
        print("Done. Output: documento_Finerenona.txt")
    elif choice == "2":
        import pymupdf

        page_input = input("Page number (1-based): ").strip()
        page_number = int(page_input)

        doc = pymupdf.open(str(pdf_path))
        try:
            if not 1 <= page_number <= doc.page_count:
                raise ValueError(
                    f"Page must be between 1 and {doc.page_count}, got {page_number}"
                )
            pno = page_number - 1
            chunks = llm.to_markdown(
                doc,
                pages=[pno],
                page_chunks=True,
                table_strategy="lines_strict",
                show_progress=True,
            )
            if chunks:
                page_name = f"page_{page_number}"
                output_txt = pdf_path.with_name(f"{pdf_path.stem}_page_{page_number}.txt")
                write_page_tables_dump(
                    chunks[0],
                    page_name,
                    output_txt,
                )
                print(f"Done. Output: {output_txt}")
        finally:
            doc.close()
    else:
        print("Invalid choice")
