import sys
from pathlib import Path


def ensure_local_import():
    """Allow running the example from the repo without installing the package."""
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg_root = repo_root / "pymupdf4llm"
    sys.path.insert(0, str(local_pkg_root))
    sys.path.insert(1, str(repo_root))
    if "pymupdf4llm" in sys.modules:
        del sys.modules["pymupdf4llm"]
    import pymupdf4llm  # noqa: F401


def read_pdf_to_txt(pdf_path: Path, output_txt: Path) -> None:
    """Read a PDF with PyMuPDF4LLM and write the extracted text to a .txt file."""
    import pymupdf4llm as llm

    chunks = llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        table_strategy="lines_strict",
    )

    output_parts = []
    for idx_chunk, chunk in enumerate(chunks, start=1):
        page_text = chunk.get("text", "")
        tables = chunk.get("tables") or []

        if tables:
            processed_text = page_text
            for table in reversed(tables):
                table_markdown = table.get("markdown", "")
                table_ascii = table.get("matriz_ascii", "")
                if not table_markdown or not table_ascii:
                    continue
                if table_markdown in processed_text:
                    processed_text = processed_text.replace(table_markdown, table_ascii)
                    continue
                stripped_markdown = table_markdown.strip()
                if stripped_markdown and stripped_markdown in processed_text:
                    processed_text = processed_text.replace(stripped_markdown, table_ascii)
            output_parts.append(f"Page {idx_chunk}\n{processed_text}")
        else:
            output_parts.append(f"Page {idx_chunk}\n{page_text}")

    output_txt.write_text("\n\n".join(output_parts), encoding="utf-8")


if __name__ == "__main__":
    ensure_local_import()

    pdf_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Jubilant.pdf")
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf_file.with_suffix(".txt")

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    read_pdf_to_txt(pdf_file, out_file)
    print(f"Done! Results in: {out_file}")

