import sys
import time
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
        page_text = chunk.get("text_ascii") or chunk.get("text", "")
        output_parts.append(f"Page {idx_chunk}\n{page_text}")

    output_txt.write_text("\n\n".join(output_parts), encoding="utf-8")


if __name__ == "__main__":
    ensure_local_import()

    default_pdf = Path(__file__).resolve().parent / "Finerenona_Hinye.pdf"
    pdf_file = Path(sys.argv[1]) if len(sys.argv) > 1 else default_pdf
    out_file = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf_file.with_suffix(".txt")

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    start_time = time.perf_counter()
    read_pdf_to_txt(pdf_file, out_file)
    elapsed = time.perf_counter() - start_time
    print(f"Done! Results in: {out_file} (tempo: {elapsed:.2f}s)")

