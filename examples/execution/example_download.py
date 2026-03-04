from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/neuraltec/pymupdf4llm.git"
REPO_DIR = Path.home() / "pymupdf4llm_main"

# Defina aqui o caminho do PDF.
PDF_PATH = "Finerenona_Hinye.pdf"

# Saída padrão: mesmo nome do PDF com .txt
OUTPUT_TXT = "FH_teste.txt"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_repo() -> Path:
    if not REPO_DIR.exists():
        run(["git", "clone", "--branch", "main", "--depth", "1", REPO_URL, str(REPO_DIR)])
    else:
        git_dir = REPO_DIR / ".git"
        if git_dir.exists():
            run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", "main"])
            run(["git", "-C", str(REPO_DIR), "checkout", "main"])
            run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", "main"])
        else:
            raise RuntimeError(
                f"O diretório {REPO_DIR} existe, mas não é um repositório git."
            )
    return REPO_DIR


def ensure_local_import(repo_root: Path) -> None:
    local_pkg_root = repo_root / "pymupdf4llm"
    sys.path.insert(0, str(local_pkg_root))
    sys.path.insert(1, str(repo_root))
    if "pymupdf4llm" in sys.modules:
        del sys.modules["pymupdf4llm"]


def read_pdf_to_txt(pdf_path: Path, output_txt: Path) -> None:
    import pymupdf4llm as llm

    chunks = llm.to_markdown(str(pdf_path), page_chunks=True)

    output_parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        page_text = chunk.get("text_ascii") or chunk.get("text", "")
        tables = chunk.get("tables") or []

        if tables:
            for table in reversed(tables):
                markdown = table.get("markdown", "") or ""
                ascii_table = (
                    table.get("matrix_ascii", "")
                    or table.get("matriz_ascii", "")
                    or ""
                )
                if markdown and ascii_table and markdown in page_text:
                    page_text = page_text.replace(markdown, ascii_table)

        output_parts.append(f"Page {idx}\n{page_text}")

    output_txt.write_text("\n\n".join(output_parts), encoding="utf-8")


def main() -> None:
    repo_root = ensure_repo()
    ensure_local_import(repo_root)

    pdf_value = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    if not pdf_value:
        raise ValueError("Defina PDF_PATH ou passe o caminho do PDF no primeiro argumento.")

    pdf_path = Path(pdf_value).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    output_value = (
        sys.argv[2]
        if len(sys.argv) > 2
        else (OUTPUT_TXT or str(pdf_path.with_suffix(".txt")))
    )
    output_txt = Path(output_value).expanduser().resolve()

    read_pdf_to_txt(pdf_path, output_txt)
    print(f"Arquivo gerado: {output_txt}")


if __name__ == "__main__":
    main()
