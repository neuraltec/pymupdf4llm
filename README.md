## PyMuPDF4LLM (fork local)

Este README contém instruções básicas para importar o projeto e ler um PDF.

### Português

#### Instalação (modo desenvolvimento)

```bash
cd /caminho/para/pymupdf4llm/pymupdf4llm
pip install -e .
```

#### Uso básico: ler PDF e salvar em TXT

```python
import pymupdf4llm as llm

text = llm.to_markdown("documento.pdf")

with open("documento.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

#### Uso com tabelas estruturadas (ASCII)

```python
import pymupdf4llm as llm

chunks = llm.to_markdown("documento.pdf", page_chunks=True)

output_parts = []
for idx, chunk in enumerate(chunks, start=1):
    page_text = chunk.get("text", "")
    tables = chunk.get("tables") or []

    if tables:
        for table in reversed(tables):
            md = table.get("markdown", "")
            ascii_table = table.get("matriz_ascii", "")
            if md and ascii_table and md in page_text:
                page_text = page_text.replace(md, ascii_table)

    output_parts.append(f"Page {idx}\n{page_text}")

with open("documento.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(output_parts))
```

#### Exemplo executável

```bash
cd /caminho/para/pymupdf4llm/examples/execution
python example.py
```

#### Executar testes

```bash
cd /caminho/para/pymupdf4llm
pytest -q
```

### English

#### Install (development mode)

```bash
cd /path/to/pymupdf4llm/pymupdf4llm
pip install -e .
```

#### Basic usage: read PDF and save to TXT

```python
import pymupdf4llm as llm

text = llm.to_markdown("document.pdf")

with open("document.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

#### Usage with structured tables (ASCII)

```python
import pymupdf4llm as llm

chunks = llm.to_markdown("document.pdf", page_chunks=True)

output_parts = []
for idx, chunk in enumerate(chunks, start=1):
    page_text = chunk.get("text", "")
    tables = chunk.get("tables") or []

    if tables:
        for table in reversed(tables):
            md = table.get("markdown", "")
            ascii_table = table.get("matriz_ascii", "")
            if md and ascii_table and md in page_text:
                page_text = page_text.replace(md, ascii_table)

    output_parts.append(f"Page {idx}\n{page_text}")

with open("document.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(output_parts))
```

#### Runnable example

```bash
cd /path/to/pymupdf4llm/examples/execution
python example.py
```

#### Run tests

```bash
cd /path/to/pymupdf4llm
pytest -q
```

