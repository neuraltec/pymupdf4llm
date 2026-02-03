import sys
from pathlib import Path
import json
import textwrap

# Add pymupdf4llm path
pymupdf4llm_base = "/home/bruna/pymupdf4llm/pymupdf4llm"
sys.path.insert(0, pymupdf4llm_base)
try:
    import pymupdf4llm as llm
except ImportError as e:
    print(f"Error importing pymupdf4llm: {e}")
    sys.exit(1)

def generate_ascii_matrix(matrix):
    """
    Generates an ASCII matrix with width control but prioritizes keeping words intact.
    """
    if not matrix or not matrix[0]:
        return ""
    
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    # --- 1. Calculate Natural Widths & Minimum Word Constraints ---
    natural_widths = [0] * num_cols
    min_word_widths = [0] * num_cols  # Track the longest single word per column
    
    for row in matrix:
        for j, cell in enumerate(row):
            if isinstance(cell, dict):
                text = cell.get("text", "").strip()
                colspan = cell.get("colspan", 1)
                
                # Analyze text content
                if not text:
                    max_line = 0
                    max_word = 0
                else:
                    lines = text.split('\n')
                    max_line = max(len(l) for l in lines) if lines else 0
                    words = text.split()
                    max_word = max(len(w) for w in words) if words else 0
                
                # Distribute width requirements among spanned columns
                # We assume the longest word needs to fit in the combined space
                needed_per_col_natural = max(3, max_line // colspan)
                needed_per_col_word = max(3, max_word // colspan)
                
                for c in range(j, min(j + colspan, num_cols)):
                    natural_widths[c] = max(natural_widths[c], needed_per_col_natural)
                    min_word_widths[c] = max(min_word_widths[c], needed_per_col_word)

    # --- 2. Calculate Final Column Widths (Compression Logic) ---
    MAX_TOTAL_WIDTH = 115 
    MIN_COL_WIDTH = 5      
    
    border_overhead = num_cols + 1
    available_space = MAX_TOTAL_WIDTH - border_overhead
    
    total_natural = sum(natural_widths)
    col_widths = [0] * num_cols
    
    if total_natural <= available_space:
        # Fits perfectly without compression
        col_widths = natural_widths
    else:
        # Needs compression, but we must respect word integrity
        if total_natural == 0: total_natural = 1
        
        for i, w in enumerate(natural_widths):
            # Calculate proportional reduction
            ratio = w / total_natural
            target_w = int(available_space * ratio)
            
            # Constraint 1: Must be at least MIN_COL_WIDTH
            # Constraint 2: Must be at least as wide as the longest word (min_word_widths)
            # This ensures we don't chop "Isopropyl" into "Isopr-" even if space is tight.
            # We prioritize readability over strict 115 char limit here.
            final_w = max(MIN_COL_WIDTH, target_w, min_word_widths[i])
            col_widths[i] = final_w
            
    # --- 3. Pre-process Text Wrapping ---
    for r in range(num_rows):
        for c in range(num_cols):
            cell = matrix[r][c]
            if isinstance(cell, dict) and not cell.get("is_merged", False):
                text = cell.get("text", "").strip()
                colspan = cell.get("colspan", 1)
                
                # Calculate total width available for this cell
                eff_width = sum(col_widths[c : c + colspan]) + (colspan - 1)
                
                if text:
                    wrapped_lines = []
                    for para in text.split('\n'):
                        if not para.strip():
                            wrapped_lines.append("")
                            continue
                        # FIX: break_long_words=False prevents splitting words in the middle
                        # The column width calculation above guarantees the word fits.
                        wrapped_lines.extend(textwrap.wrap(para, width=eff_width, break_long_words=False))
                    
                    cell["text"] = "\n".join(wrapped_lines)

    # --- 4. Map Logical Cells (Master Map) ---
    master_map = {}
    for r in range(num_rows):
        for c in range(num_cols):
            cell = matrix[r][c]
            if cell.get("is_merged"):
                master_map[(r, c)] = (cell.get("primary_row"), cell.get("primary_col"))
            else:
                master_map[(r, c)] = (r, c)

    lines = []

    # --- 5. Dynamic Separator Logic ---
    def get_separator(row_idx):
        if row_idx == 0 or row_idx == num_rows:
            total_w = sum(col_widths) + num_cols + 1
            return "-" * total_w
        
        sep = "|"
        for c in range(num_cols):
            is_vert_merged = master_map.get((row_idx - 1, c)) == master_map.get((row_idx, c))
            char = " " if is_vert_merged else "-"
            sep += char * col_widths[c]
            sep += "|"
        return sep

    # --- 6. Render Rows with Multi-line Support ---
    for r in range(num_rows):
        lines.append(get_separator(r))
        
        max_h = 1
        c = 0
        while c < num_cols:
            orig_r, orig_c = master_map[(r, c)]
            master_cell = matrix[orig_r][orig_c]
            colspan = master_cell.get("colspan", 1)
            
            if r == orig_r:
                cell_lines = master_cell.get("text", "").split('\n')
                max_h = max(max_h, len(cell_lines))
            
            c += colspan

        for h_idx in range(max_h):
            row_line = "|"
            col_ptr = 0
            while col_ptr < num_cols:
                orig_r, orig_c = master_map[(r, col_ptr)]
                master_cell = matrix[orig_r][orig_c]
                colspan = master_cell.get("colspan", 1)
                
                cell_w = sum(col_widths[col_ptr : col_ptr + colspan]) + (colspan - 1)
                
                current_text = ""
                if r == orig_r:
                    cell_lines = master_cell.get("text", "").split('\n')
                    if h_idx < len(cell_lines):
                        current_text = cell_lines[h_idx]
                
                # Padding
                row_line += current_text.ljust(cell_w)[:cell_w] + "|"
                col_ptr += colspan
            lines.append(row_line)

    lines.append(get_separator(num_rows))
    return "\n".join(lines)

def format_detailed_matrix(matrix):
    """Detailed JSON view for debugging."""
    rows = []
    for i, row in enumerate(matrix):
        rows.append(f"Row {i}:")
        for j, cell in enumerate(row):
            if isinstance(cell, dict):
                info = {
                    "text": cell.get("text", ""),
                    "rowspan": cell.get("rowspan", 1),
                    "colspan": cell.get("colspan", 1),
                    "is_merged": cell.get("is_merged", False)
                }
                rows.append(f"  [{i}][{j}]: {json.dumps(info, ensure_ascii=False)}")
    return "\n".join(rows)

def read_pdf_blocks_and_tables(pdf_path: str, output_txt: str):
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    output_content = []
    try:
        chunks = llm.to_markdown(str(pdf_path), page_chunks=True, table_strategy="lines_strict")
        
        for idx_chunk, ch in enumerate(chunks):
            page_num = idx_chunk + 1
            output_content.append(f"Page {page_num}\n")
            
            page_markdown = ch.get("text", "")
            tables = ch.get("tables") or []
            
            if tables:
                processed_text = page_markdown
                for table in reversed(tables):
                    table_markdown = table.get("markdown", "")
                    if "matriz" in table:
                        matrix = table["matriz"]
                        # Generate optimized ASCII
                        matrix_ascii = generate_ascii_matrix(matrix)
                        matrix_detailed = format_detailed_matrix(matrix)

                        table_content = (
                            "\n=== MATRIX ASCII ===\n"
                            f"{matrix_ascii}\n\n"
                            "=== DETAILED MATRIX ===\n"
                            f"{matrix_detailed}\n"
                        )
                        
                        if table_markdown and table_markdown in processed_text:
                            processed_text = processed_text.replace(table_markdown, table_content, 1)
                output_content.append(processed_text)
            else:
                output_content.append(page_markdown)
    except Exception as e:
        output_content.append(f"\nERROR: {e}\n")
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(output_content))
    print(f"Done! Results in: {output_txt}")

if __name__ == "__main__":
    read_pdf_blocks_and_tables("Jubilant.pdf", "leitura_blocks_tabelas.txt")