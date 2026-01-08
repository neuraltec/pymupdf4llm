import pytest
from pathlib import Path
import sys
import os
import json
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o caminho até o segundo nível do pacote
base_path = Path(__file__).parent.parent.parent.parent
pymupdf_path = base_path / "pymupdf4llm" / "pymupdf4llm"
if str(pymupdf_path) not in sys.path:
    sys.path.insert(0, str(pymupdf_path))

import pymupdf4llm as llm
import fitz  # PyMuPDF


@pytest.fixture
def pdf_teste(tmp_path):
    """
    Define o caminho do PDF de teste.
    O caminho é lido da variável de ambiente PDF_PATH no arquivo .env.
    """
    pdf_path_str = os.getenv("PDF_PATH")
    assert pdf_path_str, "Variável de ambiente PDF_PATH não encontrada no arquivo .env"
    pdf_path = Path(pdf_path_str)
    assert pdf_path.exists(), f"PDF de teste não encontrado em {pdf_path}"
    return pdf_path


def extrair_primeira_tabela_llm(pdf_path: Path, strategy="lines_strict", pagina=None):
    """
    Extrai a tabela da página especificada usando PyMuPDF4LLM (fork local).
    Retorna uma tupla (dados_tabela, estrutura_completa) onde:
    - dados_tabela: lista de listas (matriz) ou None
    - estrutura_completa: dict com informações completas da tabela encontrada
    """
    chunks = llm.to_markdown(str(pdf_path), page_chunks=True, table_strategy=strategy)
    
    # Debug: mostra todos os chunks e suas tabelas
    print(f"\nBuscando primeira tabela com estratégia '{strategy}'...")
    print(f"Total de chunks (páginas): {len(chunks)}")
    
    # Busca a primeira tabela em qualquer página
    for idx_chunk, ch in enumerate(chunks):
        tabelas = ch.get("tables") or []
        print(f"  Chunk {idx_chunk + 1}: {len(tabelas)} tabela(s) encontrada(s)")
        
        if tabelas:
            tabela = tabelas[0]
            print(f"  Primeira tabela encontrada no chunk {idx_chunk + 1}")
            
            # Tenta extrair a matriz de diferentes formas
            dados = None
            if "matriz" in tabela:
                dados = tabela["matriz"]
            elif "data" in tabela:
                dados = tabela["data"]
            elif "markdown" in tabela:
                # Se só tiver markdown, tenta converter
                dados = tabela["markdown"]
            else:
                # Retorna a estrutura completa para debug
                dados = tabela
            
            return dados, tabela
    
    return None, None


def extrair_primeira_tabela_pymupdf(pdf_path: Path, strategy="lines_strict", pagina=None):
    """
    Extrai a primeira tabela da página especificada usando PyMuPDF diretamente (fallback).
    Retorna uma tupla (dados_tabela, estrutura_completa).
    """
    doc = fitz.open(str(pdf_path))
    
    print(f"\nTentando PyMuPDF diretamente com estratégia '{strategy}'...")
    
    # Busca a primeira tabela em qualquer página
    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables(strategy=strategy)
        
        if tables.tables:
            print(f"  Tabela encontrada na página {page_num + 1}")
            primeira_tabela = tables.tables[0]
            
            # Converte para matriz
            try:
                matriz = primeira_tabela.extract()
                estrutura = {
                    "bbox": primeira_tabela.bbox,
                    "rows": primeira_tabela.row_count,
                    "cols": primeira_tabela.col_count,
                    "markdown": primeira_tabela.to_markdown()
                }
                doc.close()
                return matriz, estrutura
            except Exception as e:
                print(f"  Erro ao extrair tabela: {e}")
                doc.close()
                return None, None
    
    doc.close()
    return None, None



def test_matriz_ascii_comparacao_imagem(pdf_teste):
    """
    Testa se a matriz ASCII extraída corresponde
    exatamente ao formato esperado.
    """
    
    # Define o resultado esperado exato
    matriz_ascii_esperada = """------------------------------------------
| STAGE : ARP-3                          |
------------------------------------------
| Input batch size   | Output batch size |
------------------------------------------
| 55 – 60 Kg of ARP2 | 43.18 to 57.6     |
------------------------------------------"""
    
    # Tenta diferentes estratégias com pymupdf4llm
    estrategias = ["lines_strict", "lines", "text"]
    estrutura_completa = None
    estrategia_usada = None
    metodo_usado = None
    
    for strategy in estrategias:
        chunks = llm.to_markdown(str(pdf_teste), page_chunks=True, table_strategy=strategy)
        
        # Busca a primeira tabela em qualquer página
        for idx_chunk, ch in enumerate(chunks):
            tabelas = ch.get("tables") or []
            if tabelas:
                estrutura_completa = tabelas[0]
                estrategia_usada = strategy
                metodo_usado = "pymupdf4llm"
                break
        
        if estrutura_completa:
            break
    
    # Se não encontrou com pymupdf4llm, tenta PyMuPDF diretamente
    if estrutura_completa is None:
        for strategy in estrategias:
            doc = fitz.open(str(pdf_teste))
            try:
                # Busca a primeira tabela em qualquer página
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    tables = page.find_tables(strategy=strategy)
                    if tables.tables:
                        primeira_tabela = tables.tables[0]
                        matriz = primeira_tabela.extract()
                        break
                        # Converte para o formato esperado
                        # Importa a função matriz_to_ascii do módulo helpers
                        # Usa caminho relativo ao projeto
                        base_path = Path(__file__).parent.parent.parent.parent
                        helpers_path = base_path / "pymupdf4llm" / "pymupdf4llm" / "helpers"
                        if str(helpers_path) not in sys.path:
                            sys.path.insert(0, str(helpers_path))
                        from pymupdf_rag import matriz_to_ascii
                        # Converte matriz simples para formato com dicionários se necessário
                        matriz_formatada = []
                        for row_idx, row in enumerate(matriz):
                            matriz_row = []
                            for col_idx, cell in enumerate(row):
                                if isinstance(cell, dict):
                                    matriz_row.append(cell)
                                else:
                                    matriz_row.append({
                                        "text": str(cell) if cell is not None else "",
                                        "row": row_idx,
                                        "col": col_idx,
                                        "rowspan": 1,
                                        "colspan": 1
                                    })
                            matriz_formatada.append(matriz_row)
                        matriz_ascii = matriz_to_ascii(matriz_formatada)
                        estrutura_completa = {
                            "matriz_ascii": matriz_ascii,
                            "matriz": matriz
                        }
                        estrategia_usada = strategy
                        metodo_usado = "pymupdf_direto"
                        break
            finally:
                doc.close()
            if estrutura_completa:
                break
    
    # Mostra o que foi encontrado
    print("\n" + "="*80)
    print("TESTE DE COMPARAÇÃO EXATA DA MATRIZ ASCII")
    print("="*80)
    
    if estrutura_completa is None:
        print("Nenhuma tabela foi detectada no PDF.")
        pytest.fail("Nenhuma tabela foi detectada no PDF.")
    
    print(f"Tabela encontrada usando método: '{metodo_usado}' com estratégia: '{estrategia_usada}'")
    
    # Obtém a matriz ASCII
    matriz_ascii = estrutura_completa.get("matriz_ascii")
    
    if matriz_ascii is None:
        print("\nA tabela não possui o campo 'matriz_ascii'.")
        print(f"Chaves disponíveis na estrutura: {list(estrutura_completa.keys())}")
        pytest.fail("A tabela extraída não possui o campo 'matriz_ascii'.")
    
    # Normaliza ambas as matrizes para comparação (remove espaços em branco no final das linhas)
    matriz_ascii_normalizada = "\n".join(linha.rstrip() for linha in matriz_ascii.split("\n"))
    matriz_ascii_esperada_normalizada = "\n".join(linha.rstrip() for linha in matriz_ascii_esperada.split("\n"))
    
    print(f"\nMATRIZ ASCII ESPERADA:")
    print("-"*80)
    print(matriz_ascii_esperada_normalizada)
    print("-"*80)
    
    print(f"\nMATRIZ ASCII EXTRAÍDA:")
    print("-"*80)
    print(matriz_ascii_normalizada)
    print("-"*80)
    
    # Comparação exata linha por linha
    linhas_esperadas = matriz_ascii_esperada_normalizada.split("\n")
    linhas_obtidas = matriz_ascii_normalizada.split("\n")
    
    print(f"\nCOMPARAÇÃO LINHA POR LINHA:")
    print("-"*80)
    erros = []
    
    # Verifica se o número de linhas é o mesmo
    if len(linhas_obtidas) != len(linhas_esperadas):
        erros.append(
            f"Número de linhas diferente:\n"
            f"   Esperado: {len(linhas_esperadas)} linhas\n"
            f"   Obtido: {len(linhas_obtidas)} linhas"
        )
        print(f"✗ Número de linhas diferente: esperado {len(linhas_esperadas)}, obtido {len(linhas_obtidas)}")
    else:
        print(f"✓ Número de linhas correto: {len(linhas_esperadas)}")
    
    # Compara cada linha
    max_linhas = max(len(linhas_esperadas), len(linhas_obtidas))
    for i in range(max_linhas):
        if i < len(linhas_esperadas) and i < len(linhas_obtidas):
            esperada = linhas_esperadas[i]
            obtida = linhas_obtidas[i]
            if esperada == obtida:
                print(f"✓ Linha {i+1}: OK")
            else:
                erros.append(
                    f"Linha {i+1} diferente:\n"
                    f"   Esperado: '{esperada}'\n"
                    f"   Obtido:   '{obtida}'"
                )
                print(f"✗ Linha {i+1}: DIFERENTE")
                print(f"    Esperado: '{esperada}'")
                print(f"    Obtido:   '{obtida}'")
        elif i < len(linhas_esperadas):
            erros.append(f"Linha {i+1} faltando na matriz extraída. Esperado: '{linhas_esperadas[i]}'")
            print(f"✗ Linha {i+1}: FALTANDO (esperado: '{linhas_esperadas[i]}')")
        else:
            erros.append(f"Linha {i+1} extra na matriz extraída. Obtido: '{linhas_obtidas[i]}'")
            print(f"✗ Linha {i+1}: EXTRA (obtido: '{linhas_obtidas[i]}')")
    
    # Comparação exata completa
    if matriz_ascii_normalizada == matriz_ascii_esperada_normalizada:
        print("\n" + "="*80)
        print("RESULTADO FINAL: matriz ASCII corresponde EXATAMENTE ao formato esperado")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("RESULTADO FINAL: diferenças encontradas entre a matriz ASCII e o formato esperado")
        print("="*80)
        print(f"\nTotal de erros: {len(erros)}")
        print("\nErros encontrados:")
        for e in erros:
            print(f"  - {e}")
        print(f"\nMatriz ASCII esperada:")
        print("-"*80)
        print(matriz_ascii_esperada_normalizada)
        print("-"*80)
        print(f"\nMatriz ASCII obtida:")
        print("-"*80)
        print(matriz_ascii_normalizada)
        print("-"*80)
        pytest.fail(f"A matriz ASCII extraída não corresponde exatamente ao formato esperado.\nTotal de erros: {len(erros)}")

