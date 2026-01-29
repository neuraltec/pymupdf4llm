# Requirements e Dependências

Este documento descreve as dependências necessárias para executar o pymupdf4llm e seus testes.

## ⚠️ IMPORTANTE: Ative o Ambiente Virtual Primeiro!

**Antes de instalar qualquer coisa, certifique-se de que o ambiente virtual está ativado:**

```bash
# Ativar o ambiente virtual existente
source venv_pymupdf/bin/activate

# OU criar um novo
python3 -m venv venv
source venv/bin/activate
```

Você deve ver `(venv_pymupdf)` ou `(venv)` no início do seu prompt.

## Instalação Rápida

### Para uso do pacote apenas:
```bash
# Certifique-se de que o venv está ativado!
pip install -r requirements.txt
```

### Para desenvolvimento e testes:
```bash
# Certifique-se de que o venv está ativado!
pip install -r requirements-dev.txt
```

## Estrutura dos Arquivos de Requirements

### `requirements.txt`
Contém as dependências essenciais para executar o pacote:
- **pymupdf>=1.26.6**: Biblioteca principal para processamento de PDFs
- **tabulate**: Formatação de tabelas
- **opencv-python**: Verificação de OCR (usado em `check_ocr.py`)
- **numpy**: Processamento de imagens (usado em `check_ocr.py`)

### `requirements-dev.txt`
Inclui todas as dependências de produção mais as dependências de desenvolvimento:
- **pytest>=7.0.0**: Framework de testes
- **python-dotenv>=1.0.0**: Gerenciamento de variáveis de ambiente para testes
- **llama-index** (opcional): Para testes de integração com llama_index

## Dependências por Funcionalidade

### Funcionalidades Core (Sempre Necessárias)
- `pymupdf`: Processamento de PDFs, extração de texto e tabelas
- `tabulate`: Formatação de tabelas em texto

### Funcionalidades Opcionais
- `opencv-python` e `numpy`: Necessários apenas se você usar a funcionalidade de verificação de OCR (`check_ocr.py`)

### Para Testes
- `pytest`: Executar os testes
- `python-dotenv`: Carregar variáveis de ambiente do arquivo `.env` nos testes

## Instalação do Pacote em Modo Desenvolvimento

Para desenvolver ou modificar o pacote:

```bash
cd pymupdf4llm
pip install -e .
pip install -r ../requirements-dev.txt
```

## Executando os Testes

1. Configure o arquivo `.env` na raiz do projeto com:
   ```
   PDF_PATH=/caminho/para/seu/arquivo.pdf
   ```

2. Execute os testes:
   ```bash
   # Todos os testes
   pytest tests/
   
   # Testes específicos de tabelas
   pytest tests/pymupdf4llm/tables/
   
   # Teste específico
   pytest tests/pymupdf4llm/tables/test_tabela1.py
   ```

## Versões Mínimas

- Python: >= 3.10
- PyMuPDF: >= 1.26.6

## Notas

- O pacote `llama-index` é opcional e só é necessário se você quiser executar os testes de integração com llama_index
- As dependências `opencv-python` e `numpy` são necessárias apenas se você usar a funcionalidade de verificação de OCR

