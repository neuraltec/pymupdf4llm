#!/bin/bash
# Script simples de instalação que usa caminhos absolutos
# Este script evita problemas com ativação de ambiente virtual

set -e

VENV_NAME="${1:-venv_pymupdf}"

if [ ! -d "$VENV_NAME" ]; then
    echo "❌ Ambiente virtual '$VENV_NAME' não encontrado!"
    echo "   Criando novo ambiente virtual..."
    python3 -m venv "$VENV_NAME"
fi

# Caminhos absolutos
VENV_PATH="$(cd "$VENV_NAME" && pwd)"
PIP_CMD="$VENV_PATH/bin/pip"
PYTHON_CMD="$VENV_PATH/bin/python"

echo "=========================================="
echo "Instalação do pymupdf4llm"
echo "=========================================="
echo ""
echo "Ambiente virtual: $VENV_PATH"
echo "Pip: $PIP_CMD"
echo "Python: $PYTHON_CMD"
echo ""

# Atualizar pip
echo "Atualizando pip..."
"$PIP_CMD" install --upgrade pip --quiet

# Instalar o pacote em modo desenvolvimento
echo "Instalando pymupdf4llm..."
cd pymupdf4llm
"$PIP_CMD" install -e . --quiet
cd ..

# Instalar dependências de desenvolvimento
echo "Instalando dependências..."
"$PIP_CMD" install -r requirements-dev.txt --quiet

# Verificar
echo ""
echo "Verificando instalação..."
if "$PYTHON_CMD" -c "import pymupdf4llm; print('OK')" 2>/dev/null; then
    echo "✓ pymupdf4llm instalado com sucesso"
else
    echo "⚠️  Aviso: Não foi possível importar pymupdf4llm"
fi

echo ""
echo "=========================================="
echo "✅ Instalação concluída!"
echo "=========================================="
echo ""
echo "Para usar o ambiente virtual:"
echo "  source $VENV_NAME/bin/activate"
echo ""


