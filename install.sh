#!/bin/bash
# Script de instalação para pymupdf4llm
# Este script ajuda a configurar o ambiente virtual e instalar as dependências

set -e  # Parar em caso de erro

echo "=========================================="
echo "Instalação do pymupdf4llm"
echo "=========================================="
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "requirements.txt" ]; then
    echo "❌ Erro: requirements.txt não encontrado!"
    echo "   Execute este script a partir do diretório raiz do projeto."
    exit 1
fi

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Erro: Python 3 não encontrado!"
    echo "   Instale Python 3.10 ou superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python encontrado: $(python3 --version)"

# Verificar versão mínima do Python (3.10)
if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]; then
    echo "⚠️  Aviso: Python 3.10 ou superior recomendado"
fi

# Verificar se o ambiente virtual existe
VENV_NAME="venv_pymupdf"
if [ -d "$VENV_NAME" ]; then
    echo "✓ Ambiente virtual '$VENV_NAME' encontrado"
    USE_EXISTING=true
else
    echo "ℹ️  Ambiente virtual '$VENV_NAME' não encontrado"
    USE_EXISTING=false
fi

# Perguntar ao usuário
if [ "$USE_EXISTING" = true ]; then
    echo ""
    read -p "Usar ambiente virtual existente '$VENV_NAME'? (S/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        USE_EXISTING=true
    else
        USE_EXISTING=false
    fi
fi

# Criar ou usar ambiente virtual
if [ "$USE_EXISTING" = false ]; then
    echo ""
    read -p "Nome do ambiente virtual (padrão: venv): " VENV_NAME_INPUT
    VENV_NAME=${VENV_NAME_INPUT:-venv}
    
    if [ -d "$VENV_NAME" ]; then
        echo "⚠️  Ambiente virtual '$VENV_NAME' já existe!"
        read -p "Remover e recriar? (s/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            rm -rf "$VENV_NAME"
            echo "✓ Ambiente virtual removido"
        else
            echo "Usando ambiente virtual existente..."
        fi
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        echo ""
        echo "Criando ambiente virtual '$VENV_NAME'..."
        python3 -m venv "$VENV_NAME"
        echo "✓ Ambiente virtual criado"
    fi
fi

# Definir caminhos absolutos
VENV_PATH="$(cd "$VENV_NAME" && pwd)"
PIP_CMD="$VENV_PATH/bin/pip"
PYTHON_CMD="$VENV_PATH/bin/python"

# Verificar se pip existe no venv
if [ ! -f "$PIP_CMD" ]; then
    echo "❌ Erro: pip não encontrado em $PIP_CMD"
    exit 1
fi

# Verificar se python existe no venv
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Erro: python não encontrado em $PYTHON_CMD"
    exit 1
fi

echo ""
echo "✓ Usando pip do ambiente virtual: $PIP_CMD"
echo "✓ Usando python do ambiente virtual: $PYTHON_CMD"

# Ativar ambiente virtual (para PATH, mas usaremos caminhos absolutos)
source "$VENV_PATH/bin/activate" 2>/dev/null || true

# Atualizar pip
echo ""
echo "Atualizando pip..."
"$PIP_CMD" install --upgrade pip --quiet
echo "✓ pip atualizado"

# Instalar o pacote em modo desenvolvimento
echo ""
echo "Instalando pymupdf4llm em modo desenvolvimento..."
cd pymupdf4llm
"$PIP_CMD" install -e . --quiet
cd ..
echo "✓ pymupdf4llm instalado"

# Instalar dependências de desenvolvimento
echo ""
echo "Instalando dependências de desenvolvimento..."
"$PIP_CMD" install -r requirements-dev.txt --quiet
echo "✓ Dependências instaladas"

# Verificar instalação
echo ""
echo "Verificando instalação..."
if "$PYTHON_CMD" -c "import pymupdf4llm; print('OK')" 2>/dev/null; then
    echo "✓ pymupdf4llm importado com sucesso"
else
    echo "⚠️  Aviso: Não foi possível importar pymupdf4llm"
fi

PYTEST_CMD="$VENV_NAME/bin/pytest"
if [ -f "$PYTEST_CMD" ] && "$PYTEST_CMD" --version &> /dev/null; then
    echo "✓ pytest instalado: $("$PYTEST_CMD" --version)"
else
    echo "⚠️  Aviso: pytest não encontrado"
fi

echo ""
echo "=========================================="
echo "✅ Instalação concluída!"
echo "=========================================="
echo ""
echo "Para usar o ambiente virtual no futuro:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Para desativar:"
echo "  deactivate"
echo ""

