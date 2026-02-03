# Guia de Configuração e Instalação

## Problema Comum: Ambiente Virtual Não Ativado

Se você receber o erro `externally-managed-environment`, significa que o pip está tentando instalar no sistema ao invés do ambiente virtual.

## Solução: Ativar o Ambiente Virtual

### Opção 1: Usar o ambiente virtual existente (`venv_pymupdf`)

```bash
# Ativar o ambiente virtual
source venv_pymupdf/bin/activate

# Verificar se está ativado (deve mostrar o caminho do venv)
which pip
which python

# Agora instalar as dependências
pip install -r requirements-dev.txt
```

### Opção 2: Criar um novo ambiente virtual

Se preferir criar um novo ambiente virtual:

```bash
# Criar novo ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements-dev.txt
```

## Verificação

Após ativar o ambiente virtual, você deve ver `(venv)` ou `(venv_pymupdf)` no início do seu prompt:

```bash
(venv_pymupdf) bruna@bruna-Aspire-A515:~/pymupdf4llm$
```

E os comandos `which pip` e `which python` devem apontar para o ambiente virtual:

```bash
which pip
# Deve mostrar: /home/bruna/pymupdf4llm/venv_pymupdf/bin/pip

which python
# Deve mostrar: /home/bruna/pymupdf4llm/venv_pymupdf/bin/python
```

## Instalação Completa

### Passo 1: Ativar o ambiente virtual
```bash
cd ~/pymupdf4llm
source venv_pymupdf/bin/activate
```

### Passo 2: Instalar o pacote em modo desenvolvimento
```bash
cd pymupdf4llm
pip install -e .
cd ..
```

### Passo 3: Instalar dependências de desenvolvimento
```bash
pip install -r requirements-dev.txt
```

### Passo 4: Verificar instalação
```bash
python -c "import pymupdf4llm; print('OK!')"
pytest --version
```

## Desativar o Ambiente Virtual

Quando terminar de trabalhar:

```bash
deactivate
```

## Troubleshooting

### Se o ambiente virtual não ativar:

1. Verifique se o Python está instalado:
   ```bash
   python3 --version
   ```

2. Instale python3-venv se necessário:
   ```bash
   sudo apt install python3-venv python3-full
   ```

3. Recrie o ambiente virtual:
   ```bash
   rm -rf venv_pymupdf
   python3 -m venv venv_pymupdf
   source venv_pymupdf/bin/activate
   ```

### Se ainda tiver problemas:

Use o pip do ambiente virtual diretamente:
```bash
./venv_pymupdf/bin/pip install -r requirements-dev.txt
```


