#!/bin/bash

# Verification script for Mac Studio ML environment
# Run this after mac_studio_setup.sh completes
# Usage: bash verify_install.sh

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASS_COUNT++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAIL_COUNT++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "============================================================================"
echo "Mac Studio ML Environment Verification"
echo "============================================================================"
echo ""

# Activate virtual environment
source ~/ml_projects/dnd_ai_env/bin/activate 2>/dev/null || {
    check_fail "Virtual environment not found at ~/ml_projects/dnd_ai_env"
    exit 1
}

check_pass "Virtual environment activated"

echo ""
echo "--- System Tools ---"

# Homebrew
if command -v brew &>/dev/null; then
    check_pass "Homebrew: $(brew --version | head -n1)"
else
    check_fail "Homebrew not installed"
fi

# Git
if command -v git &>/dev/null; then
    check_pass "Git: $(git --version)"
else
    check_fail "Git not installed"
fi

# Python
if command -v python &>/dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    check_pass "Python: $PYTHON_VERSION"
    check_pass "Python location: $(which python)"
else
    check_fail "Python not found in virtual environment"
fi

# tmux
if command -v tmux &>/dev/null; then
    check_pass "tmux: $(tmux -V)"
else
    check_warn "tmux not installed (optional)"
fi

# Go
if command -v go &>/dev/null; then
    check_pass "Go: $(go version)"
else
    check_warn "Go not installed (optional)"
fi

echo ""
echo "--- Core Python Libraries ---"

# NumPy
python -c "import numpy; print(f'NumPy {numpy.__version__}')" &>/dev/null && \
    check_pass "NumPy: $(python -c 'import numpy; print(numpy.__version__)')" || \
    check_fail "NumPy not installed"

# Pandas
python -c "import pandas; print(f'Pandas {pandas.__version__}')" &>/dev/null && \
    check_pass "Pandas: $(python -c 'import pandas; print(pandas.__version__)')" || \
    check_fail "Pandas not installed"

# Scikit-learn
python -c "import sklearn; print(f'Scikit-learn {sklearn.__version__}')" &>/dev/null && \
    check_pass "Scikit-learn: $(python -c 'import sklearn; print(sklearn.__version__)')" || \
    check_fail "Scikit-learn not installed"

echo ""
echo "--- PyTorch & MPS ---"

# PyTorch
if python -c "import torch" &>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    check_pass "PyTorch: $TORCH_VERSION"
    
    # MPS availability
    MPS_AVAILABLE=$(python -c "import torch; print(torch.backends.mps.is_available())")
    MPS_BUILT=$(python -c "import torch; print(torch.backends.mps.is_built())")
    
    if [ "$MPS_AVAILABLE" = "True" ]; then
        check_pass "MPS (Metal) GPU acceleration: Available and working"
    else
        check_fail "MPS (Metal) GPU acceleration: Not available"
    fi
    
    if [ "$MPS_BUILT" = "True" ]; then
        check_pass "PyTorch built with MPS support"
    else
        check_warn "PyTorch not built with MPS support"
    fi
else
    check_fail "PyTorch not installed"
fi

echo ""
echo "--- Transformers & LLM Libraries ---"

# Transformers
python -c "import transformers" &>/dev/null && \
    check_pass "Transformers: $(python -c 'import transformers; print(transformers.__version__)')" || \
    check_fail "Transformers not installed"

# Sentence Transformers
python -c "import sentence_transformers" &>/dev/null && \
    check_pass "Sentence Transformers: $(python -c 'import sentence_transformers; print(sentence_transformers.__version__)')" || \
    check_fail "Sentence Transformers not installed"

# LangChain
python -c "import langchain" &>/dev/null && \
    check_pass "LangChain: $(python -c 'import langchain; print(langchain.__version__)')" || \
    check_fail "LangChain not installed"

echo ""
echo "--- MLX (Apple ML Framework) ---"

# MLX
if python -c "import mlx.core as mx" &>/dev/null; then
    MLX_VERSION=$(python -c "import mlx.core as mx; print(mx.__version__)")
    check_pass "MLX: $MLX_VERSION"
else
    check_fail "MLX not installed"
fi

# MLX-LM
python -c "import mlx_lm" &>/dev/null && \
    check_pass "MLX-LM installed" || \
    check_warn "MLX-LM not installed (optional)"

echo ""
echo "--- Vector Databases ---"

# ChromaDB
python -c "import chromadb" &>/dev/null && \
    check_pass "ChromaDB: $(python -c 'import chromadb; print(chromadb.__version__)')" || \
    check_fail "ChromaDB not installed"

# FAISS
python -c "import faiss" &>/dev/null && \
    check_pass "FAISS installed" || \
    check_fail "FAISS not installed"

echo ""
echo "--- Graph Database ---"

# Neo4j Python driver
python -c "import neo4j" &>/dev/null && \
    check_pass "Neo4j Python driver: $(python -c 'import neo4j; print(neo4j.__version__)')" || \
    check_fail "Neo4j Python driver not installed"

# Neo4j server
if brew services list | grep neo4j | grep started &>/dev/null; then
    check_pass "Neo4j server: Running"
elif brew list neo4j &>/dev/null; then
    check_warn "Neo4j server: Installed but not running (Start with: brew services start neo4j)"
else
    check_fail "Neo4j server not installed"
fi

echo ""
echo "--- Jupyter ---"

# JupyterLab
if command -v jupyter &>/dev/null; then
    JUPYTER_VERSION=$(jupyter --version 2>&1 | head -n1)
    check_pass "JupyterLab: $JUPYTER_VERSION"
else
    check_fail "JupyterLab not installed"
fi

echo ""
echo "--- Additional Utilities ---"

# Weights & Biases
python -c "import wandb" &>/dev/null && \
    check_pass "Weights & Biases installed" || \
    check_warn "Weights & Biases not installed (optional)"

# tqdm
python -c "import tqdm" &>/dev/null && \
    check_pass "tqdm installed" || \
    check_warn "tqdm not installed"

# python-dotenv
python -c "import dotenv" &>/dev/null && \
    check_pass "python-dotenv installed" || \
    check_warn "python-dotenv not installed"

echo ""
echo "--- GPU Test ---"

# Quick MPS test
echo "Running quick GPU test..."
python -c "
import torch
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print('✓ Successfully performed matrix multiplication on MPS GPU')
else:
    print('✗ MPS not available for GPU test')
" || check_warn "GPU test failed"

echo ""
echo "============================================================================"
echo "Verification Summary"
echo "============================================================================"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All critical components installed successfully!${NC}"
    echo ""
    echo "Your Mac Studio is ready for ML development."
    echo ""
    echo "Quick Start:"
    echo "  1. Activate environment: source ~/ml_projects/activate_dnd.sh"
    echo "  2. Start Neo4j: brew services start neo4j"
    echo "  3. Launch Jupyter: jupyter lab"
    echo "  4. Open Neo4j browser: http://localhost:7474"
    echo ""
else
    echo -e "${RED}✗ Some components failed to install.${NC}"
    echo "Review the errors above and re-run the installation script if needed."
    echo ""
fi

deactivate
