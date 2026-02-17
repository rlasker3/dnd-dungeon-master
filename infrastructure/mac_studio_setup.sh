#!/bin/bash

# Mac Studio M4 Max - ML/AI Development Environment Setup
# Run this script via SSH after initial macOS setup is complete
# Usage: bash mac_studio_setup.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log "Starting Mac Studio ML Environment Setup..."
log "This will take 30-60 minutes. You can disconnect and it will continue running."

# ============================================================================
# 1. SYSTEM PREPARATION
# ============================================================================
log "Step 1: System Preparation"

# Check for Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    log "Installing Xcode Command Line Tools..."
    xcode-select --install
    warn "Please complete the Xcode installation dialog, then re-run this script."
    exit 0
else
    log "Xcode Command Line Tools already installed ✓"
fi

# ============================================================================
# 2. HOMEBREW INSTALLATION
# ============================================================================
log "Step 2: Installing Homebrew"

if command -v brew &>/dev/null; then
    log "Homebrew already installed ✓"
    brew update
else
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
    log "Homebrew installed ✓"
fi

# ============================================================================
# 3. PYTHON INSTALLATION
# ============================================================================
log "Step 3: Installing Python"

if brew list python@3.11 &>/dev/null; then
    log "Python 3.11 already installed ✓"
else
    log "Installing Python 3.11..."
    brew install python@3.11
    log "Python 3.11 installed ✓"
fi

# Verify Python installation
PYTHON_PATH=$(brew --prefix python@3.11)/bin/python3.11
if [ ! -f "$PYTHON_PATH" ]; then
    error "Python installation failed"
fi

log "Python version: $($PYTHON_PATH --version)"

# ============================================================================
# 4. GIT INSTALLATION & CONFIGURATION
# ============================================================================
log "Step 4: Installing Git"

if command -v git &>/dev/null; then
    log "Git already installed ✓"
else
    brew install git
    log "Git installed ✓"
fi

log "Git version: $(git --version)"

# ============================================================================
# 5. ESSENTIAL DEVELOPMENT TOOLS
# ============================================================================
log "Step 5: Installing Essential Development Tools"

log "Installing tmux (for persistent terminal sessions)..."
brew install tmux || warn "tmux installation failed"

log "Installing htop (for system monitoring)..."
brew install htop || warn "htop installation failed"

log "Installing wget..."
brew install wget || warn "wget installation failed"

# ============================================================================
# 6. NEO4J DATABASE (for D&D graph relationships)
# ============================================================================
log "Step 6: Installing Neo4j"

if brew list neo4j &>/dev/null; then
    log "Neo4j already installed ✓"
else
    log "Installing Neo4j..."
    brew install neo4j
    log "Neo4j installed ✓"
    warn "Neo4j requires Java. It will be installed as a dependency."
fi

# ============================================================================
# 7. PYTHON ML/AI ENVIRONMENT
# ============================================================================
log "Step 7: Setting up Python ML/AI Environment"

# Create a projects directory
mkdir -p ~/ml_projects
cd ~/ml_projects

# Create virtual environment for D&D AI project
log "Creating Python virtual environment..."
$PYTHON_PATH -m venv dnd_ai_env
source dnd_ai_env/bin/activate

log "Virtual environment created at: ~/ml_projects/dnd_ai_env"

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ============================================================================
# 8. CORE ML LIBRARIES
# ============================================================================
log "Step 8: Installing Core ML Libraries (this takes a while)..."

log "Installing NumPy..."
pip install numpy

log "Installing Pandas..."
pip install pandas

log "Installing Scikit-learn..."
pip install scikit-learn

log "Installing Matplotlib and Seaborn (visualization)..."
pip install matplotlib seaborn

# ============================================================================
# 9. PYTORCH WITH MPS (Metal Performance Shaders) SUPPORT
# ============================================================================
log "Step 9: Installing PyTorch with MPS support..."

log "Installing PyTorch for Apple Silicon..."
pip install torch torchvision torchaudio

log "Verifying PyTorch MPS support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

# ============================================================================
# 10. TRANSFORMERS & LLM LIBRARIES
# ============================================================================
log "Step 10: Installing Transformers and LLM libraries..."

log "Installing Hugging Face Transformers..."
pip install transformers

log "Installing Hugging Face Accelerate (for optimization)..."
pip install accelerate

log "Installing sentence-transformers (for embeddings)..."
pip install sentence-transformers

log "Installing tokenizers..."
pip install tokenizers

# ============================================================================
# 11. MLX (Apple's ML Framework)
# ============================================================================
log "Step 11: Installing MLX framework..."

log "Installing MLX..."
pip install mlx

log "Installing MLX-LM (for language models)..."
pip install mlx-lm

log "Verifying MLX installation..."
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')" || warn "MLX verification failed"

# ============================================================================
# 12. VECTOR DATABASES (for RAG/Memory)
# ============================================================================
log "Step 12: Installing Vector Database libraries..."

log "Installing ChromaDB..."
pip install chromadb

log "Installing FAISS (Facebook AI Similarity Search)..."
pip install faiss-cpu

log "Installing LangChain (for RAG orchestration)..."
pip install langchain langchain-community

# ============================================================================
# 13. NEO4J PYTHON DRIVER
# ============================================================================
log "Step 13: Installing Neo4j Python driver..."
pip install neo4j

# ============================================================================
# 14. JUPYTER & NOTEBOOK TOOLS
# ============================================================================
log "Step 14: Installing Jupyter..."

log "Installing JupyterLab..."
pip install jupyterlab

log "Installing IPython..."
pip install ipython

log "Installing notebook extensions..."
pip install jupyter_contrib_nbextensions

# ============================================================================
# 15. ADDITIONAL AI/ML UTILITIES
# ============================================================================
log "Step 15: Installing additional utilities..."

log "Installing Weights & Biases (experiment tracking)..."
pip install wandb

log "Installing tqdm (progress bars)..."
pip install tqdm

log "Installing python-dotenv (environment variables)..."
pip install python-dotenv

log "Installing requests (HTTP library)..."
pip install requests

log "Installing PyYAML..."
pip install pyyaml

log "Installing Web framework (FastAPI/Uvicorn)..."
pip install fastapi uvicorn

log "Installing Development Tools (testing/formatting)..."
pip install pytest black flake8

log "Installing System Utilities (psutil)..."
pip install psutil

# ============================================================================
# 16. GO LANGUAGE (Optional - for high-performance components)
# ============================================================================
log "Step 16: Installing Go language..."

if command -v go &>/dev/null; then
    log "Go already installed ✓"
else
    log "Installing Go..."
    brew install go
    log "Go installed ✓"
    
    # Set up Go workspace
    mkdir -p ~/go/{bin,src,pkg}
    if ! grep -q "export GOPATH=\$HOME/go" ~/.zprofile; then
        echo 'export GOPATH=$HOME/go' >> ~/.zprofile
        echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.zprofile
    fi
fi

log "Go version: $(go version 2>/dev/null || echo 'Not installed')"

# ============================================================================
# 17. CREATE REQUIREMENTS FILE
# ============================================================================
log "Step 17: Saving installed packages to requirements.txt..."

pip freeze > ~/ml_projects/dnd_ai_env/requirements.txt
log "Requirements saved to ~/ml_projects/dnd_ai_env/requirements.txt"

# ============================================================================
# 18. SYSTEM OPTIMIZATION SETTINGS
# ============================================================================
log "Step 18: Applying system optimization settings..."

# Increase file descriptor limits for ML workloads
if ! grep -q "ulimit -n 10240" ~/.zprofile; then
    echo 'ulimit -n 10240' >> ~/.zprofile
    log "Increased file descriptor limit ✓"
fi

# ============================================================================
# 19. CREATE ACTIVATION HELPER SCRIPT
# ============================================================================
log "Step 19: Creating activation helper script..."

cat > ~/ml_projects/activate_dnd.sh << 'EOF'
#!/bin/bash
# Quick activation script for D&D AI environment
source ~/ml_projects/dnd_ai_env/bin/activate
echo "D&D AI environment activated!"
echo "Python: $(which python)"
echo "Location: ~/ml_projects/dnd_ai_env"
EOF

chmod +x ~/ml_projects/activate_dnd.sh

log "Created activation script: ~/ml_projects/activate_dnd.sh"

# ============================================================================
# INSTALLATION COMPLETE
# ============================================================================

echo ""
echo "============================================================================"
log "✓ Installation Complete!"
echo "============================================================================"
echo ""
log "Environment Details:"
echo "  - Virtual environment: ~/ml_projects/dnd_ai_env"
echo "  - Activation command: source ~/ml_projects/activate_dnd.sh"
echo "  - Requirements file: ~/ml_projects/dnd_ai_env/requirements.txt"
echo ""
log "Installed Components:"
echo "  ✓ Python 3.11 with pip"
echo "  ✓ PyTorch with MPS (Metal) GPU support"
echo "  ✓ Transformers & LLM libraries"
echo "  ✓ MLX (Apple's ML framework)"
echo "  ✓ NumPy, Pandas, Scikit-learn"
echo "  ✓ JupyterLab"
echo "  ✓ ChromaDB & FAISS (vector databases)"
echo "  ✓ LangChain (RAG framework)"
echo "  ✓ Neo4j database & Python driver"
echo "  ✓ Go language"
echo "  ✓ Development tools (tmux, htop, git)"
echo ""
log "Next Steps:"
echo "  1. Run verification script: bash verify_install.sh"
echo "  2. Activate environment: source ~/ml_projects/activate_dnd.sh"
echo "  3. Start Neo4j: brew services start neo4j"
echo "  4. Access Neo4j browser: http://localhost:7474"
echo "  5. Start JupyterLab: jupyter lab"
echo ""
log "Neo4j Default Credentials:"
echo "  - URL: bolt://localhost:7687"
echo "  - Username: neo4j"
echo "  - Password: neo4j (you'll be prompted to change on first login)"
echo ""

deactivate 2>/dev/null || true

log "Setup script finished successfully!"
