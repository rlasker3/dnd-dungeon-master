#!/bin/bash

# D&D AI Project Structure Setup
# Creates organized directory structure and starter files
# Run this after mac_studio_setup.sh completes
# Usage: bash create_project.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[CREATE]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

echo "============================================================================"
echo "D&D Dungeon Master AI - Project Structure Setup"
echo "============================================================================"
echo ""

# Base project directory
PROJECT_DIR=~/ml_projects/dnd_dungeon_master
log "Creating project at: $PROJECT_DIR"

# Create main directory structure
mkdir -p $PROJECT_DIR/{src,data,models,notebooks,configs,tests,docs,scripts}

# Create source code subdirectories
mkdir -p $PROJECT_DIR/src/{core,llm,memory,rules,agents,world,api}

# Create data subdirectories
mkdir -p $PROJECT_DIR/data/{raw,processed,campaigns,characters,world_data}

# Create models subdirectories
mkdir -p $PROJECT_DIR/models/{checkpoints,fine_tuned,embeddings}

# Create config subdirectories
mkdir -p $PROJECT_DIR/configs/{database,model,game_rules}

log "Directory structure created"

# ============================================================================
# CREATE STARTER FILES
# ============================================================================

# Main README
cat > $PROJECT_DIR/README.md << 'EOF'
# D&D Dungeon Master AI

An intelligent AI system for running D&D campaigns with natural language interaction, persistent memory, and rule enforcement.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Chat / Voice / Web)                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Core Orchestrator                       │
│         (Coordinates all AI subsystems)                  │
└─────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   LLM    │  │  Memory  │  │  Rules   │  │  World   │
│  Engine  │  │  System  │  │  Engine  │  │  State   │
│          │  │          │  │          │  │          │
│ Narrate  │  │  Vector  │  │  D&D     │  │  Neo4j   │
│ Dialogue │  │   DB     │  │  Logic   │  │  Graph   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

```

## Components

### 1. LLM Engine (`src/llm/`)
- Story narration and dialogue generation
- NPC personality and speech patterns
- Dynamic quest and encounter descriptions

### 2. Memory System (`src/memory/`)
- Vector database for campaign history
- Semantic search for relevant context
- Long-term and short-term memory management

### 3. Rules Engine (`src/rules/`)
- D&D 5e mechanics enforcement
- Combat system
- Skill checks and saving throws
- Spell and ability resolution

### 4. World State (`src/world/`)
- Neo4j graph database for relationships
- Location tracking
- NPC relationships and factions
- Quest progression

### 5. Agent System (`src/agents/`)
- NPC behavior and decision-making
- Dynamic encounter adjustment
- Collaborative storytelling agents

## Learning Objectives

Through building this system, you'll gain deep understanding of:

- **Local LLM deployment** (MLX, PyTorch)
- **Retrieval-Augmented Generation (RAG)**
- **Vector embeddings and similarity search**
- **Graph databases for complex relationships**
- **Multi-agent AI coordination**
- **Prompt engineering and context management**
- **Model fine-tuning on domain-specific data**
- **Real-time inference optimization**
- **Stateful conversation management**

## Getting Started

1. Activate environment: `source ~/ml_projects/activate_dnd.sh`
2. Start Neo4j: `brew services start neo4j`
3. Start Jupyter: `jupyter lab`
4. Open `notebooks/01_getting_started.ipynb`

## Development Roadmap

- [ ] Phase 1: Basic chatbot with D&D rules lookup
- [ ] Phase 2: Add memory and context
- [ ] Phase 3: Integrate local LLM
- [ ] Phase 4: Implement NPC agents
- [ ] Phase 5: Add world state tracking
- [ ] Phase 6: Fine-tune for D&D narration
- [ ] Phase 7: Build web interface

EOF

# .gitignore
cat > $PROJECT_DIR/.gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# Models (too large for git)
models/checkpoints/*
models/fine_tuned/*
!models/checkpoints/.gitkeep
!models/fine_tuned/.gitkeep

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Neo4j
neo4j/data/
neo4j/logs/
EOF

# Environment template
cat > $PROJECT_DIR/.env.template << 'EOF'
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Vector Database
CHROMA_PERSIST_DIR=./data/chroma_db

# Model Configuration
MODEL_PATH=./models/
DEFAULT_LLM=mlx-community/Mistral-7B-Instruct-v0.2-4bit

# API Keys (if using external services)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Logging
LOG_LEVEL=INFO
EOF

# Requirements file (in case pip freeze doesn't work)
cat > $PROJECT_DIR/requirements.txt << 'EOF'
# Core ML/AI
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.25.0
sentence-transformers>=2.2.0
mlx>=0.1.0
mlx-lm>=0.1.0

# Vector Databases
chromadb>=0.4.0
faiss-cpu>=1.7.4

# Graph Database
neo4j>=5.14.0

# RAG & LLM Tools
langchain>=0.1.0
langchain-community>=0.0.10

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0
psutil>=5.9.0

# Development
jupyterlab>=4.0.0
ipython>=8.12.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0

# Experiment Tracking
wandb>=0.16.0

# API & Web (for future)
fastapi>=0.104.0
uvicorn>=0.24.0
EOF

# Starter config file
cat > $PROJECT_DIR/configs/model/default.yaml << 'EOF'
# Default Model Configuration

llm:
  model_name: "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  device: "mps"  # Use Metal GPU

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

memory:
  vector_db: "chroma"
  collection_name: "dnd_campaign_memory"
  max_history: 50  # Number of conversation turns to remember

agents:
  npc_temperature: 0.8  # NPCs should be more creative
  dm_temperature: 0.7   # DM should be balanced
EOF

# Database config
cat > $PROJECT_DIR/configs/database/neo4j.yaml << 'EOF'
# Neo4j Configuration

connection:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "neo4j"  # Change after first login
  
indexes:
  - label: "Character"
    property: "name"
  - label: "Location"
    property: "name"
  - label: "Quest"
    property: "quest_id"
  - label: "NPC"
    property: "npc_id"

constraints:
  - label: "Character"
    property: "character_id"
    type: "unique"
  - label: "NPC"
    property: "npc_id"
    type: "unique"
EOF

# Game rules config
cat > $PROJECT_DIR/configs/game_rules/dnd5e.yaml << 'EOF'
# D&D 5th Edition Rules Configuration

ability_scores:
  - STR
  - DEX
  - CON
  - INT
  - WIS
  - CHA

skills:
  acrobatics: DEX
  animal_handling: WIS
  arcana: INT
  athletics: STR
  deception: CHA
  history: INT
  insight: WIS
  intimidation: CHA
  investigation: INT
  medicine: WIS
  nature: INT
  perception: WIS
  performance: CHA
  persuasion: CHA
  religion: INT
  sleight_of_hand: DEX
  stealth: DEX
  survival: WIS

dice:
  d4: 4
  d6: 6
  d8: 8
  d10: 10
  d12: 12
  d20: 20
  d100: 100

difficulty_classes:
  very_easy: 5
  easy: 10
  medium: 15
  hard: 20
  very_hard: 25
  nearly_impossible: 30
EOF

# Core module init
cat > $PROJECT_DIR/src/__init__.py << 'EOF'
"""
D&D Dungeon Master AI System
"""

__version__ = "0.1.0"
__author__ = "Rich"
EOF

# Core orchestrator stub
cat > $PROJECT_DIR/src/core/orchestrator.py << 'EOF'
"""
Core Orchestrator
Coordinates all AI subsystems (LLM, Memory, Rules, World State)
"""

class DungeonMasterOrchestrator:
    """
    Main coordinator for the D&D AI system.
    
    Responsibilities:
    - Route user input to appropriate subsystems
    - Coordinate responses between LLM, rules engine, and world state
    - Manage conversation flow
    - Handle error states and fallbacks
    """
    
    def __init__(self, config):
        self.config = config
        self.llm_engine = None
        self.memory_system = None
        self.rules_engine = None
        self.world_state = None
        
    def initialize(self):
        """Initialize all subsystems"""
        # TODO: Initialize LLM, Memory, Rules, World
        pass
        
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate DM response
        
        Args:
            user_input: Player's message or action
            
        Returns:
            DM's narrative response
        """
        # TODO: Implement main processing loop
        # 1. Analyze input intent
        # 2. Check if rules need to be applied
        # 3. Query memory for context
        # 4. Update world state
        # 5. Generate narrative response
        # 6. Store interaction in memory
        return "DM response placeholder"
        
    def start_campaign(self, campaign_data: dict):
        """Initialize a new campaign"""
        # TODO: Set up campaign in Neo4j, initialize starting state
        pass
        
    def save_state(self):
        """Save current game state"""
        # TODO: Persist all subsystem states
        pass
        
    def load_state(self, campaign_id: str):
        """Load existing campaign state"""
        # TODO: Restore from databases
        pass
EOF

# Getting started notebook
cat > $PROJECT_DIR/notebooks/01_getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Getting Started with D&D AI System\n",
    "\n",
    "This notebook will walk you through the basics of the environment and verify everything works."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Verify PyTorch and MPS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import sys\n",
    "\n",
    "print(f\"Python version: {sys.version}\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"MPS available: {torch.backends.mps.is_available()}\")\n",
    "print(f\"MPS built: {torch.backends.mps.is_built()}\")\n",
    "\n",
    "if torch.backends.mps.is_available():\n",
    "    device = torch.device(\"mps\")\n",
    "    x = torch.randn(1000, 1000, device=device)\n",
    "    y = torch.matmul(x, x.T)\n",
    "    print(\"\\n✓ Successfully performed matrix multiplication on MPS GPU!\")\n",
    "else:\n",
    "    print(\"\\n✗ MPS not available\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Test MLX Framework"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mlx.core as mx\n",
    "\n",
    "print(f\"MLX version: {mx.__version__}\")\n",
    "\n",
    "# Quick MLX test\n",
    "a = mx.array([1.0, 2.0, 3.0])\n",
    "b = mx.array([4.0, 5.0, 6.0])\n",
    "c = a + b\n",
    "\n",
    "print(f\"\\nMLX array operation: {a} + {b} = {c}\")\n",
    "print(\"✓ MLX working correctly!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Test Transformers Library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "# Load a small tokenizer for testing\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n",
    "\n",
    "text = \"You enter a dimly lit tavern. The smell of ale fills the air.\"\n",
    "tokens = tokenizer.encode(text)\n",
    "\n",
    "print(f\"Original text: {text}\")\n",
    "print(f\"Tokens: {tokens}\")\n",
    "print(f\"Token count: {len(tokens)}\")\n",
    "print(\"\\n✓ Transformers library working!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Test Vector Database (ChromaDB)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import chromadb\n",
    "\n",
    "# Create an in-memory ChromaDB client\n",
    "client = chromadb.Client()\n",
    "\n",
    "# Create a collection\n",
    "collection = client.create_collection(\"test_campaign\")\n",
    "\n",
    "# Add some D&D-related documents\n",
    "collection.add(\n",
    "    documents=[\n",
    "        \"The party encounters a dragon in its lair.\",\n",
    "        \"You find a magic sword in the treasure chest.\",\n",
    "        \"The wizard casts fireball at the goblins.\"\n",
    "    ],\n",
    "    ids=[\"event1\", \"event2\", \"event3\"]\n",
    ")\n",
    "\n",
    "# Query for similar content\n",
    "results = collection.query(\n",
    "    query_texts=[\"fighting monsters\"],\n",
    "    n_results=2\n",
    ")\n",
    "\n",
    "print(\"Query: 'fighting monsters'\")\n",
    "print(f\"\\nMost relevant results:\")\n",
    "for doc in results['documents'][0]:\n",
    "    print(f\"  - {doc}\")\n",
    "\n",
    "print(\"\\n✓ ChromaDB vector search working!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Test Neo4j Connection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from neo4j import GraphDatabase\n",
    "\n",
    "# Note: Update password after first Neo4j login\n",
    "uri = \"bolt://localhost:7687\"\n",
    "user = \"neo4j\"\n",
    "password = \"neo4j\"  # Change this!\n",
    "\n",
    "try:\n",
    "    driver = GraphDatabase.driver(uri, auth=(user, password))\n",
    "    \n",
    "    # Test connection\n",
    "    with driver.session() as session:\n",
    "        result = session.run(\"RETURN 'Hello from Neo4j!' AS message\")\n",
    "        message = result.single()[\"message\"]\n",
    "        print(message)\n",
    "    \n",
    "    driver.close()\n",
    "    print(\"\\n✓ Neo4j connection successful!\")\n",
    "    print(\"\\nNote: Change default password at http://localhost:7474\")\n",
    "    \n",
    "except Exception as e:\n",
    "    print(f\"✗ Neo4j connection failed: {e}\")\n",
    "    print(\"\\nMake sure Neo4j is running: brew services start neo4j\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. System Resources Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import psutil\n",
    "import platform\n",
    "\n",
    "print(\"=== System Information ===\")\n",
    "print(f\"Platform: {platform.platform()}\")\n",
    "print(f\"Processor: {platform.processor()}\")\n",
    "print(f\"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical\")\n",
    "print(f\"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total\")\n",
    "print(f\"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB\")\n",
    "print(f\"Disk Usage: {psutil.disk_usage('/').percent}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\n",
    "\n",
    "If all cells above ran successfully, your environment is ready!\n",
    "\n",
    "Proceed to:\n",
    "- `02_load_first_model.ipynb` - Download and run your first local LLM\n",
    "- `03_vector_embeddings.ipynb` - Learn about embeddings and semantic search\n",
    "- `04_neo4j_basics.ipynb` - Create your first D&D world graph\n",
    "\n",
    "**Remember to:**\n",
    "1. Change Neo4j default password at http://localhost:7474\n",
    "2. Update `.env` file with your passwords\n",
    "3. Star Neo4j service when needed: `brew services start neo4j`"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create .gitkeep files for empty directories
touch $PROJECT_DIR/models/checkpoints/.gitkeep
touch $PROJECT_DIR/models/fine_tuned/.gitkeep
touch $PROJECT_DIR/data/raw/.gitkeep
touch $PROJECT_DIR/data/processed/.gitkeep
touch $PROJECT_DIR/tests/.gitkeep

# Make scripts executable
chmod +x $PROJECT_DIR/scripts/*.sh 2>/dev/null || true

log "Project structure created successfully!"

echo ""
echo "============================================================================"
echo "Project Structure Complete!"
echo "============================================================================"
echo ""
echo "Created at: $PROJECT_DIR"
echo ""
echo "Directory Structure:"
echo "  ├── src/              # Source code"
echo "  │   ├── core/         # Core orchestrator"
echo "  │   ├── llm/          # LLM engine"
echo "  │   ├── memory/       # Memory & RAG"
echo "  │   ├── rules/        # D&D rules engine"
echo "  │   ├── agents/       # NPC agents"
echo "  │   ├── world/        # World state"
echo "  │   └── api/          # API endpoints"
echo "  ├── data/             # Datasets and campaign data"
echo "  ├── models/           # Model storage"
echo "  ├── notebooks/        # Jupyter notebooks"
echo "  ├── configs/          # Configuration files"
echo "  ├── tests/            # Test files"
echo "  └── docs/             # Documentation"
echo ""
echo "Starter Files Created:"
echo "  ✓ README.md with architecture overview"
echo "  ✓ .gitignore for Python/ML projects"
echo "  ✓ .env.template for configuration"
echo "  ✓ requirements.txt"
echo "  ✓ Config files (YAML)"
echo "  ✓ Core orchestrator stub"
echo "  ✓ Getting started notebook"
echo ""
echo "Next Steps:"
echo "  1. cd $PROJECT_DIR"
echo "  2. cp .env.template .env"
echo "  3. Edit .env with your Neo4j password"
echo "  4. jupyter lab"
echo "  5. Open notebooks/01_getting_started.ipynb"
echo ""
warn "Initialize git repository:"
echo "  cd $PROJECT_DIR"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial project structure'"
echo ""
