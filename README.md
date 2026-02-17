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

