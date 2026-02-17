#!/bin/bash
# SSH tunnel to DungeonMaster Mac Studio
# Forwards Jupyter (8888) and Neo4j (7474)

echo "Creating SSH tunnel to DungeonMaster..."
echo "Jupyter: http://localhost:8888"
echo "Neo4j:   http://localhost:7474"
echo "Neo4j Bolt: http://localhost:7687"
echo ""
echo "Press Ctrl+C to close tunnel"

ssh -L 8888:localhost:8888 -L 7474:localhost:7474 -L 7687:localhost:7687 rlasker@dm.local
