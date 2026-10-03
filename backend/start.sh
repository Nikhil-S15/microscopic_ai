#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Bacteria Detection Backend...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup...${NC}"
    ./setup.sh
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if requirements are installed
if ! pip list | grep -q fastapi; then
    echo -e "${YELLOW}Requirements not installed. Installing...${NC}"
    pip install -r requirements.txt
    pip install torch torchvision
fi

# Run the server
echo -e "${GREEN}Starting server on http://localhost:8000${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
