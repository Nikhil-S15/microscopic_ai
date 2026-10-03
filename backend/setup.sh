#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Bacteria Detection Backend...${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt

# Install PyTorch based on platform
echo -e "${YELLOW}Installing PyTorch...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    pip install torch torchvision
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # Windows or others
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << 'ENVEOF'
# Backend Configuration
PORT=8000
HOST=0.0.0.0
MODEL_PATH=best_model_fixed.pth
LOG_LEVEL=INFO

# CORS Configuration
ALLOWED_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]

# Processing Configuration
PATCH_SIZE=48
CONFIDENCE_THRESHOLD=0.5
TOP_K=3
ENVEOF
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To start the backend:${NC}"
echo -e "1. Activate virtual environment: ${GREEN}source venv/bin/activate${NC}"
echo -e "2. Run the server: ${GREEN}python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000${NC}"
echo -e "3. Open API docs: ${GREEN}http://localhost:8000/docs${NC}"
