#!/bin/bash
# Script to copy NiceGUI app files to your existing Docker container

set -e

echo "========================================"
echo "NiceGUI Container Setup Script"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# List all running containers
echo "Available Docker containers:"
echo ""
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}" || {
    echo -e "${RED}Error: Could not list Docker containers${NC}"
    echo "Is Docker Desktop running?"
    exit 1
}
echo ""

# Ask user to specify container
read -p "Enter the container NAME or ID from the list above: " CONTAINER_NAME

if [ -z "$CONTAINER_NAME" ]; then
    echo -e "${RED}Error: No container name provided${NC}"
    exit 1
fi

# Verify container exists and is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${RED}Error: Container '$CONTAINER_NAME' not found or not running${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Found container: $CONTAINER_NAME${NC}"
echo ""

# Create app directory in container
echo "Creating /app directory in container..."
docker exec "$CONTAINER_NAME" mkdir -p /app/eurosat_vit_analysis || true
echo -e "${GREEN}✓ Directory created${NC}"
echo ""

# Copy main app file
echo "Copying nicegui_app.py..."
docker cp nicegui_app.py "$CONTAINER_NAME:/app/"
echo -e "${GREEN}✓ nicegui_app.py copied${NC}"

# Copy inference module
echo "Copying eurosat_vit_analysis module..."
docker cp eurosat_vit_analysis/inference.py "$CONTAINER_NAME:/app/eurosat_vit_analysis/"
docker cp eurosat_vit_analysis/__init__.py "$CONTAINER_NAME:/app/eurosat_vit_analysis/"
docker cp eurosat_vit_analysis/models.py "$CONTAINER_NAME:/app/eurosat_vit_analysis/"
echo -e "${GREEN}✓ Module files copied${NC}"
echo ""

# Install dependencies
echo "Installing Python dependencies in container..."
docker exec "$CONTAINER_NAME" pip install --quiet httpx pillow matplotlib numpy || {
    echo -e "${YELLOW}Warning: Some dependencies may already be installed${NC}"
}
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check if app is already running
echo "Checking if nicegui_app.py is already running..."
if docker exec "$CONTAINER_NAME" pgrep -f "nicegui_app.py" > /dev/null; then
    echo -e "${YELLOW}⚠ nicegui_app.py is already running${NC}"
    read -p "Do you want to restart it? (y/n): " RESTART
    if [ "$RESTART" = "y" ]; then
        echo "Stopping existing process..."
        docker exec "$CONTAINER_NAME" pkill -f "nicegui_app.py" || true
        sleep 2
        echo -e "${GREEN}✓ Process stopped${NC}"
    fi
fi

echo ""
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set your RunPod API URL:"
echo "   Edit line 16 in nicegui_app.py OR configure it in the UI"
echo ""
echo "2. Start the app in your container:"
echo "   ${YELLOW}docker exec -d $CONTAINER_NAME python /app/nicegui_app.py${NC}"
echo ""
echo "3. Access the dashboard:"
echo "   Open your browser to the container's exposed port"
echo "   (Usually http://localhost:8080 or the port you configured)"
echo ""
echo "4. View logs (if needed):"
echo "   ${YELLOW}docker exec $CONTAINER_NAME tail -f /app/nicegui_app.log${NC}"
echo ""
echo "========================================"
