#!/bin/bash
# Script to copy NiceGUI app files to Docker container on remote machine via SSH

set -e

echo "========================================"
echo "NiceGUI Remote Container Setup Script"
echo "========================================"
echo ""

# Remote machine details
REMOTE_HOST="192.168.178.140"
REMOTE_USER="tgrytnes"
REMOTE_SSH="$REMOTE_USER@$REMOTE_HOST"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test SSH connection
echo "Testing SSH connection to $REMOTE_SSH..."
if ! ssh -o ConnectTimeout=5 "$REMOTE_SSH" "echo 'SSH connection successful'" 2>/dev/null; then
    echo -e "${RED}Error: Cannot connect to $REMOTE_SSH${NC}"
    echo "Please check:"
    echo "  1. The remote machine is on and accessible"
    echo "  2. SSH key is set up (try: ssh $REMOTE_SSH)"
    echo "  3. The IP address is correct: $REMOTE_HOST"
    exit 1
fi
echo -e "${GREEN}✓ SSH connection successful${NC}"
echo ""

# List containers on remote machine
echo "Listing Docker containers on remote machine..."
echo ""
ssh "$REMOTE_SSH" "docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}'" || {
    echo -e "${RED}Error: Could not list Docker containers${NC}"
    echo "Is Docker running on the remote machine?"
    exit 1
}
echo ""

# Ask user to specify container
read -p "Enter the container NAME or ID from the list above: " CONTAINER_NAME

if [ -z "$CONTAINER_NAME" ]; then
    echo -e "${RED}Error: No container name provided${NC}"
    exit 1
fi

# Verify container exists on remote
if ! ssh "$REMOTE_SSH" "docker ps | grep -q '$CONTAINER_NAME'"; then
    echo -e "${RED}Error: Container '$CONTAINER_NAME' not found or not running${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Found container: $CONTAINER_NAME${NC}"
echo ""

# Create temporary directory on remote machine
REMOTE_TEMP="/tmp/spatial-vit-setup-$$"
echo "Creating temporary directory on remote machine..."
ssh "$REMOTE_SSH" "mkdir -p $REMOTE_TEMP/eurosat_vit_analysis"
echo -e "${GREEN}✓ Temporary directory created${NC}"
echo ""

# Copy files to remote machine
echo "Copying files to remote machine..."
scp nicegui_app.py "$REMOTE_SSH:$REMOTE_TEMP/"
scp eurosat_vit_analysis/inference.py "$REMOTE_SSH:$REMOTE_TEMP/eurosat_vit_analysis/"
scp eurosat_vit_analysis/__init__.py "$REMOTE_SSH:$REMOTE_TEMP/eurosat_vit_analysis/"
scp eurosat_vit_analysis/models.py "$REMOTE_SSH:$REMOTE_TEMP/eurosat_vit_analysis/"
echo -e "${GREEN}✓ Files copied to remote machine${NC}"
echo ""

# Create app directory in container
echo "Creating /app directory in container..."
ssh "$REMOTE_SSH" "docker exec $CONTAINER_NAME mkdir -p /app/eurosat_vit_analysis" || true
echo -e "${GREEN}✓ Directory created in container${NC}"
echo ""

# Copy files from remote machine to container
echo "Copying files from remote machine to container..."
ssh "$REMOTE_SSH" "docker cp $REMOTE_TEMP/nicegui_app.py $CONTAINER_NAME:/app/"
ssh "$REMOTE_SSH" "docker cp $REMOTE_TEMP/eurosat_vit_analysis/inference.py $CONTAINER_NAME:/app/eurosat_vit_analysis/"
ssh "$REMOTE_SSH" "docker cp $REMOTE_TEMP/eurosat_vit_analysis/__init__.py $CONTAINER_NAME:/app/eurosat_vit_analysis/"
ssh "$REMOTE_SSH" "docker cp $REMOTE_TEMP/eurosat_vit_analysis/models.py $CONTAINER_NAME:/app/eurosat_vit_analysis/"
echo -e "${GREEN}✓ Files copied to container${NC}"
echo ""

# Clean up temporary directory
echo "Cleaning up temporary files..."
ssh "$REMOTE_SSH" "rm -rf $REMOTE_TEMP"
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Install dependencies
echo "Installing Python dependencies in container..."
ssh "$REMOTE_SSH" "docker exec $CONTAINER_NAME pip install --quiet httpx pillow matplotlib numpy" || {
    echo -e "${YELLOW}Warning: Some dependencies may already be installed${NC}"
}
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check if app is already running
echo "Checking if nicegui_app.py is already running..."
if ssh "$REMOTE_SSH" "docker exec $CONTAINER_NAME pgrep -f 'nicegui_app.py'" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ nicegui_app.py is already running${NC}"
    read -p "Do you want to restart it? (y/n): " RESTART
    if [ "$RESTART" = "y" ]; then
        echo "Stopping existing process..."
        ssh "$REMOTE_SSH" "docker exec $CONTAINER_NAME pkill -f 'nicegui_app.py'" || true
        sleep 2
        echo -e "${GREEN}✓ Process stopped${NC}"
    fi
fi

echo ""
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Files successfully copied to container on $REMOTE_HOST"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the app on the remote container:"
echo "   ${YELLOW}ssh $REMOTE_SSH 'docker exec -d $CONTAINER_NAME python /app/nicegui_app.py'${NC}"
echo ""
echo "2. Or start it interactively to see output:"
echo "   ${YELLOW}ssh $REMOTE_SSH 'docker exec -it $CONTAINER_NAME python /app/nicegui_app.py'${NC}"
echo ""
echo "3. Access the dashboard:"
echo "   Open your browser to: ${YELLOW}http://$REMOTE_HOST:8080${NC}"
echo "   (Or whatever port your container exposes)"
echo ""
echo "4. Configure RunPod API URL in the dashboard sidebar"
echo ""
echo "5. View logs (if needed):"
echo "   ${YELLOW}ssh $REMOTE_SSH 'docker logs -f $CONTAINER_NAME'${NC}"
echo ""
echo "========================================"
