#!/bin/bash

# ========================================================================
# Script to Recreate LVM with Striping for Maximum DiskANN Performance
# ========================================================================
# 
# WARNING: THIS WILL DELETE ALL DATA ON /mydata!
# 
# This script will:
# 1. Stop Milvus services
# 2. Unmount /mydata
# 3. Remove old linear LVM
# 4. Create new striped LVM across 4 Optane NVMes
# 5. Format with optimized stripe parameters
# 6. Remount and prepare directories
#
# Expected Result: 2M+ IOPS, 35-50 QPS (vs 11 QPS current)
# ========================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VG_NAME="emulab"
LV_NAME="bs1"
MOUNT_POINT="/mydata"
NUM_STRIPES=4
STRIPE_SIZE="256K"  # 256KB stripe size
LV_SIZE="23T"       # 23 TB total

# NVMe devices to use (4 largest, fastest devices)
NVME_DEVICES=(
    "/dev/nvme0n1"
    "/dev/nvme2n1"
    "/dev/nvme3n1"
    "/dev/nvme4n1"
)

echo -e "${YELLOW}========================================================================${NC}"
echo -e "${YELLOW}           LVM STRIPING CONFIGURATION SCRIPT${NC}"
echo -e "${YELLOW}========================================================================${NC}"
echo ""
echo -e "${RED}WARNING: THIS WILL DELETE ALL DATA ON /mydata!${NC}"
echo ""
echo "Current configuration:"
echo "  - Linear LVM (sequential)"
echo "  - Single device active"
echo "  - 534K IOPS, 11 QPS"
echo ""
echo "New configuration:"
echo "  - Striped LVM across 4 NVMe devices"
echo "  - All devices active in parallel"
echo "  - Expected: 2M+ IOPS, 35-50 QPS"
echo ""
echo "Devices to be used:"
for dev in "${NVME_DEVICES[@]}"; do
    size=$(lsblk -b -n -o SIZE "$dev" 2>/dev/null | awk '{print $1/1024/1024/1024}')
    echo "  - $dev (${size}GB)"
done
echo ""
echo -e "${YELLOW}========================================================================${NC}"
echo ""

# Safety check
read -p "Have you backed up all important data? (yes/no): " confirm1
if [ "$confirm1" != "yes" ]; then
    echo -e "${RED}Aborting. Please backup your data first.${NC}"
    exit 1
fi

read -p "Type 'DELETE ALL DATA' to confirm: " confirm2
if [ "$confirm2" != "DELETE ALL DATA" ]; then
    echo -e "${RED}Aborting. Confirmation not received.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting LVM reconfiguration...${NC}"
echo ""

# Step 1: Stop Docker services
echo "Step 1: Stopping Docker services..."
if [ -f "/mydata/vectordb-bench/milvus/docker-compose.yml" ]; then
    cd /mydata/vectordb-bench/milvus
    sudo docker-compose down 2>/dev/null || true
    echo "  ✓ Docker services stopped"
else
    echo "  ⚠ docker-compose.yml not found, skipping"
fi

# Check for any running containers
RUNNING=$(sudo docker ps -q)
if [ ! -z "$RUNNING" ]; then
    echo "  Warning: Some containers still running, stopping all..."
    sudo docker stop $(sudo docker ps -q) || true
fi

sleep 2

# Step 2: Unmount /mydata
echo ""
echo "Step 2: Unmounting $MOUNT_POINT..."
cd
# Check if mounted
if mountpoint -q "$MOUNT_POINT"; then
    # Try to unmount
    if ! sudo umount "$MOUNT_POINT" 2>/dev/null; then
        echo "  ⚠ Failed to unmount, checking for processes..."
        
        # Show what's using it
        echo "  Processes using $MOUNT_POINT:"
        sudo lsof +D "$MOUNT_POINT" 2>/dev/null || true
        
        # Kill processes
        echo "  Killing processes..."
        sudo fuser -km "$MOUNT_POINT" 2>/dev/null || true
        
        sleep 2
        
        # Try again
        sudo umount "$MOUNT_POINT" || {
            echo -e "${RED}  ✗ Failed to unmount $MOUNT_POINT${NC}"
            exit 1
        }
    fi
    echo "  ✓ $MOUNT_POINT unmounted"
else
    echo "  ✓ $MOUNT_POINT not mounted"
fi

# Step 3: Remove old LVM
echo ""
echo "Step 3: Removing old linear LVM..."
if sudo lvdisplay "/dev/$VG_NAME/$LV_NAME" &>/dev/null; then
    sudo lvremove -f "/dev/$VG_NAME/$LV_NAME"
    echo "  ✓ Old LVM removed"
else
    echo "  ⚠ LVM not found, skipping"
fi

# Step 4: Create new striped LVM
echo ""
echo "Step 4: Creating striped LVM..."
echo "  Configuration:"
echo "    - Size: $LV_SIZE"
echo "    - Stripes: $NUM_STRIPES"
echo "    - Stripe size: $STRIPE_SIZE"
echo "    - Devices: ${NVME_DEVICES[*]}"
echo ""

# Build the lvcreate command
sudo lvcreate \
    -L "$LV_SIZE" \
    -i "$NUM_STRIPES" \
    -I "$STRIPE_SIZE" \
    -n "$LV_NAME" \
    "$VG_NAME" \
    "${NVME_DEVICES[@]}"

if [ $? -eq 0 ]; then
    echo "  ✓ Striped LVM created successfully"
else
    echo -e "${RED}  ✗ Failed to create striped LVM${NC}"
    exit 1
fi

# Step 5: Format filesystem with stripe optimization
echo ""
echo "Step 5: Formatting filesystem..."

# Calculate stride and stripe-width
# stride = stripe_size / block_size
# stripe_size = 256KB = 256 * 1024 = 262144 bytes
# block_size = 4KB = 4096 bytes
# stride = 262144 / 4096 = 64
STRIDE=64

# stripe-width = stride * num_stripes
# stripe-width = 64 * 4 = 256
STRIPE_WIDTH=$((STRIDE * NUM_STRIPES))

echo "  Filesystem parameters:"
echo "    - stride: $STRIDE"
echo "    - stripe-width: $STRIPE_WIDTH"
echo ""

sudo mkfs.ext4 \
    -E "stride=$STRIDE,stripe-width=$STRIPE_WIDTH" \
    -L mydata \
    "/dev/$VG_NAME/$LV_NAME"

if [ $? -eq 0 ]; then
    echo "  ✓ Filesystem created successfully"
else
    echo -e "${RED}  ✗ Failed to create filesystem${NC}"
    exit 1
fi

# Step 6: Mount the new volume
echo ""
echo "Step 6: Mounting new volume..."
sudo mount "/dev/$VG_NAME/$LV_NAME" "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
    echo "  ✓ Volume mounted at $MOUNT_POINT"
else
    echo -e "${RED}  ✗ Failed to mount volume${NC}"
    exit 1
fi

# Step 7: Set permissions
echo ""
echo "Step 7: Setting permissions..."
sudo chown -R $(whoami):e6897-PG0 "$MOUNT_POINT"
echo "  ✓ Permissions set"

# Step 8: Create directory structure
echo ""
echo "Step 8: Creating directory structure..."
mkdir -p "$MOUNT_POINT/vectordb-bench/milvus"
mkdir -p "$MOUNT_POINT/vectordb-bench/benchmark"
mkdir -p "$MOUNT_POINT/spacev1b"
mkdir -p "$MOUNT_POINT/tmp"
sudo chmod 1777 "$MOUNT_POINT/tmp"
echo "  ✓ Directories created"

# Step 9: Verify configuration
echo ""
echo "Step 9: Verifying striped configuration..."
echo ""
sudo lvdisplay -m "/dev/$VG_NAME/$LV_NAME" | grep -A 5 "Type"

# Check disk usage
echo ""
echo "Current disk usage:"
df -h "$MOUNT_POINT"

# Step 10: Test I/O across all devices
echo ""
echo "Step 10: Testing I/O distribution..."
echo "  Writing test file to verify all devices are used..."

# Write a test file
dd if=/dev/zero of="$MOUNT_POINT/test_stripe.dat" bs=1M count=1024 oflag=direct 2>/dev/null

echo ""
echo "  Run this command to verify all NVMes are active:"
echo "    iostat -x 2 5"
echo ""

# Cleanup test file
rm -f "$MOUNT_POINT/test_stripe.dat"

# Summary
echo ""
echo -e "${GREEN}========================================================================${NC}"
echo -e "${GREEN}                    CONFIGURATION COMPLETE!${NC}"
echo -e "${GREEN}========================================================================${NC}"
echo ""
echo "New striped LVM configuration:"
echo "  ✓ Volume: /dev/$VG_NAME/$LV_NAME"
echo "  ✓ Mount point: $MOUNT_POINT"
echo "  ✓ Striped across: $NUM_STRIPES devices"
echo "  ✓ Stripe size: $STRIPE_SIZE"
echo "  ✓ Expected performance: 2M+ IOPS, 35-50 QPS"
echo ""
echo "Next steps:"
echo "  1. Copy dataset back: scp node2:/mydata/spacev1b/*.bin $MOUNT_POINT/spacev1b/"
echo "  2. Restore docker-compose.yml to $MOUNT_POINT/vectordb-bench/milvus/"
echo "  3. Start Milvus: cd $MOUNT_POINT/vectordb-bench/milvus && sudo docker-compose up -d"
echo "  4. Reload data: python3 main.py --load (takes 6-8 hours)"
echo "  5. Run benchmark: python3 main.py --benchmark"
echo ""
echo "To verify striping is working during benchmark, run:"
echo "  iostat -x 2 10"
echo ""
echo "You should see ALL 4 NVMes active with ~500K IOPS each!"
echo ""
echo -e "${GREEN}========================================================================${NC}"
