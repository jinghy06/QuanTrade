#!/bin/bash
# ============================================================
# 树莓派 SMB 挂载 + 部署脚本
# 在树莓派上运行此脚本完成环境配置
# ============================================================

set -e

QUANT_DIR="/home/pi/quant_system"
SMB_HOST="${SMB_HOST:-192.168.1.100}"      # Windows IP，可通过环境变量覆盖
SMB_SHARE="${SMB_SHARE:-quant_sync}"      # 共享文件夹名
SMB_USER="${SMB_USER:-pi}"                # Windows用户名
SMB_PASS="${SMB_PASS:-}"                # Windows密码（建议通过环境变量传入）
MOUNT_POINT="/mnt/quant_sync"

echo "========================================"
echo "树莓派量化系统部署脚本"
echo "========================================"

# 1. 安装依赖
echo "[1/6] 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git cifs-utils

# 2. 创建目录
echo "[2/6] 创建项目目录..."
mkdir -p "$QUANT_DIR"
mkdir -p "$MOUNT_POINT"
mkdir -p "$QUANT_DIR/models"
mkdir -p "$QUANT_DIR/data"
mkdir -p "$QUANT_DIR/plots"
mkdir -p "$QUANT_DIR/logs"

# 3. 挂载SMB共享
echo "[3/6] 挂载SMB共享 //${SMB_HOST}/${SMB_SHARE} → ${MOUNT_POINT}..."

# 检查是否已挂载
if mountpoint -q "$MOUNT_POINT"; then
    echo "SMB已挂载，跳过"
else
    if [ -z "$SMB_PASS" ]; then
        echo "错误: 请设置 SMB_PASS 环境变量"
        echo "用法: SMB_PASS=your_password bash setup_pi.sh"
        exit 1
    fi
    
    sudo mount -t cifs "//${SMB_HOST}/${SMB_SHARE}" "$MOUNT_POINT" \
        -o username="$SMB_USER",password="$SMB_PASS",uid=pi,gid=pi,file_mode=0644,dir_mode=0755
    
    # 添加到 /etc/fstab 实现开机自动挂载
    FSTAB_ENTRY="//${SMB_HOST}/${SMB_SHARE} ${MOUNT_POINT} cifs username=${SMB_USER},password=${SMB_PASS},uid=pi,gid=pi,file_mode=0644,dir_mode=0755 0 0"
    if ! grep -q "$MOUNT_POINT" /etc/fstab; then
        echo "$FSTAB_ENTRY" | sudo tee -a /etc/fstab
        echo "已添加到 /etc/fstab 开机自动挂载"
    fi
fi

# 4. 创建符号链接（让 models/ 指向SMB挂载点）
echo "[4/6] 创建符号链接..."
if [ -d "$QUANT_DIR/models" ] && [ ! -L "$QUANT_DIR/models" ]; then
    mv "$QUANT_DIR/models" "$QUANT_DIR/models.local"
fi
ln -sf "$MOUNT_POINT/models" "$QUANT_DIR/models"

echo "模型目录已链接到SMB: $QUANT_DIR/models → $MOUNT_POINT/models"

# 5. 克隆/更新代码
echo "[5/6] 准备代码..."
if [ ! -d "$QUANT_DIR/.git" ]; then
    echo "请手动克隆代码仓库到 $QUANT_DIR"
    echo "git clone <your-repo-url> $QUANT_DIR"
else
    cd "$QUANT_DIR"
    git pull origin main || true
fi

# 6. 安装Python依赖
echo "[6/6] 安装Python依赖..."
cd "$QUANT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 配置 .env 文件: cp .env.example .env && nano .env"
echo "  2. 测试模型加载: python -c 'from models.ml_trainer import MLTrainer; MLTrainer().load_models()'"
echo "  3. 启动定时服务: sudo systemctl start quantbot"
echo ""
echo "SMB挂载状态:"
df -h "$MOUNT_POINT"
