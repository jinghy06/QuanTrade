#!/bin/bash
# ============================================================
# 树莓派模型同步守护脚本
# 持续监控SMB共享中的模型更新，有更新时自动拉取
# 建议配合 systemd 服务运行
# ============================================================

QUANT_DIR="/home/pi/quant_system"
MOUNT_POINT="/mnt/quant_sync"
INTERVAL=60  # 检测间隔秒数

LOG_FILE="$QUANT_DIR/logs/sync_daemon.log"
mkdir -p "$QUANT_DIR/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型同步守护进程启动" >> "$LOG_FILE"

while true; do
    # 检查SMB挂载
    if ! mountpoint -q "$MOUNT_POINT"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 警告: SMB未挂载，尝试重新挂载..." >> "$LOG_FILE"
        sudo mount -a || true
        sleep 10
        continue
    fi
    
    # 检测模型更新
    cd "$QUANT_DIR"
    source venv/bin/activate
    
    RESULT=$(python -c "
import sys
sys.path.insert(0, '.')
from sync.smb_sync import SMBSyncManager
sync = SMBSyncManager(local_root='.', remote_root='/mnt/quant_sync', remote_type='smb_mount')
result = sync.sync_all_pull()
print('MODELS:' + ','.join(result['models']))
print('DB:' + str(result['database']))
" 2>&1)
    
    UPDATED_MODELS=$(echo "$RESULT" | grep "^MODELS:" | cut -d: -f2)
    DB_UPDATED=$(echo "$RESULT" | grep "^DB:" | cut -d: -f2)
    
    if [ -n "$UPDATED_MODELS" ] && [ "$UPDATED_MODELS" != "," ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型已更新: $UPDATED_MODELS" >> "$LOG_FILE"
        
        # 可选: 发送飞书通知
        # python -c "from notify.feishu_bot import FeishuBot; FeishuBot().send_text('模型已更新: $UPDATED_MODELS')"
    fi
    
    if [ "$DB_UPDATED" = "True" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 数据库已更新" >> "$LOG_FILE"
    fi
    
    sleep $INTERVAL
done
