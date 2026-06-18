# A股量化信号系统 - 部署手册（SMB同步版）

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Windows + 树莓派 双机架构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────┐          ┌─────────────────────────┐         │
│   │    Windows (训练工厂)    │          │   树莓派 (信号哨兵)      │         │
│   │                         │          │                         │         │
│   │  ┌─────────────────┐   │  SMB     │  ┌─────────────────┐   │         │
│   │  │ 数据获取         │   │ 共享     │  │ 数据获取         │   │         │
│   │  │ 特征工程         │   │◄────────►│  │ 特征工程         │   │         │
│   │  │ 模型训练         │   │          │  │ 模型推理         │   │         │
│   │  │ 模型评估         │   │          │  │ 走势预测         │   │         │
│   │  │ 生成预测K线图     │   │          │  │ 生成预测K线图     │   │         │
│   │  └─────────────────┘   │          │  └─────────────────┘   │         │
│   │           │            │          │           │            │         │
│   │           ▼            │          │           ▼            │         │
│   │  ┌─────────────────┐   │          │  ┌─────────────────┐   │         │
│   │  │ models/         │   │          │  │ models/ ──SMB───┘   │         │
│   │  │ data/quant.db   │───┼──SMB共享──┼─►│ (符号链接)        │   │         │
│   │  │ plots/          │   │          │  │                   │   │         │
│   │  └─────────────────┘   │          │  └─────────────────┘   │         │
│   │           │            │          │           │            │         │
│   │           ▼            │          │           ▼            │         │
│   │  ┌─────────────────┐   │          │  ┌─────────────────┐   │         │
│   │  │ SMB共享文件夹    │◄─┘          │  │ LLM分析          │   │         │
│   │  │ //192.168.x.x/  │              │  │ 飞书推送         │   │         │
│   │  │   quant_sync    │              │  │ 定时任务         │   │         │
│   │  └─────────────────┘              │  └─────────────────┘   │         │
│   │                                     │                         │         │
│   │  职责: 训练+评估+同步                │  职责: 推理+预测+推送   │         │
│   │  频率: 每周一次（或模型衰减时）       │  频率: 每日2次（9:00/17:30）│      │
│   └─────────────────────────┘          └─────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 一、Windows端配置（训练工厂）

### 1.1 开启SMB共享

**方式A: PowerShell（推荐）**

以管理员身份运行PowerShell：

```powershell
# 创建共享文件夹
$sharePath = "C:\quant_sync"
New-Item -ItemType Directory -Path $sharePath -Force

# 创建子目录
New-Item -ItemType Directory -Path "$sharePath\models" -Force
New-Item -ItemType Directory -Path "$sharePath\data" -Force
New-Item -ItemType Directory -Path "$sharePath\plots" -Force

# 开启SMB共享（给指定用户读写权限）
$username = "$env:USERNAME"  # 或指定其他用户
New-SmbShare -Name "quant_sync" -Path $sharePath -FullAccess $username

# 查看共享状态
Get-SmbShare -Name "quant_sync"
```

**方式B: 图形界面**

1. 右键文件夹 → 属性 → 共享 → 高级共享
2. 勾选"共享此文件夹"
3. 权限 → 添加用户 → 勾选"完全控制"
4. 记下网络路径，如 `\\192.168.1.100\quant_sync`

### 1.2 防火墙设置

```powershell
# 确保SMB通过防火墙
netsh advfirewall firewall set rule group="文件和打印机共享" new enable=yes
```

### 1.3 训练+同步

```bash
# 进入项目目录
cd C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system

# 激活虚拟环境
.venv\Scripts\activate

# 训练并同步（手动）
python scripts\train_and_sync.py --smb "//192.168.1.100/quant_sync"

# 训练+超参调优+同步
python scripts\train_and_sync.py --smb "//192.168.1.100/quant_sync" --tune

# 仅同步（不训练）
python sync\smb_sync.py --mode push --remote "//192.168.1.100/quant_sync" --local "."
```

### 1.4 设置定时训练（Windows任务计划程序）

```powershell
# 创建每周日凌晨2点的训练任务
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts\train_and_sync.py --smb //192.168.1.100/quant_sync" -WorkingDirectory "C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "02:00"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd
Register-ScheduledTask -TaskName "QuantTrainAndSync" -Action $action -Trigger $trigger -Settings $settings
```

---

## 二、树莓派端配置（信号哨兵）

### 2.1 一键部署

```bash
# 设置环境变量后运行
export SMB_HOST="192.168.1.100"
export SMB_SHARE="quant_sync"
export SMB_USER="your_windows_username"
export SMB_PASS="your_windows_password"

# 运行部署脚本
bash scripts/setup_pi.sh
```

### 2.2 手动步骤（如果一键脚本失败）

```bash
# 1. 安装依赖
sudo apt update
sudo apt install -y python3-pip python3-venv git cifs-utils

# 2. 挂载SMB
sudo mkdir -p /mnt/quant_sync
sudo mount -t cifs //192.168.1.100/quant_sync /mnt/quant_sync \
    -o username=your_user,password=your_pass,uid=pi,gid=pi

# 3. 创建符号链接
ln -sf /mnt/quant_sync/models /home/pi/quant_system/models

# 4. 安装Python依赖
cd /home/pi/quant_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.3 配置systemd服务

```bash
# 复制服务文件
sudo cp systemd/quantbot.service /etc/systemd/system/
sudo cp systemd/quantbot-sync.service /etc/systemd/system/

# 编辑配置（填入你的API Key和SMB密码）
sudo nano /etc/systemd/system/quantbot.service

# 重载并启动
sudo systemctl daemon-reload
sudo systemctl enable quantbot quantbot-sync
sudo systemctl start quantbot quantbot-sync

# 查看状态
sudo systemctl status quantbot
sudo systemctl status quantbot-sync

# 查看日志
journalctl -u quantbot -f
journalctl -u quantbot-sync -f
```

### 2.4 树莓派目录结构

```
/home/pi/quant_system/
├── models -> /mnt/quant_sync/models    # SMB符号链接（模型从Windows同步）
├── data/
│   └── quant.db                        # 本地SQLite（树莓派自己维护）
├── plots/                              # 本地预测图
├── logs/
│   ├── quant.log                       # 主日志
│   └── sync_daemon.log                 # 同步守护进程日志
├── venv/                               # Python虚拟环境
└── ...
```

---

## 三、定时预测机制

### 3.1 树莓派定时任务

`signal_bot.py --schedule` 已内置定时：

```python
# config/settings.py
SCHEDULE_TIMES = ["09:00", "17:30"]  # 每日开盘前和收盘后
```

每次运行完整流水线：
1. **数据同步** → 增量拉取最新日K（AkShare）
2. **特征计算** → 计算最新技术指标
3. **ML多视野预测** → 加载SMB同步的模型，预测1/3/5/10日走势
4. **走势分类** → 判断"强势上涨"/"先抑后扬"等
5. **生成预测K线图** → 保存到本地 plots/
6. **LLM分析** → 传入走势预测做深度分析
7. **飞书推送** → 发送信号卡片（含预测图路径）

### 3.2 模型自动更新

`quantbot-sync.service` 持续监控：
- 每60秒检测SMB共享中的模型文件
- 发现Windows推送新模型 → 自动拉取到本地
- 无需重启服务，新模型即时生效

### 3.3 数据更新策略

| 数据类型 | 更新方式 | 负责端 |
|---------|---------|--------|
| 日K数据 | 树莓派本地增量拉取（AkShare） | 树莓派 |
| 特征数据 | 树莓派本地计算 | 树莓派 |
| ML模型 | SMB同步（Windows→树莓派） | Windows |
| 预测K线图 | 树莓派本地生成 | 树莓派 |

---

## 四、SMB同步命令速查

### Windows端（推送）

```bash
# 推送模型
python sync/smb_sync.py --mode push --remote "//192.168.1.100/quant_sync" --local "."

# 推送模型+数据库
python sync/smb_sync.py --mode push --remote "//192.168.1.100/quant_sync" --local "."

# 训练+推送（推荐）
python scripts/train_and_sync.py --smb "//192.168.1.100/quant_sync"
```

### 树莓派端（拉取）

```bash
# 手动拉取
python sync/smb_sync.py --mode pull --remote "/mnt/quant_sync" --local "." --type smb_mount

# 持续监控
python sync/smb_sync.py --mode watch --remote "/mnt/quant_sync" --local "." --type smb_mount --watch-interval 60
```

---

## 五、常见问题

### Q1: SMB挂载失败
```bash
# 检查网络连通性
ping 192.168.1.100

# 检查SMB共享是否存在
smbclient -L //192.168.1.100 -U your_user

# 手动挂载测试
sudo mount -t cifs //192.168.1.100/quant_sync /mnt/quant_sync -o username=your_user,password=your_pass
```

### Q2: 模型同步后未生效
- 检查 `models/` 符号链接是否正确: `ls -la models/`
- 检查模型文件时间戳: `ls -la models/*.txt`
- 重启服务: `sudo systemctl restart quantbot`

### Q3: 树莓派内存不足
```bash
# 查看内存使用
free -h

# 减少股票池（config/settings.py）
WATCHLIST = WATCHLIST[:20]  # 只监控20只

# 或增加swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # 修改CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Q4: Windows共享无法访问
- 确保Windows和树莓派在同一局域网
- 检查Windows防火墙是否放行SMB（端口445）
- 确保Windows用户有共享文件夹的完全控制权限

---

## 六、维护操作

### 每日（自动）
- 树莓派定时运行预测流水线
- 同步守护进程检测模型更新

### 每周（手动/定时）
- Windows端运行训练+同步
- 审查信号质量，记录准确率

### 每月
- 运行模型衰减评估
- 清理旧预测图（`find plots/ -mtime +30 -delete`）
- 备份数据库

### 紧急回滚
```bash
# 树莓派上恢复旧模型
sudo systemctl stop quantbot
cp models/backup/20240115_020000/lgb_regressor_*.txt models/
sudo systemctl start quantbot
```
