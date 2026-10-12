# 量化程序邮件通知脚本

使用 `yagmail` + 126 邮箱 SMTP 发送量化程序日报或异常告警。

## 1. 安装依赖

```bash
pip install yagmail
```

## 2. 配置 126 邮箱

登录 126 邮箱 → 设置 → POP3/SMTP/IMAP → 开启 SMTP → 获取 **16 位授权码**。

## 3. 设置环境变量

复制示例文件并修改：

```bash
cp send_report_126.env.example .env
# 编辑 .env 填入你的邮箱和授权码
source .env
```

或者直接把变量写入 `~/.bashrc`：

```bash
echo 'export EMAIL_126="yourname@126.com"' >> ~/.bashrc
echo 'export EMAIL_126_CODE="你的授权码"' >> ~/.bashrc
echo 'export EMAIL_TO="收件邮箱@126.com"' >> ~/.bashrc
source ~/.bashrc
```

## 4. 使用方式

### 发送纯文本通知

```bash
python3 /root/send_report_126.py
```

### 发送带日志附件的日报

```bash
python3 /root/send_report_126.py \
  --subject "量化日报 2024-01-01" \
  --body "今日策略运行正常" \
  --attach /path/to/quant.log
```

### 在量化程序里调用（异常告警）

```python
import subprocess

try:
    run_strategy()
except Exception as e:
    subprocess.run([
        "python3", "/root/send_report_126.py",
        "--subject", "量化程序异常告警",
        "--body", f"策略运行出错：{e}",
    ])
    raise
```

## 5. 定时发送（crontab）

```bash
crontab -e
```

每天 20:00 发送：

```cron
0 20 * * * /usr/bin/python3 /root/send_report_126.py --attach /path/to/quant.log >> /root/send_report_126.log 2>&1
```

注意：crontab 默认不加载 `~/.bashrc` 环境变量，建议把环境变量写在 crontab 文件顶部，或使用 `.env` 文件 + `source` 包装脚本。

## 6. 安全提醒

- 不要把授权码硬编码在脚本里
- 不要把 `.env` 文件提交到 Git
- 日志文件发送前检查是否包含敏感信息（API Key、密码、持仓金额）
- 设置文件权限：`chmod 600 .env send_report_126.py`
