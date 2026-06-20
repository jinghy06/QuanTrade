#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuanTrade 每日自动化运行脚本
============================
每个交易日前执行：
  1. 增量更新 ETF 数据
  2. 生成交易信号报告
  3. 发送邮件通知

用法:
    cd /root/QuanTrade
    python3 daily_run.py

crontab 示例（工作日 08:30 运行）:
    30 8 * * 1-5 cd /root/QuanTrade && source /root/mail_env.sh && python3 daily_run.py >> logs/daily_run.log 2>&1
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
REPORT_DIR = PROJECT_DIR / "reports"

# 邮件发送脚本路径
MAIL_SCRIPT = "send_report_126.py"
ENV_FILE = "/root/mail_env.sh"


def log(msg: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_data_update() -> bool:
    """运行每日数据更新"""
    log("开始每日数据更新...")
    try:
        import daily_update
        daily_update.main()
        log("数据更新完成")
        return True
    except Exception as e:
        log(f"数据更新失败: {e}")
        return False


def run_signal_report() -> str:
    """生成交易信号报告，返回报告路径"""
    log("开始生成交易信号报告...")
    try:
        import daily_signal_report
        report_path = daily_signal_report.generate_report()
        log(f"信号报告生成完成: {report_path}")
        return report_path
    except Exception as e:
        log(f"信号报告生成失败: {e}")
        raise


def send_email_report(report_path: str) -> bool:
    """发送邮件报告"""
    log("开始发送邮件报告...")

    if not os.path.exists(MAIL_SCRIPT):
        log(f"邮件脚本不存在: {MAIL_SCRIPT}")
        return False

    today = datetime.now().strftime('%Y-%m-%d')
    subject = f"量化交易日报 - {today}"
    body = f"今日 QuanTrade 量化交易信号报告已生成，请查看附件。\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    cmd = [
        "python3", MAIL_SCRIPT,
        "--subject", subject,
        "--body", body,
        "--attach", report_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            log("邮件发送成功")
            return True
        else:
            log(f"邮件发送失败: {result.stdout}\n{result.stderr}")
            return False
    except Exception as e:
        log(f"邮件发送异常: {e}")
        return False


def main():
    ensure_dirs()

    log("=" * 70)
    log("QuanTrade 每日自动化运行开始")
    log("=" * 70)

    # 1. 更新数据
    update_ok = run_data_update()
    if not update_ok:
        log("数据更新失败，继续生成报告（使用已有数据）")

    # 2. 生成信号报告
    try:
        report_path = run_signal_report()
    except Exception as e:
        log(f"报告生成失败，退出: {e}")
        sys.exit(1)

    # 3. 发送邮件
    if os.path.exists(ENV_FILE):
        send_email_report(report_path)
    else:
        log(f"环境变量文件不存在: {ENV_FILE}，跳过邮件发送")

    log("=" * 70)
    log("QuanTrade 每日自动化运行完成")
    log("=" * 70)


if __name__ == "__main__":
    main()
