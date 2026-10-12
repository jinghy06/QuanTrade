#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化程序邮件通知脚本
使用 yagmail + 126 邮箱 SMTP 发送日报/告警邮件

依赖:
    pip install yagmail

环境变量（必须）:
    EMAIL_126        发件邮箱，例如: yourname@126.com
    EMAIL_126_CODE   126 邮箱 SMTP 授权码（不是登录密码）
    EMAIL_TO         收件邮箱，多个用逗号分隔

用法:
    # 发送纯文本通知
    python3 send_report_126.py

    # 发送带附件的日志
    python3 send_report_126.py --subject "量化日报" --body "今日运行正常" --attach /path/to/quant.log

    # 仅发送告警
    python3 send_report_126.py --subject "量化程序异常" --body "策略运行出错，请检查"
"""

import os
import sys
import argparse
import smtplib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yagmail


# 126 邮箱 SMTP 配置
SMTP_HOST = "smtp.126.com"
SMTP_PORT = 465  # SSL 端口


def get_env_or_die(name: str) -> str:
    """读取环境变量，缺失则退出"""
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"[错误] 环境变量 {name} 未设置")
        print("请先在 ~/.bashrc 中设置：")
        print(f"    export {name}=\"你的值\"")
        sys.exit(1)
    return value


def load_config():
    """加载邮件配置"""
    sender = get_env_or_die("EMAIL_126")
    auth_code = get_env_or_die("EMAIL_126_CODE")
    receivers = get_env_or_die("EMAIL_TO")
    receiver_list = [r.strip() for r in receivers.split(",") if r.strip()]
    return sender, auth_code, receiver_list


def send_email(
    subject: str,
    body: str,
    attachments: Optional[List[str]] = None,
):
    """发送邮件"""
    sender, auth_code, receivers = load_config()

    # 过滤掉不存在的附件
    valid_attachments = []
    if attachments:
        for path in attachments:
            p = Path(path)
            if p.exists() and p.is_file():
                valid_attachments.append(str(p))
            else:
                print(f"[警告] 附件不存在，已跳过: {path}")

    try:
        print(f"[{datetime.now()}] 正在连接 126 SMTP...")
        yag = yagmail.SMTP(user=sender, password=auth_code, host=SMTP_HOST)

        print(f"[{datetime.now()}] 正在发送邮件到: {', '.join(receivers)}")
        yag.send(
            to=receivers,
            subject=subject,
            contents=body,
            attachments=valid_attachments,
        )
        print(f"[{datetime.now()}] 邮件发送成功")

    except smtplib.SMTPAuthenticationError:
        print("[错误] 126 邮箱认证失败，请检查：")
        print("  1. EMAIL_126 是否为正确的 126 邮箱地址")
        print("  2. EMAIL_126_CODE 是否为 16 位 SMTP 授权码（不是登录密码）")
        print("  3. 126 邮箱设置中是否已开启 SMTP/IMAP 服务")
        sys.exit(1)
    except smtplib.SMTPConnectError as e:
        print(f"[错误] 无法连接到 126 SMTP 服务器: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 邮件发送失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="发送量化程序邮件通知")
    parser.add_argument(
        "--subject",
        default=f"量化程序日报 - {datetime.now().strftime('%Y-%m-%d')}",
        help="邮件主题",
    )
    parser.add_argument(
        "--body",
        default=f"量化程序运行日报\n\n时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n状态：正常\n",
        help="邮件正文",
    )
    parser.add_argument(
        "--attach",
        nargs="+",
        default=None,
        help="附件路径，可指定多个",
    )
    args = parser.parse_args()

    send_email(
        subject=args.subject,
        body=args.body,
        attachments=args.attach,
    )


if __name__ == "__main__":
    main()
