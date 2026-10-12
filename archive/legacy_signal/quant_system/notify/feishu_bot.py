"""
A股量化信号系统 - 飞书通知
飞书自定义机器人Webhook推送
对应 plan.md Phase 3: "飞书推送测试"
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from config.settings import FEISHU_WEBHOOK

logger = logging.getLogger(__name__)


class FeishuBot:
    """
    飞书自定义机器人
    支持文本消息和交互式卡片消息
    """

    def __init__(self, webhook: str = None):
        self.webhook = webhook or FEISHU_WEBHOOK
        if not self.webhook:
            logger.warning("飞书Webhook未设置，通知将只记录不发送")

    def _send(self, payload: Dict) -> bool:
        """发送消息到飞书"""
        if not self.webhook:
            logger.info("[模拟发送] %s", payload.get("msg_type", "unknown"))
            return False

        try:
            resp = requests.post(
                self.webhook,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") == 0:
                logger.debug("飞书消息发送成功")
                return True
            else:
                logger.error("飞书API错误: %s", result)
                return False

        except requests.exceptions.RequestException as e:
            logger.error("飞书消息发送失败: %s", e)
            return False

    # ============================================================
    # 信号卡片消息（主用）
    # ============================================================

    def send_signal_card(self, signal: Dict[str, Any]) -> bool:
        """
        发送交易信号卡片
        对应 plan.md 6.5节输出格式的可视化呈现
        """
        symbol = signal.get("symbol", "")
        name = signal.get("name", symbol)
        signal_type = signal.get("signal", "观望")
        confidence = signal.get("confidence", 0)

        # 根据信号类型选择颜色
        color_map = {
            "加仓": "green",
            "轻仓试探": "blue",
            "观望": "grey",
            "减仓": "orange",
            "清仓": "red",
        }
        template = color_map.get(signal_type, "blue")

        # 构建卡片内容
        suggestion = signal.get("suggestion", {})
        ml_pred = signal.get("ml_prediction", {})
        tech = signal.get("technical", {})
        macro = signal.get("macro", {})

        elements = [
            # ML预测
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**ML预测**: 上涨概率 {ml_pred.get('up_prob', 0):.1%} | 信号强度: {signal_type}",
                },
            },
            # 技术面
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**技术面**: {tech.get('trend', 'N/A')} | "
                        f"RSI: {tech.get('rsi', 'N/A')} | "
                        f"MACD: {tech.get('macd', 'N/A')}"
                    ),
                },
            },
            # 宏观
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**宏观**: {macro.get('rate_env', 'N/A')} | "
                        f"情绪: {macro.get('market_sentiment', 'N/A')}"
                    ),
                },
            },
            {"tag": "hr"},
            # 操作建议
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**操作建议**: {suggestion.get('action', 'N/A')}",
                },
            },
            # 目标价和止损
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**目标价**: ¥{suggestion.get('target_price', 'N/A')} | "
                        f"**止损**: ¥{suggestion.get('stop_loss', 'N/A')} | "
                        f"**仓位**: {suggestion.get('position_pct', 0):.0%}"
                    ),
                },
            },
            {"tag": "hr"},
            # 决策理由
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**理由**: {suggestion.get('rationale', 'N/A')}",
                },
            },
        ]

        # 风险因素
        risk_factors = suggestion.get("risk_factors", [])
        if risk_factors:
            risk_text = "\n".join([f"- {r}" for r in risk_factors])
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**⚠️ 风险因素**:\n{risk_text}",
                },
            })

        # 持仓建议
        holding = signal.get("holding_advice")
        if holding:
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**持仓建议**: {holding}",
                },
            })

        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📊 {name} ({symbol}) - {signal_type}",
                    },
                    "template": template,
                },
                "elements": elements,
            },
        }

        return self._send(payload)

    # ============================================================
    # 其他消息类型
    # ============================================================

    def send_text(self, message: str) -> bool:
        """发送纯文本消息"""
        payload = {
            "msg_type": "text",
            "content": {"text": message},
        }
        return self._send(payload)

    def send_markdown(self, title: str, content: str) -> bool:
        """发送Markdown消息"""
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue",
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": content},
                    }
                ],
            },
        }
        return self._send(payload)

    def send_daily_summary(self, signals: List[Dict]) -> bool:
        """
        发送每日信号汇总
        对应 plan.md: "今日信号"命令
        """
        if not signals:
            return self.send_text("📋 今日无交易信号")

        # 构建汇总内容
        lines = [f"📋 今日信号汇总 ({datetime.now().strftime('%Y-%m-%d')})\n"]

        for sig in signals:
            symbol = sig.get("symbol", "")
            name = sig.get("name", symbol)
            signal_type = sig.get("signal", "")
            confidence = sig.get("confidence", 0)
            ml_prob = sig.get("ml_prediction", {}).get("up_prob", 0)

            emoji = {"加仓": "🟢", "轻仓试探": "🔵", "观望": "⚪", "减仓": "🟠", "清仓": "🔴"}.get(
                signal_type, "⚪"
            )
            lines.append(
                f"{emoji} **{name}** ({symbol}): {signal_type} | "
                f"ML: {ml_prob:.1%} | 置信度: {confidence:.2f}"
            )

        content = "\n".join(lines)
        return self.send_markdown("📋 每日信号汇总", content)

    def send_performance_report(self, stats: Dict) -> bool:
        """
        发送绩效报告
        对应 plan.md: "绩效"命令
        """
        content = (
            f"📈 信号绩效报告\n\n"
            f"**统计周期**: 近{stats.get('days', 30)}天\n"
            f"**总信号数**: {stats.get('total_signals', 0)}\n"
            f"**已执行**: {stats.get('executed', 0)}\n"
            f"**盈利信号**: {stats.get('profitable', 0)}\n"
            f"**亏损信号**: {stats.get('unprofitable', 0)}\n"
            f"**胜率**: {stats.get('win_rate', 0):.1%}\n"
            f"**平均盈亏**: ¥{stats.get('avg_pnl', 0):.2f}\n"
            f"**累计盈亏**: ¥{stats.get('total_pnl', 0):.2f}"
        )

        return self.send_markdown("📈 绩效报告", content)

    def send_error_alert(self, error_msg: str) -> bool:
        """发送系统错误告警"""
        content = (
            f"🚨 **系统异常**\n\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"错误: {error_msg}\n\n"
            f"请检查日志: `logs/quant.log`"
        )
        return self.send_markdown("🚨 系统异常", content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试发送（不配置webhook时仅打印）
    bot = FeishuBot()

    test_signal = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "000001.SZ",
        "name": "平安银行",
        "signal": "轻仓试探",
        "confidence": 0.62,
        "ml_prediction": {"up_prob": 0.58, "volatility": "中"},
        "technical": {"trend": "短期反弹", "rsi": 45, "macd": "金叉初期"},
        "macro": {"rate_env": "降息周期", "market_sentiment": "谨慎"},
        "suggestion": {
            "action": "MACD金叉，ML显示58%上涨概率，轻仓试探",
            "target_price": 12.50,
            "stop_loss": 11.80,
            "position_pct": 0.05,
            "rationale": "ML模型显示58%上涨概率，MACD刚形成金叉。但大盘情绪谨慎，建议不超过5%仓位。",
            "risk_factors": ["大盘情绪谨慎", "ML置信度仅58%未达强信号阈值"],
        },
        "holding_advice": None,
    }

    bot.send_signal_card(test_signal)
    print("测试信号已发送（或模拟发送）")
