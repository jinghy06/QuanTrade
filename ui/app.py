"""Read-only v3 dashboard; all calculations live in the package."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from quantrade.config import load_config
from quantrade.data import validate_store
from quantrade.storage import Store

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args, _ = parser.parse_known_args(sys.argv[1:])
config = load_config(args.config)
store = Store(config["paths"]["database"])
st.set_page_config(page_title="QuanTrade 3.0 · ETF", layout="wide")
st.title("QuanTrade 3.0")
st.caption("ETF 中低频交易 · 统一账本与决策")
account_tab, targets_tab, runs_tab, data_tab, forecast_tab = st.tabs(
    ["账户", "调仓", "回测", "数据", "预测与决策"]
)
with store.connect() as connection:
    accounts = [
        r[0]
        for r in connection.execute(
            "SELECT artifact_id FROM artifacts WHERE kind='account' ORDER BY artifact_id"
        )
    ]
    reports = [
        (r[0], json.loads(r[1]))
        for r in connection.execute(
            "SELECT artifact_id,payload FROM artifacts WHERE kind='report' ORDER BY artifact_id DESC"
        )
    ]
    saved_forecasts = [
        json.loads(r[0])
        for r in connection.execute(
            "SELECT payload FROM artifacts WHERE kind='forecast' ORDER BY artifact_id DESC"
        )
    ]
with account_tab:
    if not accounts:
        st.info("尚未建立账户。先通过 account init 明确实际或模拟账户及初始现金。")
    else:
        name = st.selectbox("账户", accounts)
        account = store.get("account", name)
        st.caption(
            f"账户模式：{account['mode']} · 状态：{'完整' if account['complete'] else '待核对'}"
        )
        latest = next((report for key, report in reports if report["account"] == name), None)
        if latest:
            valuation = latest["valuation"]
            a, b, c = st.columns(3)
            a.metric("总资产", f"{valuation['nav']:,.2f}")
            b.metric("现金", f"{valuation['cash']:,.2f}")
            c.metric("累计费用", f"{valuation['fees']:,.2f}")
            st.caption(f"估值日期：{latest['as_of']}")
            st.dataframe(pd.DataFrame(valuation["positions"]).T, width="stretch")
        else:
            st.info("尚无估值报告；运行 report 更新指定日期状态。")
with targets_tab:
    if reports:
        key, report = reports[0]
        st.caption(f"报告：{key} · 建议订单尚未成交")
        target = report.get("target")
        if target:
            st.write("决策理由", target["reasons"])
            st.dataframe(
                pd.DataFrame(list(target["weights"].items()), columns=["ETF", "目标权重"]),
                width="stretch",
            )
        else:
            st.info("当前报告没有新的调仓目标。")
        st.dataframe(pd.DataFrame(report["orders"]), width="stretch")
    else:
        st.info("尚无调仓报告。")
with runs_tab:
    folders = sorted(
        Path(config["paths"]["runs"]).glob("v3-*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if folders:
        manifest_path = st.selectbox("固定实验", folders, format_func=lambda p: p.parent.name)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        st.caption(f"数据状态：{manifest['performance_status']}")
        if manifest["performance_status"] == "exploratory":
            st.warning("数据口径或规则尚未核验，此结果仅供探索。")
        st.dataframe(pd.DataFrame(manifest["metrics"]).T, width="stretch")
        curves = {}
        for mode in manifest["modes"]:
            nav = pd.read_csv(manifest_path.parent / mode / "nav.csv")
            if len(nav):
                curves[mode] = nav.set_index("date").nav
        if curves:
            st.line_chart(pd.DataFrame(curves))
        mode = st.selectbox("成交流水", manifest["modes"])
        st.dataframe(
            pd.DataFrame(
                json.loads((manifest_path.parent / mode / "fills.json").read_text(encoding="utf-8"))
            ),
            width="stretch",
        )
    else:
        st.info("尚无 v3 回测实验。")
with data_tab:
    validation = validate_store(store, config["data"]["jump_threshold"], config["data"])
    st.metric("行情行数", validation["rows"])
    st.metric("ETF 数量", validation["symbols"])
    st.caption(f"校验状态：{validation['validity']}")
    st.dataframe(pd.DataFrame(validation["issues"]), width="stretch")
    st.dataframe(pd.DataFrame([vars(i) for i in store.instruments().values()]), width="stretch")
with forecast_tab:
    if reports or saved_forecasts:
        report = reports[0][1] if reports else {}
        forecast = report.get("forecast", {})
        if not forecast.get("forecasts") and saved_forecasts:
            forecast = saved_forecasts[0]
        st.caption(
            f"预测时点：{forecast.get('as_of', '未指定')} · 模式：{forecast.get('mode', 'report')}"
        )
        if forecast.get("forecasts"):
            symbol = st.selectbox("ETF 预测", sorted(forecast["forecasts"]))
            item = forecast["forecasts"][symbol]
            candles = pd.DataFrame(item["candles"])
            # Plotly is optional: clear anchor/path bounds also work without it.
            try:
                import plotly.graph_objects as go

                fig = go.Figure(
                    go.Candlestick(
                        x=candles.date,
                        open=candles.open,
                        high=candles.high,
                        low=candles.low,
                        close=candles.close,
                        name="预测场景",
                    )
                )
                fig.add_scatter(x=candles.date, y=candles.upper, name="上界")
                fig.add_scatter(x=candles.date, y=candles.lower, name="下界")
                st.plotly_chart(fig, width="stretch")
            except ImportError:
                st.line_chart(candles.set_index("date")[["close", "lower", "upper"]])
            st.caption(item["note"])
        else:
            st.info(f"预测状态：{forecast.get('reason', 'unavailable')}")
        if report.get("target"):
            st.json(report["target"]["factors"])
    else:
        st.info("生成报告后查看模型预测与因子贡献。")
