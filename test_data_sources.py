"""测试各种数据源获取黄金ETF数据"""

print("=== 测试数据源 ===\n")

# 1. baostock
print("[1] baostock测试...")
try:
    import baostock as bs
    lg = bs.login()
    print(f"  登录: {lg.error_msg}")

    # 尝试获取更多数据
    rs = bs.query_history_k_data_plus(
        "sh.518880",
        "date,code,open,high,low,close,volume,amount",
        start_date="2020-01-01",
        end_date="2026-06-18",
        frequency="d",
        adjustflag="2"
    )

    data = []
    while rs.next():
        data.append(rs.get_row_data())

    print(f"  日线记录数: {len(data)}")
    if data:
        print(f"  时间范围: {data[0][0]} ~ {data[-1][0]}")

    # 尝试周线
    rs_w = bs.query_history_k_data_plus(
        "sh.518880",
        "date,code,open,high,low,close,volume,amount",
        start_date="2014-01-01",
        end_date="2026-06-18",
        frequency="w",
        adjustflag="2"
    )

    data_w = []
    while rs_w.next():
        data_w.append(rs_w.get_row_data())

    print(f"  周线记录数: {len(data_w)}")
    if data_w:
        print(f"  时间范围: {data_w[0][0]} ~ {data_w[-1][0]}")

    bs.logout()
except Exception as e:
    print(f"  baostock失败: {e}")

# 2. efinance
print("\n[2] efinance测试...")
try:
    import efinance as ef
    df = ef.stock.get_quote_history("518880")
    print(f"  记录数: {len(df)}")
    print(f"  列名: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"  时间范围: {df['日期'].min()} ~ {df['日期'].max()}")
except Exception as e:
    print(f"  efinance失败: {e}")

# 3. 直接爬取东方财富
print("\n[3] 东方财富API测试...")
try:
    import requests
    import json

    # 东方财富K线API
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "1.518880",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",  # 日K
        "fqt": "1",    # 前复权
        "beg": "20140101",
        "end": "20260618",
        "lmt": "10000"
    }

    r = requests.get(url, params=params, timeout=10)
    data = json.loads(r.text)

    if data.get("data") and data["data"].get("klines"):
        klines = data["data"]["klines"]
        print(f"  记录数: {len(klines)}")
        if klines:
            print(f"  首条: {klines[0]}")
            print(f"  末条: {klines[-1]}")
    else:
        print(f"  返回数据: {str(data)[:200]}")
except Exception as e:
    print(f"  东方财富失败: {e}")

print("\n=== 测试完成 ===")
