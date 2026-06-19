"""尝试更多数据源获取黄金ETF历史数据"""

import requests
import json
import time

print("=== 尝试更多数据源 ===\n")

# 方法1: 新浪财经API
print("[1] 新浪财经API...")
try:
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {
        "symbol": "sh518880",
        "scale": "240",  # 日K
        "ma": "no",
        "datalen": "2000"
    }
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200:
        data = json.loads(r.text)
        print(f"  记录数: {len(data)}")
        if data:
            print(f"  首条: {data[0]}")
            print(f"  末条: {data[-1]}")
    else:
        print(f"  状态码: {r.status_code}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 方法2: 腾讯财经API
print("\n[2] 腾讯财经API...")
try:
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh518880,day,2014-01-01,2026-06-18,1000,qfq"
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200:
        data = json.loads(r.text)
        if "data" in data and "sh518880" in data["data"]:
            klines = data["data"]["sh518880"].get("qfqday", data["data"]["sh518880"].get("day", []))
            print(f"  记录数: {len(klines)}")
            if klines:
                print(f"  首条: {klines[0]}")
                print(f"  末条: {klines[-1]}")
        else:
            print(f"  数据结构: {list(data.keys())}")
    else:
        print(f"  状态码: {r.status_code}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 方法3: 163财经
print("\n[3] 网易财经API...")
try:
    url = "https://quotes.money.163.com/service/chddata.html"
    params = {
        "code": "1518880",  # 1表示上海
        "start": "20140101",
        "end": "20260618",
        "fields": "TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER"
    }
    r = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200:
        lines = r.text.strip().split('\n')
        print(f"  记录数: {len(lines) - 1}")  # 减去标题行
        if len(lines) > 1:
            print(f"  首条: {lines[1][:80]}")
            print(f"  末条: {lines[-1][:80]}")
    else:
        print(f"  状态码: {r.status_code}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

# 方法4: AKShare的另一个接口
print("\n[4] AKShare基金净值接口...")
try:
    import akshare as ak
    # 尝试获取基金净值数据
    df = ak.fund_etf_fund_info_em(fund="518880", start_date="20140101", end_date="20260618")
    print(f"  记录数: {len(df)}")
    if len(df) > 0:
        print(f"  列名: {df.columns.tolist()}")
except Exception as e:
    print(f"  失败: {type(e).__name__}: {e}")

print("\n=== 测试完成 ===")
