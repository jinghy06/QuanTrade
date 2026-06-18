# -*- coding: utf-8 -*-
"""
新闻情绪爬虫 v4
================
为量化模型提供新闻情绪分析数据

数据源:
  1. 新浪财经个股新闻 (主要)
  2. 东方财富公告 (辅助)
  3. 财联社电报 (尝试, 失败则跳过)
  4. 新浪财经滚动新闻 (通用, 关键词筛选)

作者: 量化系统数据工程
"""

import json
import logging
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# ---------- 路径配置 ----------
SCRIPT_DIR = Path(__file__).resolve().parent
QUANT_SYSTEM_DIR = SCRIPT_DIR.parent
DATA_DIR = QUANT_SYSTEM_DIR / "data"
DB_PATH = DATA_DIR / "quant.db"

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("news_crawler_v4")

# ---------- 标的-关键词映射 + 相关关键词（用于通用新闻筛选） ----------
SYMBOL_KEYWORDS: Dict[str, Tuple[str, List[str]]] = {
    "562500.SH": ("中证A500", ["A500", "中证500", "沪深300", "大盘", "宽基", "沪指", "深成指", "A股", "指数"]),
    "588200.SH": ("科创芯片", ["芯片", "半导体", "科创"]),
    "588790.SH": ("科创AI", ["AI", "人工智能", "科创"]),
    "159382.SZ": ("创业板人工智能", ["人工智能", "AI", "创业板"]),
    "159241.SZ": ("创业板新能源", ["新能源", "光伏", "储能", "创业板"]),
    "000002.SZ": ("万科A", ["万科", "房地产", "地产"]),
    "000333.SZ": ("美的集团", ["美的", "家电", "智能家居"]),
    "000568.SZ": ("泸州老窖", ["泸州老窖", "白酒", "酒业"]),
    "000651.SZ": ("格力电器", ["格力", "家电", "空调"]),
}

# ---------- 情绪词典 ----------
POSITIVE_WORDS = [
    "涨停", "大涨", "反弹", "利好", "突破", "创新高", "强劲", "复苏",
    "增持", "买入", "推荐", "超预期", "盈利增长", "景气", "繁荣",
    "上涨", "攀升", "飙升", "回暖", "改善", "优化", "扩张", "增长",
    "净利润增长", "营收增长", "订单饱满", "供不应求", "龙头", "领先",
]

NEGATIVE_WORDS = [
    "跌停", "大跌", "暴跌", "利空", "跌破", "创新低", "疲软", "衰退",
    "减持", "卖出", "回避", "低于预期", "亏损", "下滑", "萎缩",
    "风险", "警示", "下跌", "下挫", "回落", "恶化", "收缩", "下降",
    "净利润下滑", "营收下降", "订单减少", "产能过剩", "落后", "承压",
    "债务违约", "资金链断裂", "裁员", "停产", "亏损扩大", "业绩暴雷",
]

MAJOR_EVENT_WORDS = [
    "政策", "监管", "制裁", "重组", "并购", "退市", "停牌",
    "财报", "业绩", "分红", "配股", "定增", "减持", "增持",
    "股东大会", "董事会", "关联交易", "担保", "诉讼", "仲裁",
    "立案调查", "行政处罚", "问询函", "关注函", "警示函",
]

# ---------- HTTP请求配置 ----------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 1.0


# ============================================================
# 数据库操作
# ============================================================

@contextmanager
def db_connection(db_path: str = None):
    """SQLite 上下文管理器"""
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_news_tables(db_path: str = None):
    """初始化新闻相关表结构（幂等）"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 原始新闻表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_raw_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                sentiment_score REAL DEFAULT 0,
                is_major_event INTEGER DEFAULT 0,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 情绪聚合表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                sentiment_1d REAL DEFAULT 0,
                sentiment_3d REAL DEFAULT 0,
                sentiment_7d REAL DEFAULT 0,
                major_events_1d INTEGER DEFAULT 0,
                major_events_3d INTEGER DEFAULT 0,
                news_count_1d INTEGER DEFAULT 0,
                news_count_3d INTEGER DEFAULT 0,
                latest_title TEXT,
                latest_source TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)

        # 索引优化
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_raw_symbol_date ON news_raw_v4(symbol, date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_raw_source ON news_raw_v4(source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_date ON news_sentiment_v4(symbol, date)"
        )

        conn.commit()
        logger.info("新闻表初始化完成: %s", db_path)


def get_existing_news_dates(symbol: str, db_path: str = None) -> List[str]:
    """获取某标的已有新闻的日期列表"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT date FROM news_raw_v4 WHERE symbol = ? ORDER BY date DESC",
            (symbol,),
        )
        return [row[0] for row in cursor.fetchall()]


def save_raw_news(news_items: List[Dict], db_path: str = None):
    """批量保存原始新闻（INSERT OR IGNORE，避免重复）"""
    if not news_items:
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        inserted = 0
        for item in news_items:
            # 使用 symbol + source + title + date 作为去重键
            cursor.execute(
                """
                SELECT id FROM news_raw_v4
                WHERE symbol = ? AND source = ? AND title = ? AND date = ?
                LIMIT 1
            """,
                (item["symbol"], item["source"], item["title"], item["date"]),
            )
            if cursor.fetchone():
                continue

            cursor.execute(
                """
                INSERT INTO news_raw_v4
                (symbol, date, source, title, summary, sentiment_score, is_major_event, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["symbol"],
                    item["date"],
                    item["source"],
                    item["title"],
                    item.get("summary", ""),
                    item.get("sentiment_score", 0.0),
                    1 if item.get("is_major_event") else 0,
                    item.get("url", ""),
                ),
            )
            inserted += 1

    logger.info("保存 %d 条原始新闻 (新增 %d 条)", len(news_items), inserted)


def save_sentiment_aggregates(aggregates: List[Dict], db_path: str = None):
    """保存情绪聚合数据（INSERT OR REPLACE）"""
    if not aggregates:
        return

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        for agg in aggregates:
            cursor.execute(
                """
                INSERT OR REPLACE INTO news_sentiment_v4
                (symbol, date, sentiment_1d, sentiment_3d, sentiment_7d,
                 major_events_1d, major_events_3d, news_count_1d, news_count_3d,
                 latest_title, latest_source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    agg["symbol"],
                    agg["date"],
                    agg.get("sentiment_1d", 0.0),
                    agg.get("sentiment_3d", 0.0),
                    agg.get("sentiment_7d", 0.0),
                    agg.get("major_events_1d", 0),
                    agg.get("major_events_3d", 0),
                    agg.get("news_count_1d", 0),
                    agg.get("news_count_3d", 0),
                    agg.get("latest_title", ""),
                    agg.get("latest_source", ""),
                    datetime.now().isoformat(),
                ),
            )

    logger.info("保存 %d 条情绪聚合数据", len(aggregates))


# ============================================================
# 情绪分析
# ============================================================

def analyze_sentiment(title: str, summary: str = "") -> Tuple[float, bool]:
    """
    对单条新闻进行情绪打分

    Returns:
        (sentiment_score, is_major_event)
    """
    score = 0.0
    is_major_event = False

    text_title = title or ""
    text_summary = summary or ""
    text_all = text_title + " " + text_summary

    # 1. 标题情绪词
    for word in POSITIVE_WORDS:
        if word in text_title:
            score += 0.3
    for word in NEGATIVE_WORDS:
        if word in text_title:
            score -= 0.3

    # 2. 摘要情绪词
    for word in POSITIVE_WORDS:
        if word in text_summary:
            score += 0.2
    for word in NEGATIVE_WORDS:
        if word in text_summary:
            score -= 0.2

    # 3. 重大事件词
    for word in MAJOR_EVENT_WORDS:
        if word in text_all:
            is_major_event = True
            break

    # 4. 特殊规则
    if "涨停" in text_title:
        score = 1.0
    if "跌停" in text_title:
        score = -1.0

    # 5. 截断到 [-1, +1]
    score = max(-1.0, min(1.0, score))

    return score, is_major_event


# ============================================================
# 新闻抓取器
# ============================================================

class NewsCrawler:
    """新闻抓取器，封装各数据源"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """带错误处理的 GET 请求"""
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return resp
        except requests.exceptions.Timeout:
            logger.warning("请求超时: %s", url)
        except requests.exceptions.RequestException as e:
            logger.warning("请求失败: %s | %s", url, e)
        return None

    # ---------- 1. 新浪财经个股/基金新闻 ----------
    def fetch_sina_stock_news(self, symbol: str, keyword: str, related_keywords: List[str] = None) -> List[Dict]:
        """抓取新浪财经个股或基金新闻"""
        code = symbol.split(".")[0]
        market = "sz" if symbol.endswith(".SZ") else "sh"

        # 判断是否为ETF（以5或1开头的6位代码通常是ETF/LOF）
        is_etf = code.startswith(("5", "1", "5"))

        if is_etf:
            # ETF使用基金页面
            url = f"https://finance.sina.com.cn/fund/quotes/{code}/bc.shtml"
        else:
            # 个股使用个股新闻页面
            url = f"https://finance.sina.com.cn/realstock/company/{market}{code}/nc.shtml"

        resp = self._get(url)
        if not resp or resp.status_code != 200:
            logger.warning("[新浪财经] %s 抓取失败", symbol)
            return []

        try:
            # 尝试UTF-8和gb2312解码
            for enc in ["utf-8", "gb2312"]:
                try:
                    text = resp.content.decode(enc, errors="ignore")
                    if keyword in text or (not is_etf and len(text) > 1000):
                        break
                except Exception:
                    pass
            else:
                text = resp.content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(text, "html.parser")
        except Exception as e:
            logger.warning("[新浪财经] %s 解析失败: %s", symbol, e)
            return []

        news_items = []
        year = datetime.now().year

        if is_etf:
            # ETF页面：从URL中提取日期，标题在a标签中
            for link in soup.find_all("a"):
                href = link.get("href", "")
                title = link.get_text(strip=True)

                if not title or len(title) < 10 or len(title) > 120:
                    continue

                # 过滤广告/无关内容
                if any(x in title for x in ["SINA English", "意见反馈", "举报中心"]):
                    continue

                # 从URL提取日期: .../2026-05-31/doc-...shtml
                date_match = re.search(r"/(\d{4}-\d{2}-\d{2})/", href)
                if date_match:
                    full_date = date_match.group(1)
                else:
                    continue

                # 关键词匹配：标题或URL中包含关键词或相关词
                all_keywords = [keyword] + (related_keywords or [])
                if not any(kw in title or kw in href for kw in all_keywords):
                    continue

                news_items.append({
                    "symbol": symbol,
                    "date": full_date,
                    "source": "sina_fund",
                    "title": title,
                    "summary": "",
                    "url": href if href.startswith("http") else "",
                })
        else:
            # 个股页面：span 包含日期 (MM-DD)，相邻 a 标签包含标题
            for span in soup.find_all(string=re.compile(r"\(\d{2}-\d{2}\)")):
                date_text = span.strip()
                m = re.match(r"\((\d{2}-\d{2})\)", date_text)
                if not m:
                    continue

                month, day = m.group(1).split("-")
                full_date = f"{year}-{month}-{day}"

                # 向上查找包含 a 标签的父元素
                parent = span.parent
                a_tag = None
                for ancestor in [parent, parent.parent if parent else None,
                                   parent.parent.parent if parent and parent.parent else None]:
                    if ancestor and ancestor.name in ["li", "div", "td", "p"]:
                        a_tag = ancestor.find("a")
                        if a_tag:
                            break

                if not a_tag:
                    continue

                title = a_tag.get_text(strip=True)
                href = a_tag.get("href", "")

                if not title or len(title) < 5:
                    continue

                # 过滤广告/无关内容
                if any(x in title for x in ["level2", "名师团", "主力动向", "股市雷达", "SINA English"]):
                    continue

                news_items.append({
                    "symbol": symbol,
                    "date": full_date,
                    "source": "sina_stock",
                    "title": title,
                    "summary": "",
                    "url": href if href.startswith("http") else "",
                })

        logger.info("[新浪财经] %s 抓取 %d 条", symbol, len(news_items))
        return news_items

    # ---------- 2. 东方财富公告 ----------
    def fetch_eastmoney_announce(self, symbol: str, keyword: str) -> List[Dict]:
        """抓取东方财富公告"""
        code = symbol.split(".")[0]
        url = (
            "https://np-anotice-stock.eastmoney.com/api/security/ann"
            f"?sr=-1&page_size=30&page_index=1&ann_type=A&stock_list={code}"
        )

        resp = self._get(url, headers={**DEFAULT_HEADERS, "Accept": "application/json"})
        if not resp or resp.status_code != 200:
            logger.warning("[东方财富公告] %s 抓取失败", symbol)
            return []

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.warning("[东方财富公告] %s JSON解析失败: %s", symbol, e)
            return []

        news_items = []
        if data.get("data") and data["data"].get("list"):
            for item in data["data"]["list"]:
                title = item.get("title", "")
                notice_date = item.get("notice_date", "")
                # notice_date 格式: "2026-06-02 00:00:00"
                date_only = notice_date.split()[0] if notice_date else ""

                if not title or not date_only:
                    continue

                # 构建公告详情URL
                art_code = item.get("art_code", "")
                detail_url = f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html" if art_code else ""

                news_items.append({
                    "symbol": symbol,
                    "date": date_only,
                    "source": "eastmoney_announce",
                    "title": title,
                    "summary": "",
                    "url": detail_url,
                })

        logger.info("[东方财富公告] %s 抓取 %d 条", symbol, len(news_items))
        return news_items

    # ---------- 3. 财联社电报 (尝试) ----------
    def fetch_cls_telegraph(self, keyword: str) -> List[Dict]:
        """尝试抓取财联社电报，失败返回空列表"""
        url = "https://www.cls.cn/telegraph"
        resp = self._get(url)
        if not resp or resp.status_code != 200:
            logger.warning("[财联社电报] 页面请求失败")
            return []

        # 财联社有WAF/JS渲染，直接解析HTML通常拿不到数据
        # 尝试从页面中提取
        try:
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.warning("[财联社电报] 解析失败: %s", e)
            return []

        news_items = []
        # 财联社电报格式通常是: 时间【标题】内容
        # 但由于是JS渲染，HTML中可能没有这些内容
        # 尝试多种匹配模式
        text = resp.text

        # 模式1: 时间+【关键词】
        pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2})【([^】]*?" + re.escape(keyword) + r"[^】]*?)】(.+?)(?=\d{2}:\d{2}:\d{2}【|$)",
            re.DOTALL,
        )
        matches = pattern.findall(text)

        if not matches:
            # 模式2: 更宽松的匹配
            pattern2 = re.compile(
                r"(\d{2}:\d{2}:\d{2}).*?【(.+?)】(.+?)(?=\d{2}:\d{2}:\d{2}|$)",
                re.DOTALL,
            )
            all_matches = pattern2.findall(text)
            matches = [m for m in all_matches if keyword in m[1] or keyword in m[2]]

        year = datetime.now().year
        month = datetime.now().month
        day = datetime.now().day
        today_str = f"{year}-{month:02d}-{day:02d}"

        for time_str, title, content in matches[:20]:
            title = title.replace("\n", " ").strip()
            content = content.replace("\n", " ").strip()[:200]
            news_items.append({
                "symbol": "",  # 后续根据关键词分配
                "date": today_str,
                "source": "cls_telegraph",
                "title": title,
                "summary": content,
                "url": "https://www.cls.cn/telegraph",
            })

        if news_items:
            logger.info("[财联社电报] 关键词 '%s' 抓取 %d 条", keyword, len(news_items))
        else:
            logger.info("[财联社电报] 未抓取到数据 (可能被WAF拦截或JS渲染)")

        return news_items

    # ---------- 4. 新浪财经首页通用新闻 (作为ETF和个股的补充源) ----------
    def fetch_sina_homepage_news(self, keyword: str, related_keywords: List[str]) -> List[Dict]:
        """抓取新浪财经首页通用新闻，按关键词筛选"""
        url = "https://finance.sina.com.cn/"
        resp = self._get(url)
        if not resp or resp.status_code != 200:
            logger.warning("[新浪财经首页] 抓取失败")
            return []

        try:
            text = resp.content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(text, "html.parser")
        except Exception as e:
            logger.warning("[新浪财经首页] 解析失败: %s", e)
            return []

        news_items = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            title = link.get_text(strip=True)

            if not title or len(title) < 15 or len(title) > 100:
                continue

            # 从URL提取日期: .../2026-05-31/doc-...shtml
            date_match = re.search(r"/(\d{4}-\d{2}-\d{2})/", href)
            if not date_match:
                continue

            full_date = date_match.group(1)

            # 关键词匹配
            all_keywords = [keyword] + related_keywords
            if not any(kw in title or kw in href for kw in all_keywords):
                continue

            # 过滤广告/无关内容
            if any(x in title for x in ["SINA English", "意见反馈", "举报中心", "今日财经要闻TOP10"]):
                continue

            news_items.append({
                "symbol": "",  # 后续分配
                "date": full_date,
                "source": "sina_homepage",
                "title": title,
                "summary": "",
                "url": href if href.startswith("http") else "",
            })

        logger.info("[新浪财经首页] 关键词 '%s' 抓取 %d 条", keyword, len(news_items))
        return news_items


# ============================================================
# 数据聚合
# ============================================================

def aggregate_sentiment(
    symbol: str, news_items: List[Dict], target_dates: List[str]
) -> List[Dict]:
    """
    按 (symbol, date) 聚合情绪数据

    Returns:
        每日聚合记录列表
    """
    from collections import defaultdict

    # 按日期分组
    daily_news = defaultdict(list)
    for item in news_items:
        daily_news[item["date"]].append(item)

    aggregates = []
    for date_str in target_dates:
        # 收集该日期及之前的新闻
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        # 近1日
        d1_start = date_obj
        news_1d = [
            item
            for d, items in daily_news.items()
            if datetime.strptime(d, "%Y-%m-%d") >= d1_start
            for item in items
        ]

        # 近3日
        d3_start = date_obj - timedelta(days=2)
        news_3d = [
            item
            for d, items in daily_news.items()
            if d3_start <= datetime.strptime(d, "%Y-%m-%d") <= date_obj
            for item in items
        ]

        # 近7日
        d7_start = date_obj - timedelta(days=6)
        news_7d = [
            item
            for d, items in daily_news.items()
            if d7_start <= datetime.strptime(d, "%Y-%m-%d") <= date_obj
            for item in items
        ]

        # 计算情绪均值
        scores_1d = [item["sentiment_score"] for item in news_1d]
        scores_3d = [item["sentiment_score"] for item in news_3d]
        scores_7d = [item["sentiment_score"] for item in news_7d]

        sentiment_1d = sum(scores_1d) / len(scores_1d) if scores_1d else 0.0
        sentiment_3d = sum(scores_3d) / len(scores_3d) if scores_3d else 0.0
        sentiment_7d = sum(scores_7d) / len(scores_7d) if scores_7d else 0.0

        # 重大事件计数
        major_1d = sum(1 for item in news_1d if item.get("is_major_event"))
        major_3d = sum(1 for item in news_3d if item.get("is_major_event"))

        # 最新新闻
        latest = None
        if date_str in daily_news and daily_news[date_str]:
            latest = daily_news[date_str][0]
        elif news_1d:
            latest = news_1d[0]

        aggregates.append({
            "symbol": symbol,
            "date": date_str,
            "sentiment_1d": round(sentiment_1d, 4),
            "sentiment_3d": round(sentiment_3d, 4),
            "sentiment_7d": round(sentiment_7d, 4),
            "major_events_1d": major_1d,
            "major_events_3d": major_3d,
            "news_count_1d": len(scores_1d),
            "news_count_3d": len(scores_3d),
            "latest_title": latest["title"] if latest else "",
            "latest_source": latest["source"] if latest else "",
        })

    return aggregates


# ============================================================
# 主运行逻辑
# ============================================================

def get_missing_dates(symbol: str, days_back: int = 7, db_path: str = None) -> List[str]:
    """获取最近N天中缺失数据的日期"""
    today = datetime.now().date()
    all_dates = [
        (today - timedelta(days=i)).isoformat()
        for i in range(days_back)
    ]

    existing = set(get_existing_news_dates(symbol, db_path))
    missing = [d for d in all_dates if d not in existing]
    return missing


def run_crawler(
    symbols: Optional[List[str]] = None,
    days_back: int = 7,
    db_path: str = None,
):
    """
    运行新闻爬虫主流程

    Args:
        symbols: 标的列表，默认全部
        days_back: 回溯天数
        db_path: 数据库路径
    """
    db_path = db_path or str(DB_PATH)
    symbols = symbols or list(SYMBOL_KEYWORDS.keys())

    # 1. 初始化表
    init_news_tables(db_path)

    crawler = NewsCrawler()
    all_raw_news: List[Dict] = []

    # 2. 抓取通用新闻（财联社、新浪首页）
    logger.info("=" * 50)
    logger.info("开始抓取通用新闻源...")
    logger.info("=" * 50)

    # 财联社电报（尝试一次，按关键词分别筛选）
    cls_news_by_keyword: Dict[str, List[Dict]] = {}
    for symbol, (keyword, related_kws) in SYMBOL_KEYWORDS.items():
        cls_items = crawler.fetch_cls_telegraph(keyword)
        if cls_items:
            cls_news_by_keyword[symbol] = cls_items

    # 新浪首页通用新闻（尝试一次，按关键词分别筛选）
    sina_homepage_by_keyword: Dict[str, List[Dict]] = {}
    for symbol, (keyword, related_kws) in SYMBOL_KEYWORDS.items():
        homepage_items = crawler.fetch_sina_homepage_news(keyword, related_kws)
        if homepage_items:
            sina_homepage_by_keyword[symbol] = homepage_items

    # 3. 逐个标的抓取个股新闻
    for symbol in symbols:
        keyword_info = SYMBOL_KEYWORDS.get(symbol)
        if not keyword_info:
            logger.warning("未知标的: %s，跳过", symbol)
            continue
        keyword, related_kws = keyword_info

        logger.info("-" * 40)
        logger.info("处理标的: %s (%s)", symbol, keyword)
        logger.info("-" * 40)

        # 检查缺失日期
        missing_dates = get_missing_dates(symbol, days_back, db_path)
        if not missing_dates:
            logger.info("%s 最近%d天数据已完整，跳过抓取", symbol, days_back)
            continue
        logger.info("缺失日期: %s", ", ".join(missing_dates))

        symbol_news: List[Dict] = []

        # 3.1 新浪财经个股/基金新闻
        try:
            items = crawler.fetch_sina_stock_news(symbol, keyword, related_kws)
            for item in items:
                if item["date"] in missing_dates:
                    score, is_major = analyze_sentiment(item["title"], item.get("summary", ""))
                    item["sentiment_score"] = score
                    item["is_major_event"] = is_major
                    symbol_news.append(item)
        except Exception as e:
            logger.warning("[新浪财经] %s 异常: %s", symbol, e)

        # 3.2 东方财富公告
        try:
            items = crawler.fetch_eastmoney_announce(symbol, keyword)
            for item in items:
                if item["date"] in missing_dates:
                    score, is_major = analyze_sentiment(item["title"], item.get("summary", ""))
                    item["sentiment_score"] = score
                    item["is_major_event"] = is_major
                    symbol_news.append(item)
        except Exception as e:
            logger.warning("[东方财富公告] %s 异常: %s", symbol, e)

        # 3.3 财联社电报（分配symbol）
        if symbol in cls_news_by_keyword:
            for item in cls_news_by_keyword[symbol]:
                item["symbol"] = symbol
                if item["date"] in missing_dates:
                    score, is_major = analyze_sentiment(item["title"], item.get("summary", ""))
                    item["sentiment_score"] = score
                    item["is_major_event"] = is_major
                    symbol_news.append(item)

        # 3.4 新浪首页通用新闻（分配symbol）
        if symbol in sina_homepage_by_keyword:
            for item in sina_homepage_by_keyword[symbol]:
                item["symbol"] = symbol
                if item["date"] in missing_dates:
                    score, is_major = analyze_sentiment(item["title"], item.get("summary", ""))
                    item["sentiment_score"] = score
                    item["is_major_event"] = is_major
                    symbol_news.append(item)

        # 4. 保存原始新闻
        if symbol_news:
            save_raw_news(symbol_news, db_path)
            all_raw_news.extend(symbol_news)

        # 5. 重新加载该标的全部新闻，进行聚合
        with db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT date, title, source, sentiment_score, is_major_event
                FROM news_raw_v4 WHERE symbol = ? ORDER BY date DESC
            """,
                (symbol,),
            )
            rows = cursor.fetchall()

        all_symbol_news = [
            {
                "date": row[0],
                "title": row[1],
                "source": row[2],
                "sentiment_score": row[3] or 0.0,
                "is_major_event": bool(row[4]),
            }
            for row in rows
        ]

        # 聚合目标日期 = 缺失日期 + 最近7天（确保更新）
        target_dates = sorted(
            set(missing_dates + [
                (datetime.now().date() - timedelta(days=i)).isoformat()
                for i in range(days_back)
            ])
        )

        aggregates = aggregate_sentiment(symbol, all_symbol_news, target_dates)
        save_sentiment_aggregates(aggregates, db_path)

    # 6. 汇总报告
    logger.info("=" * 50)
    logger.info("抓取完成，汇总报告")
    logger.info("=" * 50)

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        for symbol in symbols:
            keyword, _ = SYMBOL_KEYWORDS.get(symbol, ("", []))
            cursor.execute(
                "SELECT COUNT(*) FROM news_raw_v4 WHERE symbol = ?",
                (symbol,),
            )
            total_count = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT date, sentiment_1d, major_events_1d, news_count_1d, latest_title
                FROM news_sentiment_v4
                WHERE symbol = ? ORDER BY date DESC LIMIT 1
            """,
                (symbol,),
            )
            row = cursor.fetchone()

            if row:
                logger.info(
                    "%s (%s): 总新闻 %d 条 | 最新 %s | 1日情绪 %.2f | 重大事件 %d | 新闻数 %d | 最新标题: %s",
                    symbol,
                    keyword,
                    total_count,
                    row[0],
                    row[1] or 0.0,
                    row[2] or 0,
                    row[3] or 0,
                    (row[4] or "")[:40],
                )
            else:
                logger.info("%s (%s): 总新闻 %d 条 | 无聚合数据", symbol, keyword, total_count)

    return all_raw_news


def print_sentiment_distribution(db_path: str = None):
    """打印情绪分布统计"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, date, sentiment_1d, sentiment_3d, sentiment_7d,
                   major_events_1d, major_events_3d, news_count_1d, news_count_3d
            FROM news_sentiment_v4
            ORDER BY symbol, date DESC
        """)
        rows = cursor.fetchall()

    if not rows:
        print("暂无情绪聚合数据")
        return

    print("\n" + "=" * 80)
    print("新闻情绪分布报告")
    print("=" * 80)
    print(
        f"{'标的':<12} {'日期':<12} {'1日情绪':<8} {'3日情绪':<8} {'7日情绪':<8} "
        f"{'1日重大':<6} {'3日重大':<6} {'1日数量':<6} {'3日数量':<6}"
    )
    print("-" * 80)

    for row in rows:
        symbol, date, s1, s3, s7, m1, m3, n1, n3 = row
        print(
            f"{symbol:<12} {date:<12} {s1 or 0:>+7.2f} {s3 or 0:>+7.2f} {s7 or 0:>+7.2f} "
            f"{m1 or 0:>5} {m3 or 0:>5} {n1 or 0:>5} {n3 or 0:>5}"
        )

    # 统计各标的新闻总数
    print("\n" + "-" * 80)
    print("各标的新闻数量统计:")
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, COUNT(*) as cnt,
                   AVG(sentiment_score) as avg_score,
                   SUM(is_major_event) as major_cnt
            FROM news_raw_v4
            GROUP BY symbol
            ORDER BY cnt DESC
        """)
        for row in cursor.fetchall():
            symbol, cnt, avg_score, major_cnt = row
            keyword, _ = SYMBOL_KEYWORDS.get(symbol, ("", []))
            print(
                f"  {symbol:<12} ({keyword:<10}): 新闻 {cnt:>3} 条, "
                f"平均情绪 {avg_score or 0:>+.3f}, 重大事件 {major_cnt or 0} 条"
            )

    print("=" * 80)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="新闻情绪爬虫 v4")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help='标的列表，逗号分隔，如 "000002.SZ,000333.SZ"',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="回溯天数，默认7",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DB_PATH),
        help=f"数据库路径，默认 {DB_PATH}",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="只打印报告，不抓取",
    )

    args = parser.parse_args()

    if args.report:
        print_sentiment_distribution(args.db)
    else:
        symbols = args.symbols.split(",") if args.symbols else None
        run_crawler(symbols=symbols, days_back=args.days, db_path=args.db)
        print_sentiment_distribution(args.db)
