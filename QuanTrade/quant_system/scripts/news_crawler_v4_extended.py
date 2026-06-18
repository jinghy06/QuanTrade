# -*- coding: utf-8 -*-
"""
新闻情绪爬虫 v4 扩展版
==================
获取更多历史新闻数据来训练量化模型

数据源:
  1. 东方财富新闻搜索API (主要, JSON API, 历史数据丰富)
  2. 新浪财经滚动新闻 (辅助, 按日期分页)
  3. 财联社电报 (尝试 requests, 失败则跳过)
  4. 同花顺财经 (尝试)

技术要求:
  - 保存到 news_raw_v4_extended 表
  - 字段: symbol, date, source, title, summary, sentiment_score, is_major_event, url
  - time.sleep(1) 控制频率
  - 每个源失败时打印警告并继续

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
# Windows 下强制 UTF-8 输出，避免 GBK 编码错误
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("news_crawler_v4_extended")

# ---------- 标的-关键词映射 ----------
SYMBOL_KEYWORDS: Dict[str, Tuple[str, List[str]]] = {
    "562500.SH": ("中证A500", ["A500", "中证500", "沪深300", "大盘", "宽基", "沪指", "深成指", "A股", "指数"]),
    "588200.SH": ("科创芯片", ["芯片", "半导体", "科创", "集成电路", "晶圆", "光刻"]),
    "588790.SH": ("科创AI", ["AI", "人工智能", "科创", "大模型", "算力", "机器学习"]),
    "159382.SZ": ("创业板人工智能", ["人工智能", "AI", "创业板", "大模型", "算力"]),
    "159241.SZ": ("创业板新能源", ["新能源", "光伏", "储能", "创业板", "风电", "锂电"]),
    "000002.SZ": ("万科A", ["万科", "房地产", "地产", "房企", "楼市", "住建部"]),
    "000333.SZ": ("美的集团", ["美的", "家电", "智能家居", "白色家电", "空调"]),
    "000568.SZ": ("泸州老窖", ["泸州老窖", "白酒", "酒业", "茅台", "五粮液", "名酒"]),
    "000651.SZ": ("格力电器", ["格力", "家电", "空调", "白色家电", "董明珠"]),
}

# ---------- 情绪词典 ----------
POSITIVE_WORDS = [
    "涨停", "大涨", "反弹", "利好", "突破", "创新高", "强劲", "复苏",
    "增持", "买入", "推荐", "超预期", "盈利增长", "景气", "繁荣",
    "上涨", "攀升", "飙升", "回暖", "改善", "优化", "扩张", "增长",
    "净利润增长", "营收增长", "订单饱满", "供不应求", "龙头", "领先",
    "净流入", "资金抢筹", "配置价值", "上行", "韧性", "看好",
]

NEGATIVE_WORDS = [
    "跌停", "大跌", "暴跌", "利空", "跌破", "创新低", "疲软", "衰退",
    "减持", "卖出", "回避", "低于预期", "亏损", "下滑", "萎缩",
    "风险", "警示", "下跌", "下挫", "回落", "恶化", "收缩", "下降",
    "净利润下滑", "营收下降", "订单减少", "产能过剩", "落后", "承压",
    "债务违约", "资金链断裂", "裁员", "停产", "亏损扩大", "业绩暴雷",
    "净流出", "资金流出", "下行", "调整", "震荡调整", "探底",
]

MAJOR_EVENT_WORDS = [
    "政策", "监管", "制裁", "重组", "并购", "退市", "停牌",
    "财报", "业绩", "分红", "配股", "定增", "减持", "增持",
    "股东大会", "董事会", "关联交易", "担保", "诉讼", "仲裁",
    "立案调查", "行政处罚", "问询函", "关注函", "警示函",
    "央行", "美联储", "降准", "降息", "加息", "汇率", "关税",
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
    "Referer": "https://www.eastmoney.com/",
}

REQUEST_TIMEOUT = 20
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
    """初始化新闻扩展表结构（幂等）"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 扩展原始新闻表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_raw_v4_extended (
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

        # 索引优化
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_ext_symbol_date ON news_raw_v4_extended(symbol, date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_ext_source ON news_raw_v4_extended(source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_ext_date ON news_raw_v4_extended(date)"
        )

        conn.commit()
        logger.info("新闻扩展表初始化完成: %s", db_path)


def get_existing_news_dates(symbol: str, source: str = None, db_path: str = None) -> List[str]:
    """获取某标的已有新闻的日期列表"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        if source:
            cursor.execute(
                "SELECT DISTINCT date FROM news_raw_v4_extended WHERE symbol = ? AND source = ? ORDER BY date DESC",
                (symbol, source),
            )
        else:
            cursor.execute(
                "SELECT DISTINCT date FROM news_raw_v4_extended WHERE symbol = ? ORDER BY date DESC",
                (symbol,),
            )
        return [row[0] for row in cursor.fetchall()]


def get_existing_titles(symbol: str, source: str, db_path: str = None) -> set:
    """获取某标的某源已有新闻的标题集合（用于去重）"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title FROM news_raw_v4_extended WHERE symbol = ? AND source = ?",
            (symbol, source),
        )
        return {row[0] for row in cursor.fetchall()}


def save_raw_news(news_items: List[Dict], db_path: str = None):
    """批量保存原始新闻（INSERT OR IGNORE，避免重复）"""
    if not news_items:
        return 0

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        inserted = 0
        for item in news_items:
            # 使用 symbol + source + title + date 作为去重键
            cursor.execute(
                """
                SELECT id FROM news_raw_v4_extended
                WHERE symbol = ? AND source = ? AND title = ? AND date = ?
                LIMIT 1
            """,
                (item["symbol"], item["source"], item["title"], item["date"]),
            )
            if cursor.fetchone():
                continue

            cursor.execute(
                """
                INSERT INTO news_raw_v4_extended
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
    return inserted


def get_news_stats(db_path: str = None) -> Dict:
    """获取新闻统计信息"""
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM news_raw_v4_extended")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM news_raw_v4_extended")
        symbol_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT date) FROM news_raw_v4_extended")
        date_count = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM news_raw_v4_extended GROUP BY source")
        source_stats = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT MIN(date), MAX(date) FROM news_raw_v4_extended")
        min_date, max_date = cursor.fetchone()

    return {
        "total": total,
        "symbol_count": symbol_count,
        "date_count": date_count,
        "source_stats": source_stats,
        "date_range": (min_date, max_date),
    }


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

class NewsCrawlerExtended:
    """扩展新闻抓取器，封装各数据源"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.results = {
            "eastmoney": {"success": False, "count": 0, "error": ""},
            "sina_roll": {"success": False, "count": 0, "error": ""},
            "cls": {"success": False, "count": 0, "error": ""},
            "10jqka": {"success": False, "count": 0, "error": ""},
        }

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

    # ---------- 1. 东方财富新闻搜索API (主要数据源) ----------
    def fetch_eastmoney_news(self, keyword: str, symbol: str, count: int = 50) -> List[Dict]:
        """
        通过东方财富搜索API获取新闻
        API: https://searchapi.eastmoney.com/api/suggest/get?input=KEYWORD&type=20&count=N
        返回JSON格式新闻数据
        """
        url = (
            f"https://searchapi.eastmoney.com/api/suggest/get"
            f"?input={requests.utils.quote(keyword)}&type=20&count={count}"
        )

        resp = self._get(url, headers={**DEFAULT_HEADERS, "Accept": "application/json"})
        if not resp or resp.status_code != 200:
            self.results["eastmoney"]["error"] = f"HTTP {resp.status_code if resp else 'No response'}"
            logger.warning("[东方财富新闻] %s 请求失败", keyword)
            return []

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            self.results["eastmoney"]["error"] = f"JSON解析失败: {e}"
            logger.warning("[东方财富新闻] %s JSON解析失败: %s", keyword, e)
            return []

        news_items = []
        cms_data = data.get("CMSArticle", {})
        articles = cms_data.get("Data", []) if isinstance(cms_data, dict) else []

        if not articles:
            logger.info("[东方财富新闻] %s 无返回数据", keyword)
            return []

        for item in articles:
            title = item.get("title", "")
            content = item.get("content", "")
            media = item.get("mediaName", "")
            date_str = item.get("date", "")
            code = item.get("code", "")

            if not title or not date_str:
                continue

            # 清理标题中的HTML标签
            title_clean = re.sub(r"<[^>]+>", "", title)
            content_clean = re.sub(r"<[^>]+>", "", content)

            # 解析日期: "2026-06-04 15:28:32" -> "2026-06-04"
            date_only = date_str.split()[0] if " " in date_str else date_str

            # 构建URL
            url_detail = f"https://finance.eastmoney.com/a/{code}.html" if code else ""

            score, is_major = analyze_sentiment(title_clean, content_clean)

            news_items.append({
                "symbol": symbol,
                "date": date_only,
                "source": "eastmoney_news",
                "title": title_clean,
                "summary": content_clean[:300],
                "sentiment_score": score,
                "is_major_event": is_major,
                "url": url_detail,
            })

        self.results["eastmoney"]["success"] = True
        self.results["eastmoney"]["count"] += len(news_items)
        logger.info("[东方财富新闻] 关键词 '%s' -> 标的 %s 抓取 %d 条", keyword, symbol, len(news_items))
        return news_items

    # ---------- 2. 新浪财经滚动新闻 (按日期/分页) ----------
    def fetch_sina_roll_news(self, page: int = 1) -> List[Dict]:
        """
        抓取新浪财经滚动新闻
        尝试多种URL模式
        """
        # 尝试多个可能的URL模式
        urls_to_try = [
            f"https://finance.sina.com.cn/stock/roll/index.d.html?page={page}",
            f"https://finance.sina.com.cn/roll/index.d.html?page={page}",
            f"https://finance.sina.com.cn/stock/roll/2026-06-0{page}.shtml" if page <= 7 else "",
        ]
        urls_to_try = [u for u in urls_to_try if u]

        for url in urls_to_try:
            resp = self._get(url)
            if not resp or resp.status_code != 200:
                continue

            try:
                # 尝试多种编码
                for enc in ["utf-8", "gb2312", "gbk"]:
                    try:
                        text = resp.content.decode(enc, errors="ignore")
                        if len(text) > 500:
                            break
                    except Exception:
                        continue
                else:
                    text = resp.content.decode("utf-8", errors="ignore")

                soup = BeautifulSoup(text, "html.parser")
            except Exception as e:
                logger.warning("[新浪财经滚动] 解析失败: %s", e)
                continue

            news_items = []
            year = datetime.now().year

            # 模式1: 从链接中提取日期
            for link in soup.find_all("a"):
                href = link.get("href", "")
                title = link.get_text(strip=True)

                if not title or len(title) < 10 or len(title) > 120:
                    continue

                # 过滤广告/无关内容
                if any(x in title for x in ["SINA English", "意见反馈", "举报中心", "今日财经要闻TOP10", "客服热线"]):
                    continue

                # 从URL提取日期: .../2026-05-31/doc-...shtml
                date_match = re.search(r"/(\d{4}-\d{2}-\d{2})/", href)
                if date_match:
                    full_date = date_match.group(1)
                else:
                    # 尝试其他日期格式
                    date_match2 = re.search(r"/(\d{4})(\d{2})(\d{2})/", href)
                    if date_match2:
                        full_date = f"{date_match2.group(1)}-{date_match2.group(2)}-{date_match2.group(3)}"
                    else:
                        continue

                # 只保留最近60天的新闻
                try:
                    news_date = datetime.strptime(full_date, "%Y-%m-%d").date()
                    if (datetime.now().date() - news_date).days > 60:
                        continue
                except ValueError:
                    continue

                score, is_major = analyze_sentiment(title)

                news_items.append({
                    "symbol": "",  # 后续根据关键词分配
                    "date": full_date,
                    "source": "sina_roll",
                    "title": title,
                    "summary": "",
                    "sentiment_score": score,
                    "is_major_event": is_major,
                    "url": href if href.startswith("http") else "",
                })

            if news_items:
                self.results["sina_roll"]["success"] = True
                self.results["sina_roll"]["count"] += len(news_items)
                logger.info("[新浪财经滚动] 第%d页 抓取 %d 条", page, len(news_items))
                return news_items

        self.results["sina_roll"]["error"] = "所有URL模式均失败"
        logger.warning("[新浪财经滚动] 第%d页 所有URL模式均失败", page)
        return []

    # ---------- 3. 财联社电报 (requests尝试, 预期被WAF拦截) ----------
    def fetch_cls_telegraph(self, keyword: str, symbol: str) -> List[Dict]:
        """
        尝试用requests抓取财联社电报
        财联社有WAF/JS渲染保护，requests通常无法获取内容
        """
        url = "https://www.cls.cn/telegraph"
        resp = self._get(url)
        if not resp or resp.status_code != 200:
            self.results["cls"]["error"] = f"HTTP {resp.status_code if resp else 'No response'}"
            logger.warning("[财联社电报] 页面请求失败 (可能被WAF拦截)")
            return []

        try:
            resp.encoding = "utf-8"
            text = resp.text
            soup = BeautifulSoup(text, "html.parser")
        except Exception as e:
            self.results["cls"]["error"] = f"解析失败: {e}"
            logger.warning("[财联社电报] 解析失败: %s", e)
            return []

        news_items = []
        # 财联社电报格式: 时间【标题】内容
        # 但由于JS渲染，HTML中通常没有这些内容
        # 尝试从纯文本中匹配

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
            score, is_major = analyze_sentiment(title, content)

            news_items.append({
                "symbol": symbol,
                "date": today_str,
                "source": "cls_telegraph",
                "title": title,
                "summary": content,
                "sentiment_score": score,
                "is_major_event": is_major,
                "url": "https://www.cls.cn/telegraph",
            })

        if news_items:
            self.results["cls"]["success"] = True
            self.results["cls"]["count"] += len(news_items)
            logger.info("[财联社电报] 关键词 '%s' 抓取 %d 条", keyword, len(news_items))
        else:
            self.results["cls"]["error"] = "未匹配到数据 (WAF/JS渲染)"
            logger.info("[财联社电报] 未抓取到数据 (可能被WAF拦截或JS渲染)")

        return news_items

    # ---------- 4. 同花顺财经 (尝试) ----------
    def fetch_10jqka_news(self, keyword: str, symbol: str) -> List[Dict]:
        """
        尝试抓取同花顺财经新闻
        """
        # 同花顺搜索URL
        url = f"https://search.10jqka.com.cn/search?wd={requests.utils.quote(keyword)}"

        resp = self._get(url)
        if not resp or resp.status_code != 200:
            self.results["10jqka"]["error"] = f"HTTP {resp.status_code if resp else 'No response'}"
            logger.warning("[同花顺] %s 请求失败", keyword)
            return []

        try:
            text = resp.content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(text, "html.parser")
        except Exception as e:
            self.results["10jqka"]["error"] = f"解析失败: {e}"
            logger.warning("[同花顺] %s 解析失败: %s", keyword, e)
            return []

        news_items = []
        year = datetime.now().year

        # 尝试多种选择器
        for article in soup.find_all(["div", "li", "article"], class_=re.compile(r"news|article|item|list")):
            a_tag = article.find("a")
            if not a_tag:
                continue

            title = a_tag.get_text(strip=True)
            href = a_tag.get("href", "")

            if not title or len(title) < 10:
                continue

            # 尝试从URL或文本中提取日期
            date_match = re.search(r"/(\d{4}-\d{2}-\d{2})/", href)
            if date_match:
                full_date = date_match.group(1)
            else:
                # 从文本中找日期
                date_text = article.get_text()
                date_match2 = re.search(r"(\d{4}-\d{2}-\d{2})", date_text)
                if date_match2:
                    full_date = date_match2.group(1)
                else:
                    continue

            score, is_major = analyze_sentiment(title)

            news_items.append({
                "symbol": symbol,
                "date": full_date,
                "source": "10jqka",
                "title": title,
                "summary": "",
                "sentiment_score": score,
                "is_major_event": is_major,
                "url": href if href.startswith("http") else f"https://search.10jqka.com.cn{href}",
            })

        if news_items:
            self.results["10jqka"]["success"] = True
            self.results["10jqka"]["count"] += len(news_items)
            logger.info("[同花顺] 关键词 '%s' 抓取 %d 条", keyword, len(news_items))
        else:
            self.results["10jqka"]["error"] = "未匹配到数据"
            logger.info("[同花顺] 关键词 '%s' 未抓取到数据", keyword)

        return news_items

    # ---------- 5. 东方财富公告 (原有功能保留) ----------
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
                date_only = notice_date.split()[0] if notice_date else ""

                if not title or not date_only:
                    continue

                art_code = item.get("art_code", "")
                detail_url = f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html" if art_code else ""

                score, is_major = analyze_sentiment(title)

                news_items.append({
                    "symbol": symbol,
                    "date": date_only,
                    "source": "eastmoney_announce",
                    "title": title,
                    "summary": "",
                    "sentiment_score": score,
                    "is_major_event": is_major,
                    "url": detail_url,
                })

        logger.info("[东方财富公告] %s 抓取 %d 条", symbol, len(news_items))
        return news_items

    # ---------- 6. 新浪财经个股/基金新闻 (原有功能保留) ----------
    def fetch_sina_stock_news(self, symbol: str, keyword: str, related_keywords: List[str] = None) -> List[Dict]:
        """抓取新浪财经个股或基金新闻"""
        code = symbol.split(".")[0]
        market = "sz" if symbol.endswith(".SZ") else "sh"
        is_etf = code.startswith(("5", "1", "5"))

        if is_etf:
            url = f"https://finance.sina.com.cn/fund/quotes/{code}/bc.shtml"
        else:
            url = f"https://finance.sina.com.cn/realstock/company/{market}{code}/nc.shtml"

        resp = self._get(url)
        if not resp or resp.status_code != 200:
            logger.warning("[新浪财经个股] %s 抓取失败", symbol)
            return []

        try:
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
            logger.warning("[新浪财经个股] %s 解析失败: %s", symbol, e)
            return []

        news_items = []
        year = datetime.now().year

        if is_etf:
            for link in soup.find_all("a"):
                href = link.get("href", "")
                title = link.get_text(strip=True)

                if not title or len(title) < 10 or len(title) > 120:
                    continue
                if any(x in title for x in ["SINA English", "意见反馈", "举报中心"]):
                    continue

                date_match = re.search(r"/(\d{4}-\d{2}-\d{2})/", href)
                if not date_match:
                    continue
                full_date = date_match.group(1)

                all_keywords = [keyword] + (related_keywords or [])
                if not any(kw in title or kw in href for kw in all_keywords):
                    continue

                score, is_major = analyze_sentiment(title)

                news_items.append({
                    "symbol": symbol,
                    "date": full_date,
                    "source": "sina_stock",
                    "title": title,
                    "summary": "",
                    "sentiment_score": score,
                    "is_major_event": is_major,
                    "url": href if href.startswith("http") else "",
                })
        else:
            for span in soup.find_all(string=re.compile(r"\(\d{2}-\d{2}\)")):
                date_text = span.strip()
                m = re.match(r"\((\d{2}-\d{2})\)", date_text)
                if not m:
                    continue

                month, day = m.group(1).split("-")
                full_date = f"{year}-{month}-{day}"

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
                if any(x in title for x in ["level2", "名师团", "主力动向", "股市雷达", "SINA English"]):
                    continue

                score, is_major = analyze_sentiment(title)

                news_items.append({
                    "symbol": symbol,
                    "date": full_date,
                    "source": "sina_stock",
                    "title": title,
                    "summary": "",
                    "sentiment_score": score,
                    "is_major_event": is_major,
                    "url": href if href.startswith("http") else "",
                })

        logger.info("[新浪财经个股] %s 抓取 %d 条", symbol, len(news_items))
        return news_items


# ============================================================
# 主运行逻辑
# ============================================================

def run_crawler_extended(
    symbols: Optional[List[str]] = None,
    days_back: int = 60,
    db_path: str = None,
):
    """
    运行扩展新闻爬虫主流程

    Args:
        symbols: 标的列表，默认全部
        days_back: 回溯天数，默认60
        db_path: 数据库路径
    """
    db_path = db_path or str(DB_PATH)
    symbols = symbols or list(SYMBOL_KEYWORDS.keys())

    # 1. 初始化表
    init_news_tables(db_path)

    crawler = NewsCrawlerExtended()
    total_inserted = 0
    all_news_items: List[Dict] = []

    logger.info("=" * 60)
    logger.info("开始运行扩展新闻爬虫 (目标: %d 天历史数据)", days_back)
    logger.info("=" * 60)

    # 2. 逐个标的抓取
    for symbol in symbols:
        keyword_info = SYMBOL_KEYWORDS.get(symbol)
        if not keyword_info:
            logger.warning("未知标的: %s，跳过", symbol)
            continue
        keyword, related_kws = keyword_info

        logger.info("-" * 50)
        logger.info("处理标的: %s (%s)", symbol, keyword)
        logger.info("-" * 50)

        symbol_news: List[Dict] = []

        # 2.1 东方财富新闻搜索API (主要数据源)
        try:
            items = crawler.fetch_eastmoney_news(keyword, symbol, count=50)
            for item in items:
                symbol_news.append(item)
        except Exception as e:
            logger.warning("[东方财富新闻] %s 异常: %s", symbol, e)

        # 2.2 东方财富公告
        try:
            items = crawler.fetch_eastmoney_announce(symbol, keyword)
            for item in items:
                symbol_news.append(item)
        except Exception as e:
            logger.warning("[东方财富公告] %s 异常: %s", symbol, e)

        # 2.3 新浪财经个股/基金新闻
        try:
            items = crawler.fetch_sina_stock_news(symbol, keyword, related_kws)
            for item in items:
                symbol_news.append(item)
        except Exception as e:
            logger.warning("[新浪财经个股] %s 异常: %s", symbol, e)

        # 2.4 财联社电报 (requests尝试)
        try:
            items = crawler.fetch_cls_telegraph(keyword, symbol)
            for item in items:
                symbol_news.append(item)
        except Exception as e:
            logger.warning("[财联社电报] %s 异常: %s", symbol, e)

        # 2.5 同花顺
        try:
            items = crawler.fetch_10jqka_news(keyword, symbol)
            for item in items:
                symbol_news.append(item)
        except Exception as e:
            logger.warning("[同花顺] %s 异常: %s", symbol, e)

        # 3. 保存该标的的新闻
        if symbol_news:
            inserted = save_raw_news(symbol_news, db_path)
            total_inserted += inserted
            all_news_items.extend(symbol_news)

    # 4. 抓取通用滚动新闻（新浪首页）
    logger.info("-" * 50)
    logger.info("抓取通用滚动新闻...")
    logger.info("-" * 50)

    for page in range(1, 6):
        try:
            items = crawler.fetch_sina_roll_news(page)
            if items:
                # 为滚动新闻分配symbol（根据关键词匹配）
                for item in items:
                    for symbol, (keyword, related_kws) in SYMBOL_KEYWORDS.items():
                        all_kws = [keyword] + related_kws
                        if any(kw in item["title"] for kw in all_kws):
                            item["symbol"] = symbol
                            break
                # 只保存匹配到symbol的
                matched_items = [it for it in items if it["symbol"]]
                if matched_items:
                    inserted = save_raw_news(matched_items, db_path)
                    total_inserted += inserted
                    all_news_items.extend(matched_items)
        except Exception as e:
            logger.warning("[新浪财经滚动] 第%d页 异常: %s", page, e)

    # 5. 汇总报告
    logger.info("=" * 60)
    logger.info("抓取完成，汇总报告")
    logger.info("=" * 60)

    stats = get_news_stats(db_path)
    logger.info("数据库统计:")
    logger.info("  总新闻数: %d", stats["total"])
    logger.info("  标的覆盖: %d 个", stats["symbol_count"])
    logger.info("  日期覆盖: %d 天", stats["date_count"])
    logger.info("  日期范围: %s ~ %s", stats["date_range"][0] or "N/A", stats["date_range"][1] or "N/A")
    logger.info("  各源统计:")
    for source, count in stats["source_stats"].items():
        logger.info("    %s: %d 条", source, count)

    logger.info("-" * 50)
    logger.info("各源抓取结果:")
    for source, result in crawler.results.items():
        status = "[OK] 成功" if result["success"] else "[FAIL] 失败"
        logger.info("  %s: %s | 抓取 %d 条 | %s", source, status, result["count"], result["error"])

    logger.info("-" * 50)
    logger.info("本次新增: %d 条", total_inserted)

    return all_news_items, crawler.results


def print_news_report(db_path: str = None):
    """打印新闻分布报告"""
    stats = get_news_stats(db_path)

    print("\n" + "=" * 80)
    print("新闻数据分布报告")
    print("=" * 80)
    print(f"总新闻数: {stats['total']}")
    print(f"标的覆盖: {stats['symbol_count']} 个")
    print(f"日期覆盖: {stats['date_count']} 天")
    print(f"日期范围: {stats['date_range'][0] or 'N/A'} ~ {stats['date_range'][1] or 'N/A'}")
    print("-" * 80)
    print("各数据源统计:")
    for source, count in sorted(stats["source_stats"].items(), key=lambda x: -x[1]):
        print(f"  {source:<25}: {count:>5} 条")

    print("-" * 80)
    print("各标的新闻统计:")
    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, COUNT(*) as cnt,
                   AVG(sentiment_score) as avg_score,
                   SUM(is_major_event) as major_cnt,
                   MIN(date) as min_date,
                   MAX(date) as max_date
            FROM news_raw_v4_extended
            GROUP BY symbol
            ORDER BY cnt DESC
        """)
        for row in cursor.fetchall():
            symbol, cnt, avg_score, major_cnt, min_date, max_date = row
            keyword, _ = SYMBOL_KEYWORDS.get(symbol, ("", []))
            print(
                f"  {symbol:<12} ({keyword:<12}): 新闻 {cnt:>4} 条, "
                f"平均情绪 {avg_score or 0:>+.3f}, 重大事件 {major_cnt or 0:>3} 条, "
                f"日期 {min_date or 'N/A'} ~ {max_date or 'N/A'}"
            )

    print("=" * 80)


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="新闻情绪爬虫 v4 扩展版")
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help='标的列表，逗号分隔，如 "000002.SZ,000333.SZ"',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="回溯天数，默认60",
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
        print_news_report(args.db)
    else:
        symbols = args.symbols.split(",") if args.symbols else None
        run_crawler_extended(symbols=symbols, days_back=args.days, db_path=args.db)
        print_news_report(args.db)
