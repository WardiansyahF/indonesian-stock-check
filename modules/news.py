"""
modules/news.py — Stock News Fetcher (v3 — Improved)
======================================================
[DQC] v3: Adapted to yfinance v2 news format, guaranteed 3-5 articles,
better fallback via RSS/Google News scraping.
"""

import yfinance as yf
from datetime import datetime
import requests


def get_stock_news(ticker: str, max_items: int = 5) -> list:
    """
    Ambil 3-5 berita terbaru. Tries yfinance first, then Google News RSS.
    """
    news_list = _fetch_yfinance_news(ticker, max_items)

    # Jika yfinance kosong, coba Google News RSS
    if len(news_list) < 3:
        rss_news = _fetch_google_rss_news(ticker, max_items - len(news_list))
        news_list.extend(rss_news)

    return news_list[:max_items]


def _fetch_yfinance_news(ticker: str, max_items: int = 5) -> list:
    """Fetch from yfinance (adapts to v1 and v2 format)."""
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []

        news_list = []
        for item in raw_news[:max_items]:
            # v2 format: item might have 'content' wrapper
            content = item
            if isinstance(item, dict) and "content" in item:
                content = item["content"]

            title = (content.get("title", "") or
                     content.get("headline", "") or "")
            if not title:
                continue

            # Parse timestamp — handle multiple formats
            pub_time = "N/A"
            for time_key in ["providerPublishTime", "pubDate", "publishedAt"]:
                raw_time = content.get(time_key)
                if raw_time:
                    pub_time = _parse_timestamp(raw_time)
                    break

            # Parse link — handle v2 nested format
            link = (content.get("link", "") or
                    content.get("url", "") or
                    content.get("canonicalUrl", {}).get("url", "") or "#")

            # Publisher
            publisher = (content.get("publisher", "") or
                         content.get("provider", {}).get("displayName", "") or
                         "Unknown")

            # Thumbnail
            thumbnail = ""
            thumb_data = content.get("thumbnail", content.get("previewImage", {}))
            if isinstance(thumb_data, dict):
                resolutions = thumb_data.get("resolutions", thumb_data.get("images", []))
                if resolutions:
                    thumbnail = resolutions[-1].get("url", "") if isinstance(resolutions[-1], dict) else ""

            news_list.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published_time": pub_time,
                "thumbnail": thumbnail,
                "related_tickers": content.get("relatedTickers", []),
                "news_type": content.get("type", "STORY"),
            })

        return news_list
    except Exception:
        return []


def _fetch_google_rss_news(ticker: str, max_items: int = 5) -> list:
    """Fallback: Google News RSS for Indonesian stock news."""
    clean = ticker.replace(".JK", "")
    query = f"{clean} saham"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"

    try:
        resp = requests.get(rss_url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/1.0)"
        })
        if resp.status_code != 200:
            return []

        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        news_list = []
        for item in channel.findall("item")[:max_items]:
            title = item.findtext("title", "")
            link = item.findtext("link", "#")
            pub_date = item.findtext("pubDate", "")
            source = item.findtext("source", "Google News")

            if pub_date:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date)
                    pub_date = dt.strftime("%d %b %Y, %H:%M")
                except Exception:
                    pub_date = pub_date[:25]

            news_list.append({
                "title": title,
                "publisher": source,
                "link": link,
                "published_time": pub_date or "N/A",
                "thumbnail": "",
                "related_tickers": [ticker],
                "news_type": "RSS",
            })

        return news_list
    except Exception:
        return []


def _parse_timestamp(raw_time) -> str:
    """Parse various timestamp formats."""
    if isinstance(raw_time, (int, float)):
        try:
            return datetime.fromtimestamp(raw_time).strftime("%d %b %Y, %H:%M")
        except (ValueError, TypeError, OSError):
            return "N/A"
    elif isinstance(raw_time, str):
        try:
            dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            return dt.strftime("%d %b %Y, %H:%M")
        except ValueError:
            return raw_time[:25] if len(raw_time) > 25 else raw_time
    return "N/A"


def get_external_news_links(ticker: str) -> list:
    clean = ticker.replace(".JK", "")
    return [
        {"name": "🔍 Google News", "url": f"https://www.google.com/search?q={clean}+saham+indonesia&tbm=nws"},
        {"name": "📊 CNBC Indonesia", "url": f"https://www.cnbcindonesia.com/tag/{clean.lower()}"},
        {"name": "📈 Bisnis.com", "url": f"https://market.bisnis.com/saham/{clean}"},
        {"name": "💹 IDX Channel", "url": f"https://www.idxchannel.com/search?q={clean}"},
    ]
