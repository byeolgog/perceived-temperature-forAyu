"""추천 옷 → 쇼핑 사이트 연계.

· 네이버 쇼핑 검색 API(NAVER_CLIENT_ID / NAVER_CLIENT_SECRET)가 있으면 실제 상품
  (사진·가격·구매 링크)을 보여줌 — developers.naver.com 에서 무료 발급, 하루 25,000건
· 키가 없어도 무신사·지그재그·네이버쇼핑 검색 결과 링크는 항상 제공
"""
from __future__ import annotations

import os
import re
from urllib.parse import quote

import requests

SEARCH_LINKS = {
    "무신사": "https://www.musinsa.com/search/goods?keyword={q}",
    "지그재그": "https://zigzag.kr/search?keyword={q}",
    "네이버쇼핑": "https://search.shopping.naver.com/search/all?query={q}",
}


def search_links(item: str) -> dict:
    return {site: url.format(q=quote(item)) for site, url in SEARCH_LINKS.items()}


def naver_enabled() -> bool:
    return bool(os.environ.get("NAVER_CLIENT_ID") and os.environ.get("NAVER_CLIENT_SECRET"))


def naver_products(item: str, n: int = 4) -> list:
    """네이버 쇼핑 검색 결과 [{title, price, image, link, mall}]."""
    resp = requests.get(
        "https://openapi.naver.com/v1/search/shop.json",
        params={"query": item, "display": n, "sort": "sim"},
        headers={
            "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
            "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
        },
        timeout=10,
    )
    resp.raise_for_status()
    return [
        {
            "title": re.sub(r"<[^>]+>", "", it["title"]),  # <b> 강조 태그 제거
            "price": int(it["lprice"]) if it.get("lprice") else None,
            "image": it["image"],
            "link": it["link"],
            "mall": it.get("mallName", ""),
        }
        for it in resp.json().get("items", [])
    ]
