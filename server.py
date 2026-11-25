"""오늘춥나?? 웹 서버.

실행: python server.py  →  http://localhost:5000
"""
from functools import lru_cache

from flask import Flask, jsonify, request, send_from_directory

import config
from feels_like import deployed_model_status
import outfit
import service
import shop

# 화면 파일은 docs/ (GitHub Pages와 같은 폴더) — model.js 등을 루트 경로로 제공
app = Flask(__name__, static_folder="docs", static_url_path="")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/home")
def api_home():
    return jsonify(service.home())


@app.get("/api/model-status")
def api_model_status():
    """검증 1위 체크포인트가 로컬호스트 앱에 연결됐는지 확인한다."""
    return jsonify(deployed_model_status())


@app.post("/api/analyze")
def api_analyze():
    body = request.get_json(force=True)
    return jsonify(service.analyze(body.get("day", "오늘"), body.get("schedule", ""), body.get("sensitivity", "보통")))


@app.post("/api/feedback")
def api_feedback():
    body = request.get_json(force=True)
    fb = body.get("feedback")
    if fb not in outfit.FEEDBACK_CHOICES:
        return jsonify({"error": "잘못된 제보"}), 400
    sens = body.get("sensitivity", "보통")
    if sens not in config.SENSITIVITY_OFFSET:
        sens = "보통"
    level = max(0, min(int(body.get("worn_level", 0)), len(outfit.LEVELS) - 1))
    service.save_feedback(level, fb, sens)
    return jsonify(service.feedback_stats())


@lru_cache(maxsize=256)
def _products(item: str) -> tuple:
    return tuple(shop.naver_products(item))


@app.get("/api/products")
def api_products():
    item = request.args.get("item", "")[:30]
    products = []
    if shop.naver_enabled():
        try:
            products = list(_products(item))
        except Exception:
            products = []
    return jsonify({"products": products, "links": shop.search_links(item)})


if __name__ == "__main__":
    print("오늘춥나?? → http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
