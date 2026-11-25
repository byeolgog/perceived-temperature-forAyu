"""옷차림 단계 정의 (기온별 옷차림 표 기반, 0=가장 가볍게 ~ 7=가장 두껍게).

단계마다 상의 / 하의 / 신발 / 의류 잡화 4개 카테고리로 추천합니다.
"""
from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ("상의", "하의", "신발", "의류 잡화")
CATEGORY_EMOJI = {"상의": "🧥", "하의": "👖", "신발": "👟", "의류 잡화": "🧣"}


@dataclass(frozen=True)
class OutfitLevel:
    level: int
    name: str
    min_feels: float  # 체감온도가 이 값 이상이면 이 단계
    top: tuple
    bottom: tuple
    shoes: tuple
    acc: tuple
    emoji: str

    @property
    def categories(self) -> dict:
        return dict(zip(CATEGORIES, (self.top, self.bottom, self.shoes, self.acc)))

    @property
    def items(self) -> tuple:
        """전체 아이템 (상의부터 순서대로)."""
        return self.top + self.bottom + self.shoes + self.acc


# 체감온도가 min_feels 이상이면 해당 단계 (위에서부터 검사)
LEVELS = [
    OutfitLevel(0, "한여름", 28,
                ("민소매", "반팔 티셔츠"), ("반바지", "린넨 팬츠"), ("샌들", "슬리퍼"), ("모자", "선글라스", "양산"), "🩳"),
    OutfitLevel(1, "더움", 23,
                ("반팔 티셔츠", "얇은 셔츠"), ("반바지", "면바지"), ("스니커즈", "샌들"), ("모자", "선글라스"), "👕"),
    OutfitLevel(2, "따뜻함", 20,
                ("긴팔 티셔츠", "얇은 가디건"), ("면바지", "청바지"), ("스니커즈", "로퍼"), ("캡모자", "에코백"), "👚"),
    OutfitLevel(3, "선선함", 17,
                ("맨투맨", "얇은 니트", "가디건"), ("청바지", "슬랙스"), ("스니커즈", "로퍼"), ("얇은 스카프",), "🧶"),
    OutfitLevel(4, "쌀쌀함", 12,
                ("자켓", "야상", "니트"), ("청바지", "슬랙스"), ("스니커즈", "첼시부츠"), ("스카프",), "🧥"),
    OutfitLevel(5, "추움", 9,
                ("트렌치코트", "자켓", "니트"), ("청바지", "스타킹"), ("앵클부츠", "스니커즈"), ("스카프", "얇은 목도리"), "🧥"),
    OutfitLevel(6, "많이 추움", 5,
                ("코트", "가죽자켓", "히트텍"), ("기모 바지", "두꺼운 청바지"), ("워커", "앵클부츠"), ("목도리", "비니"), "🧣"),
    OutfitLevel(7, "한겨울", -99,
                ("롱패딩", "히트텍", "니트"), ("기모 바지", "내복"), ("털부츠", "방한화"), ("목도리", "장갑", "핫팩", "귀마개"), "🧤"),
]

RAIN_EXTRA = {"신발": ("레인부츠",), "의류 잡화": ("우산",)}

ITEM_EMOJI = {
    "민소매": "🎽", "반팔 티셔츠": "👕", "얇은 셔츠": "👔", "긴팔 티셔츠": "👚", "얇은 가디건": "🧥",
    "맨투맨": "👕", "얇은 니트": "🧶", "가디건": "🧥", "자켓": "🧥", "야상": "🧥", "니트": "🧶",
    "트렌치코트": "🧥", "코트": "🧥", "가죽자켓": "🧥", "히트텍": "🌡️", "롱패딩": "🧥",
    "반바지": "🩳", "린넨 팬츠": "👖", "면바지": "👖", "청바지": "👖", "슬랙스": "👖", "스타킹": "🧦",
    "기모 바지": "👖", "두꺼운 청바지": "👖", "내복": "🩲",
    "샌들": "🩴", "슬리퍼": "🩴", "스니커즈": "👟", "로퍼": "👞", "첼시부츠": "👢", "앵클부츠": "👢",
    "워커": "🥾", "털부츠": "👢", "방한화": "🥾", "레인부츠": "👢",
    "모자": "👒", "선글라스": "🕶️", "양산": "⛱️", "캡모자": "🧢", "에코백": "👜", "얇은 스카프": "🧣",
    "스카프": "🧣", "얇은 목도리": "🧣", "목도리": "🧣", "비니": "🧶", "장갑": "🧤", "핫팩": "🔥",
    "귀마개": "🎧", "우산": "☂️",
}

FEEDBACK_CHOICES = ("추워요", "적당해요", "더워요")


def recommend(level: int, rain: bool = False) -> dict:
    """카테고리별 추천 {상의: [...], 하의: [...], 신발: [...], 의류 잡화: [...]}."""
    cats = {k: list(v) for k, v in get(level).categories.items()}
    if rain:
        for k, extra in RAIN_EXTRA.items():
            cats[k] = list(extra) + [i for i in cats[k] if i not in ("샌들", "슬리퍼")]
    return cats


def rule_level(feels_c: float) -> int:
    """체감온도 → 옷차림 단계 (규칙 기반, 모델 학습의 초기 라벨로도 사용)."""
    for lv in LEVELS:
        if feels_c >= lv.min_feels:
            return lv.level
    return LEVELS[-1].level


def corrected_level(worn_level: int, feedback: str) -> int:
    """'이렇게 입었더니 추웠다/더웠다' 제보 → 실제로 맞았어야 할 단계."""
    if feedback == "추워요":
        return min(worn_level + 1, len(LEVELS) - 1)
    if feedback == "더워요":
        return max(worn_level - 1, 0)
    return worn_level


def get(level: int) -> OutfitLevel:
    return LEVELS[max(0, min(level, len(LEVELS) - 1))]
