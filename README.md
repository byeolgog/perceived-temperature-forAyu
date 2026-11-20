# 오늘춥나?? — 안양대 캠퍼스 체감온도 AI

### 👉 바로 보기: https://byeolgog.github.io/perceived-temperature-forAyu/

2026 안양대학교 AI 경진대회 (AI26-016, 팀 환전)

날씨앱만 믿고 나왔다가 캠퍼스에서 떠는 일이 없도록, **날씨앱 예보(안양시 만안구)를 AI로 안양대 캠퍼스 날씨로 보정**해 체감온도를 알려주고, 캠퍼스에 머무는 시간 중 가장 추운 때에 맞춰 **옷차림을 추천**합니다.

| 날씨앱 값 그대로 쓸 때 | 캠퍼스 AI 보정 후 |
|---|---|
| 캠퍼스 체감온도 ±1°C 적중 56.7% | **72.0%** |
| ±2°C 적중 71.8% | **90.8%** |

<sub>2025년 1년치로 시험 (2023~2024년으로 학습, 정답은 기상청 안양대 500m 격자 값)</sub>

오늘 실측까지 반영하면 가까운 시간은 더 정확해져요 (±1°C 적중, 2025년 시험): 1시간 뒤 **78.9%**, 3시간 뒤 74.6%, 6시간 뒤는 거의 72%로 돌아옴. 겨울(10~4월) 1시간 뒤는 65.2 → 72.8%.

## 어떻게 동작하나요

1. **날씨 수집**: 기상청 만안구 초단기실황(지금 기온)과 단기예보(시간별 기온·습도·풍속·강수확률). 없으면 Open-Meteo
2. **캠퍼스 날씨 (AI)**: 남은 시각은 캠퍼스 보정 모델이 안양대 기온·습도·풍속을 예측, 지난 시각은 안양대 500m 격자 실측값
   - **오늘 실측 반영**: 최근 정시 실측 − (그 시각 예보로 AI가 낸 예측) = 오늘의 오차. 예보가 빗나간 만큼과 AI가 빗나간 만큼이 함께 잡히고, 몇 시간은 같은 방향으로 이어지므로 앞으로의 예측에 `exp(−경과시간/4시간)`만큼 더함 (1시간 뒤 78%, 4시간 뒤 37%, 12시간 뒤 5%)
3. **체감온도 (공식)**: 기상청 공식 그대로 (10~4월 바람: 기온 10°C 이하·풍속 1.3 m/s 이상일 때, 5~9월 습도) + 추위 민감도(±2°C)
4. **옷차림 (AI)**: 시간대별 옷차림 단계를 예측해 가장 두껍게 입어야 하는 시간 기준으로 추천
5. **실시간 갱신**: GitHub Actions가 매시간 안양대 500m 격자 실측값과 기상청 만안구 실황·예보를 받아 사이트에 반영

### 캠퍼스 보정 모델 (Random Forest 회귀)

| 항목 | 내용 |
|---|---|
| 입력 | 만안 기온·습도·풍속, 시간대·월 |
| 정답 | 안양대 500m 격자 기온·습도·풍속 ('안양대 − 만안' 보정량을 정답마다 따로 학습) |
| 학습 데이터 | 안양만안 AWS(377) ↔ 안양대 500m 격자, 2023~2025년 시간별 26,150줄 (`data/train_data.csv`) |
| 크기 | 정답마다 나무 100그루 (서버) / 20그루 축소판 (웹) |
| 한계 | 겨울은 풍속이 1.3 m/s 기준선을 넘나들면 체감온도가 크게 바뀌어 정확도가 낮음 → 기준선 근처는 화면에 범위로 표시 |

고도·바람 배율 같은 추정 계수는 쓰지 않아요. 기온의 고도 차이도 500m 격자 자료에 이미 들어 있어 모델이 데이터로 배웁니다.

### 옷차림 모델 (Random Forest 분류)

캠퍼스 기온·풍속·습도, 개인 체감온도, 시각, 월, 추위 민감도, 강수확률 → 옷차림 8단계(한여름~한겨울).
'기온별 옷차림 표' 규칙 라벨로 시작하고, 학생 제보("이렇게 입었는데 추워요/더워요")로 다시 학습합니다.
현재 정확도(약 82%)는 규칙 라벨 기준이라, 본선 기준(7/10)은 실제 사용자 제보로 다시 재야 합니다.

## 내 컴퓨터에서 실행하기

준비물: **Python 3.9 이상** (설치할 때 "Add python.exe to PATH" 체크)

```powershell
git clone https://github.com/byeolgog/perceived-temperature-forAyu.git
cd perceived-temperature-forAyu
pip install -r requirements.txt
python train_campus.py     # 캠퍼스 모델: 채점 결과 출력 + data/campus_model.joblib
python train.py            # 옷차림 모델
python server.py           # → http://localhost:5000
```

- Git이 없으면 저장소의 **Code → Download ZIP**으로 받아 압축을 풀고, 폴더에서 **우클릭 → 터미널에서 열기**
- 모델 파일(`data/*.joblib`)은 용량이 커서 올리지 않아요. 받은 컴퓨터마다 한 번 학습하세요 (약 20초)
- 설치 없이 보려면 `docs/index.html`을 더블클릭 (웹 체험판: 실시간 파일을 못 읽어 Open-Meteo 예보와 AI 예측 표시)

| 증상 | 해결 |
|---|---|
| `python`을 찾을 수 없음 | PATH 체크하고 재설치, 또는 `py server.py` |
| `No module named ...` | `pip install -r requirements.txt` 다시 실행 |
| `Address already in use` | 이미 켜져 있음 → 브라우저에서 열기 |
| 화면이 예전 그대로 | 서버를 껐다 켜고 `Ctrl + F5` |

### API 키 (선택)

키가 없어도 동작합니다(Open-Meteo·데모 데이터). `.env.example`을 `.env`로 복사해 키를 넣고 서버를 다시 켜세요. **`.env`는 깃허브에 올리거나 공유하지 마세요.**

| 키 | 용도 | 발급처 |
|---|---|---|
| `KMA_API_KEY` | 기상청 단기예보·초단기실황 | data.go.kr '기상청_단기예보 조회서비스' |
| `KMA_APIHUB_KEY` | 위와 같음 + 안양대 500m 격자 실측값 | apihub.kma.go.kr (고해상도 격자자료 활용신청) |
| `NAVER_CLIENT_ID/SECRET` | 추천 옷 예시 이미지 | developers.naver.com 검색 API |
| `OPENAI_API_KEY` | 'AI 한마디' 문장 | platform.openai.com |

## 웹사이트 배포 (GitHub Pages)

`.github/workflows/pages.yml`이 main에 올라올 때와 **매시간** `docs/`를 배포합니다. 배포할 때 저장소 비밀값의 인증키로 안양대 실측값(`campus_now.json`)과 기상청 만안구 실황·단기예보(`forecast.json`)를 받아 함께 올려요. 사이트 파일에는 키가 들어가지 않습니다.

API허브에서 **고해상도 격자자료**와 **단기예보** 둘 다 활용신청되어 있어야 해요. 단기예보 권한이 없으면 사이트는 Open-Meteo 예보로 계산합니다.

처음 한 번만:

1. Settings → Secrets and variables → Actions → **New repository secret**: `KMA_APIHUB_KEY` = API허브 인증키
2. Settings → Pages → Source: **GitHub Actions**
3. Actions → 'Pages (캠퍼스 실측 갱신)' → **Run workflow**

실측값 받기에 실패하면 직전 값을 유지하고 다음 시간에 다시 시도합니다. 6시간 넘게 갱신이 없으면 화면은 AI 예측으로 바뀝니다. 저장소에 60일 동안 변경이 없으면 GitHub이 매시간 실행을 멈추니, Actions 탭에서 다시 켜세요.

## 파일 구조

| 파일 | 역할 |
|---|---|
| `campus_model.py` | 캠퍼스 보정 모델 + 기상청 체감온도 공식 |
| `train_campus.py` | 캠퍼스 모델 학습·채점 (`--data`, `--test-year`) |
| `feels_like.py` | 캠퍼스 체감 계산: 실측 또는 AI 예측 → 기상청 공식 → 개인 민감도 |
| `weather.py` | 예보 수집(기상청·Open-Meteo·데모) + 안양대 500m 격자 실측 |
| `model.py`, `train.py`, `outfit.py` | 옷차림 모델과 옷차림 8단계 표 |
| `service.py`, `server.py` | 화면용 계산 묶음, Flask 서버 + API |
| `shop.py` | 추천 옷 예시 (네이버 쇼핑 검색) |
| `config.py` | 좌표, 경로, 민감도 등 설정 |
| `docs/` | 화면 (`index.html`), 웹 체험판 로직 (`standalone.js`), 웹용 모델 (`model.js`) |
| `web_export.py` | 웹용 모델 만들기: `config.py`·`outfit.py`·학습 데이터를 바꾸면 다시 실행 |
| `update_live_weather.py` | 안양대 실측 → `docs/campus_now.json`, 만안구 실황·예보 → `docs/forecast.json` (Actions가 매시간 실행) |
| `data/train_data.csv` | 캠퍼스 모델 학습 데이터 |
| `notebooks/` | 학습 데이터를 만들고 실험한 Colab 노트북 (아래 참고) |

## 학습 데이터 다시 만들기 (`notebooks/`)

| 순서 | 노트북 | 하는 일 |
|---|---|---|
| 1 | `1_campus_grid_collect.ipynb` | API허브 500m 격자(특정지점 다중요소, 37.3921, 126.9206)로 안양대 시간별 값 수집 → 구글 드라이브 |
| 2 | `2_aws_377_clean.ipynb` | 기상자료개방포털에서 받은 안양만안(377) AWS 시간자료 CSV 정리 |
| 3 | `3_merge_train_data.ipynb` | 같은 시각끼리 합쳐 `train_data.csv` 만들기 |
| 4 | `4_train_campus_model.ipynb` | 모델 비교(평균 편차·선형 회귀·랜덤포레스트), 적중률 채점 |

새 `train_data.csv`를 `data/`에 넣고 `python train_campus.py` → `python web_export.py`.

## 앞으로

- **Air Fit 모델로 옷차림 분류 교체** (예선 자료 6쪽, MIT 라이선스): `model.py`의 `make_model()`·`train()`·`predict_level()`만 바꾸면 됩니다. 사용 시 출처를 남겨야 해요.
- **실제 사용자 테스트**: 학생 5명 체감 제보로 본선 기준(7/10) 확인

## 시연 팁

- 주소 끝에 `#analyze`를 붙이면 바로 분석 화면부터 시작 (녹화용)
- '하교 시간 알림'은 브라우저 알림이라 탭이 열려 있어야 울려요
- 로그인은 '사용자 정보 수집 안 함' 범위에 따라 넣지 않았어요
